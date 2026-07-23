"""
MalGAN -- Multi-Run Augmentation Experiment
============================================
Runs *N* independent training runs per augmentation ratio to produce
mean ± std results with error bars (3 runs x 5 ratios by default).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from config import (
    EXPERIMENTS_DIR,
    GAN_DIR,
    LATENT_DIM,
    SELECTED_FAMILIES,
    AUGMENTATION_RATIOS,
    N_MULTI_RUNS,
    NUM_CLASSES,
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_EARLY_STOPPING_PATIENCE,
    CNN_LR_PATIENCE,
    CNN_MIN_LR,
    CNN_IMG_SIZE,
    ensure_dir,
)
from load_balanced_data import create_balanced_subset
from cnn_arch import get_compiled_model

# ---------------------------------------------------------------------------
MULTI_RUN_DIR = EXPERIMENTS_DIR / "multi_run"


# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_generator():
    path = str(GAN_DIR / "checkpoints" / "generator_final.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generator not found at {path}")
    return tf.keras.models.load_model(path, compile=False)


def generate_all_synth(generator, num_per_class):
    all_synth = {}
    for class_idx, class_name in enumerate(SELECTED_FAMILIES):
        images = []
        remaining = num_per_class
        while remaining > 0:
            batch = min(32, remaining)
            noise = tf.random.normal(shape=(batch, LATENT_DIM))
            labels = tf.convert_to_tensor(
                np.full((batch, 1), class_idx), dtype=tf.int32)
            gen_imgs = generator([noise, labels], training=False).numpy()
            for i in range(batch):
                img = ((gen_imgs[i] + 1.0) * 127.5).astype(np.uint8)
                img_224 = cv2.resize(img, CNN_IMG_SIZE, interpolation=cv2.INTER_CUBIC)
                images.append(img_224)
            remaining -= batch
        all_synth[class_idx] = np.array(images)
    return all_synth


def subsample_synth(all_synth, num_per_class, seed_offset):
    if num_per_class == 0:
        return (np.empty((0, *CNN_IMG_SIZE, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32))
    rng = np.random.RandomState(seed_offset)
    X_aug, y_aug = [], []
    for class_idx in range(NUM_CLASSES):
        pool = all_synth[class_idx]
        idx = rng.choice(len(pool), size=num_per_class, replace=False)
        X_aug.append(pool[idx])
        y_aug.append(np.full(num_per_class, class_idx))
    return np.concatenate(X_aug), np.concatenate(y_aug)


def train_one_run(X_train_norm, y_train, X_val_norm, y_val, seed):
    set_seed(seed)
    model = get_compiled_model("resnet", NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=CNN_EARLY_STOPPING_PATIENCE,
                      mode="max", restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=CNN_LR_PATIENCE, min_lr=CNN_MIN_LR, verbose=0),
    ]

    history = model.fit(
        X_train_norm, y_train,
        batch_size=CNN_BATCH_SIZE,
        epochs=CNN_EPOCHS,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=0,
    )

    val_accs = [float(x) for x in history.history["val_accuracy"]]
    val_losses = [float(x) for x in history.history["val_loss"]]
    best_idx = np.argmax(val_accs)

    return {
        "best_val_acc": val_accs[best_idx],
        "best_val_loss": val_losses[best_idx],
        "best_epoch": int(best_idx + 1),
        "val_accuracy": val_accs,
        "val_loss": val_losses,
    }


# ---------------------------------------------------------------------------
def plot_results(all_results):
    ensure_dir(MULTI_RUN_DIR)
    ratios = AUGMENTATION_RATIOS

    means, stds = [], []
    for ratio_results in all_results:
        accs = [run["best_val_acc"] * 100 for run in ratio_results["runs"]]
        means.append(np.mean(accs))
        stds.append(np.std(accs))

    # Table
    print(f"\n{'=' * 70}")
    print(f"MULTI-RUN RESULTS (mean +/- std, N={N_MULTI_RUNS})")
    print(f"{'=' * 70}")
    print(f"{'Synth/class':<15}{'Mean Acc':<12}{'Std':<10}{'Min':<10}{'Max':<10}")
    print("-" * 70)
    for i, ratio in enumerate(ratios):
        accs = [run["best_val_acc"] * 100 for run in all_results[i]["runs"]]
        print(f"{ratio:<15}{means[i]:>8.2f}%  {stds[i]:>6.2f}%  "
              f"{min(accs):>6.2f}%  {max(accs):>6.2f}%")

    best_idx = np.argmax(means)
    gain = means[best_idx] - means[0]
    print(f"\nBest: {ratios[best_idx]} synth/class @ {means[best_idx]:.2f}% "
          f"(+{gain:.2f}% over baseline)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#404040", "#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]

    x = np.arange(len(ratios))
    bars = axes[0].bar(x, means, yerr=stds, color=colors, edgecolor="black",
                       linewidth=1.0, capsize=6, error_kw={"linewidth": 1.5})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(r) for r in ratios])
    axes[0].set_xlabel("Synthetic Images per Class", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Accuracy vs Augmentation ({N_MULTI_RUNS} runs, mean +/- std)",
                      fontsize=13, fontweight="bold")
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.15,
                     f"{mean:.2f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Learning curves with std bands
    for i, ratio in enumerate(ratios):
        all_curves = [run["val_accuracy"] for run in all_results[i]["runs"]]
        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        epochs_range = np.arange(1, min_len + 1)

        axes[1].plot(epochs_range, mean_curve, linewidth=1.8, color=colors[i],
                     label=f"{ratio} synth/class")
        axes[1].fill_between(epochs_range, mean_curve - std_curve,
                             mean_curve + std_curve, color=colors[i], alpha=0.15)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Validation Accuracy", fontsize=11, fontweight="bold")
    axes[1].set_title("Learning Curves (mean +/- std)", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = MULTI_RUN_DIR / "multi_run_summary.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {plot_path}")

    # JSON summary
    summary = {
        "ratios": ratios,
        "n_runs": N_MULTI_RUNS,
        "means": means,
        "stds": stds,
        "per_run": {
            str(r): [run["best_val_acc"] for run in res["runs"]]
            for r, res in zip(ratios, all_results)
        },
    }
    json_path = MULTI_RUN_DIR / "multi_run_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved: {json_path}")


# ---------------------------------------------------------------------------
def main():
    ensure_dir(MULTI_RUN_DIR)
    progress_path = MULTI_RUN_DIR / "_checkpoint.json"

    all_results = []
    completed_runs = set()

    if progress_path.exists():
        cp = json.loads(progress_path.read_text())
        completed_runs = {tuple(r) for r in cp.get("completed", [])}
        for saved in cp.get("results", []):
            all_results.append({"ratio": saved["ratio"], "runs": saved["runs"]})
        print(f"Resuming: {len(completed_runs)} runs done, "
              f"{len(all_results)} ratios loaded")

    print("=" * 70)
    print(f"MULTI-RUN EXPERIMENT: {N_MULTI_RUNS} runs x {len(AUGMENTATION_RATIOS)} ratios")
    print(f"Ratios: {AUGMENTATION_RATIOS}")
    print("=" * 70)

    # Data
    print("\n[1/3] Loading original balanced dataset ...")
    X_train_orig, y_train_orig, X_val, y_val, _ = create_balanced_subset()
    X_val_norm = X_val.astype("float32") / 255.0

    # Generator + synth pool
    print("\n[2/3] Loading generator + generating synth pool ...")
    generator = load_generator()
    all_synth = generate_all_synth(generator, max(AUGMENTATION_RATIOS))
    print(f"  Total: {sum(v.shape[0] for v in all_synth.values())} images")

    # Train
    total_runs = N_MULTI_RUNS * len(AUGMENTATION_RATIOS)
    remaining = total_runs - len(completed_runs)
    print(f"\n[3/3] {remaining} training runs remaining ...")

    for ratio_idx, ratio in enumerate(AUGMENTATION_RATIOS):
        if ratio_idx < len(all_results):
            ratio_results = all_results[ratio_idx]
        else:
            ratio_results = {"ratio": ratio, "runs": []}
            all_results.append(ratio_results)

        for run_id in range(1, N_MULTI_RUNS + 1):
            run_key = (ratio, run_id)
            if run_key in completed_runs:
                continue

            seed = ratio * 100 + run_id
            done = len(completed_runs)
            print(f"\n{'#' * 50}")
            print(f"[{done + 1}/{total_runs}] Ratio={ratio}, Run={run_id}/{N_MULTI_RUNS}")
            print(f"{'#' * 50}")

            if ratio == 0:
                X_train, y_train = X_train_orig, y_train_orig
            else:
                X_aug, y_aug = subsample_synth(all_synth, ratio, seed)
                X_train = np.concatenate([X_train_orig, X_aug], axis=0)
                y_train = np.concatenate([y_train_orig, y_aug], axis=0)
                perm = np.random.permutation(len(X_train))
                X_train, y_train = X_train[perm], y_train[perm]

            X_train_norm = X_train.astype("float32") / 255.0

            t0 = time.time()
            result = train_one_run(X_train_norm, y_train, X_val_norm, y_val, seed)
            elapsed = time.time() - t0

            ratio_results["runs"].append(result)
            completed_runs.add(run_key)

            # Persist checkpoint
            cp_data = {
                "completed": [list(r) for r in completed_runs],
                "results": [{
                    "ratio": r["ratio"],
                    "runs": r["runs"],
                } for r in all_results],
            }
            progress_path.write_text(json.dumps(cp_data, indent=2))

            print(f"  acc={result['best_val_acc']:.4f} "
                  f"loss={result['best_val_loss']:.4f} "
                  f"ep={result['best_epoch']}  |  {elapsed:.1f}s")

            del X_train, y_train, X_train_norm, result
            tf.keras.backend.clear_session()
            gc.collect()

    if remaining == 0:
        print("\nAll runs already completed. Generating plots ...")

    plot_results(all_results)
    print("\n" + "=" * 70)
    print("MULTI-RUN EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
