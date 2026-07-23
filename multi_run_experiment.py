# ============================================
# Multi-Run Augmentation Ratios Experiment
# 3 runs per ratio -> mean +/- std with error bars
# ============================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import matplotlib.pyplot as plt
import time
import json
import cv2
import gc

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import MalGAN.load_balanced_data as load_balanced_data

# ============================================
# CONFIGURATION
# ============================================
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'
GENERATOR_PATH = f'{PROJECT_DIR}/gan_training/checkpoints/generator_final.h5'
LATENT_DIM = 100
CLASS_NAMES = ['Androm', 'Elex', 'Expiro', 'HackKMS', 'Hlux', 'Sality']
NUM_CLASSES = len(CLASS_NAMES)
AUGMENTATION_RATIOS = [0, 25, 50, 100, 200]
N_RUNS = 3
EPOCHS = 50
BATCH_SIZE = 32
RESULTS_DIR = f'{PROJECT_DIR}/multi_run_results'


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(num_classes):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_generator():
    if not os.path.exists(GENERATOR_PATH):
        raise FileNotFoundError(f"Generator not found at {GENERATOR_PATH}")
    return tf.keras.models.load_model(GENERATOR_PATH, compile=False)


def generate_all_synth(generator, num_per_class):
    all_synth = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        images = []
        remaining = num_per_class
        while remaining > 0:
            batch = min(32, remaining)
            noise = tf.random.normal(shape=(batch, LATENT_DIM))
            labels = tf.convert_to_tensor(np.full((batch, 1), class_idx), dtype=tf.int32)
            gen_imgs = generator([noise, labels], training=False).numpy()
            for i in range(batch):
                img_64 = gen_imgs[i]
                img_uint8 = ((img_64 + 1.0) * 127.5).astype(np.uint8)
                img_224 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(img_224)
            remaining -= batch
        all_synth[class_idx] = np.array(images)
    return all_synth


def subsample_synth(all_synth, num_per_class, seed_offset):
    if num_per_class == 0:
        return np.empty((0, 224, 224, 3), dtype=np.uint8), np.empty((0,), dtype=np.int32)
    rng = np.random.RandomState(seed_offset)
    X_aug, y_aug = [], []
    for class_idx in range(NUM_CLASSES):
        pool = all_synth[class_idx]
        indices = rng.choice(len(pool), size=num_per_class, replace=False)
        X_aug.append(pool[indices])
        y_aug.append(np.full(num_per_class, class_idx))
    return np.concatenate(X_aug), np.concatenate(y_aug)


def train_one_run(X_train_norm, y_train, X_val_norm, y_val, seed):
    set_seed(seed)
    model = build_model(NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
    ]

    history = model.fit(
        X_train_norm, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=0
    )

    val_accs = [float(x) for x in history.history['val_accuracy']]
    val_losses = [float(x) for x in history.history['val_loss']]
    best_idx = np.argmax(val_accs)

    return {
        'best_val_acc': val_accs[best_idx],
        'best_val_loss': val_losses[best_idx],
        'best_epoch': int(best_idx + 1),
        'val_accuracy': val_accs,
        'val_loss': val_losses,
    }


def plot_results(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ratios = AUGMENTATION_RATIOS

    means = []
    stds = []
    for r in all_results:
        accs = [run['best_val_acc'] * 100 for run in r['runs']]
        means.append(np.mean(accs))
        stds.append(np.std(accs))

    # Print table
    print("\n" + "=" * 70)
    print("MULTI-RUN RESULTS (mean +/- std, N=3)")
    print("=" * 70)
    print(f"{'Synth/class':<15}{'Mean Acc':<12}{'Std':<10}{'Min':<10}{'Max':<10}")
    print("-" * 70)
    for i, ratio in enumerate(ratios):
        accs = [run['best_val_acc'] * 100 for run in all_results[i]['runs']]
        print(f"{ratio:<15}{means[i]:>8.2f}%  {stds[i]:>6.2f}%  {min(accs):>6.2f}%  {max(accs):>6.2f}%")

    best_ratio = ratios[np.argmax(means)]
    gain = means[np.argmax(means)] - means[0]
    print(f"\nBest: {best_ratio} synth/class at {max(means):.2f}% (+{gain:.2f}% over baseline)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#404040', '#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']

    # Accuracy bar chart with error bars
    x = np.arange(len(ratios))
    bars = axes[0].bar(x, means, yerr=stds, color=colors, edgecolor='black',
                       linewidth=1.0, capsize=6, error_kw={'linewidth': 1.5})
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(r) for r in ratios])
    axes[0].set_xlabel('Synthetic Images per Class', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Accuracy vs Augmentation ({N_RUNS} runs, mean +/- std)', fontsize=13, fontweight='bold')
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.15,
                     f'{mean:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Learning curve with confidence band (average across runs per ratio)
    for i, ratio in enumerate(ratios):
        all_curves = [run['val_accuracy'] for run in all_results[i]['runs']]
        min_len = min(len(c) for c in all_curves)
        curves_trimmed = [c[:min_len] for c in all_curves]
        curves_arr = np.array(curves_trimmed)
        mean_curve = np.mean(curves_arr, axis=0)
        std_curve = np.std(curves_arr, axis=0)
        epochs_range = np.arange(1, min_len + 1)

        axes[1].plot(epochs_range, mean_curve, linewidth=1.8, color=colors[i],
                     label=f'{ratio} synth/class')
        axes[1].fill_between(epochs_range, mean_curve - std_curve, mean_curve + std_curve,
                             color=colors[i], alpha=0.15)

    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('Learning Curves (mean +/- std)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'multi_run_summary.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to: {plot_path}")

    # Save JSON
    summary = {
        'ratios': ratios,
        'n_runs': N_RUNS,
        'means': means,
        'stds': stds,
        'per_run': {str(r): [run['best_val_acc'] for run in res['runs']]
                    for r, res in zip(ratios, all_results)}
    }
    json_path = os.path.join(RESULTS_DIR, 'multi_run_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved to: {json_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    progress_path = os.path.join(RESULTS_DIR, '_checkpoint.json')

    # Load checkpoint if resuming
    all_results = []
    completed_runs = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            cp = json.load(f)
        completed_runs = {tuple(r) for r in cp.get('completed', [])}
        for saved in cp.get('results', []):
            all_results.append({
                'ratio': saved['ratio'],
                'runs': saved['runs'],
            })
        print(f"Resuming: {len(completed_runs)} runs done, {len(all_results)} ratios loaded")

    print("=" * 70)
    print(f"MULTI-RUN EXPERIMENT: {N_RUNS} runs x {len(AUGMENTATION_RATIOS)} ratios")
    print(f"Ratios: {AUGMENTATION_RATIOS}")
    print("=" * 70)

    # Data
    print("\n[1/3] Loading original balanced dataset...")
    X_train_orig, y_train_orig, X_val, y_val, _ = load_balanced_data.create_balanced_subset()
    X_val_norm = X_val.astype('float32') / 255.0
    print(f"  Train: {X_train_orig.shape}, Val: {X_val.shape}")

    # Generator
    print("\n[2/3] Loading generator + generating 200 synth/class...")
    generator = load_generator()
    all_synth = generate_all_synth(generator, max(AUGMENTATION_RATIOS))
    print(f"  Done: {sum(v.shape[0] for v in all_synth.values())} total images")

    # Train
    total_runs = N_RUNS * len(AUGMENTATION_RATIOS)
    todo = total_runs - len(completed_runs)
    print(f"\n[3/3] {todo} training runs remaining...")

    for ratio_idx, ratio in enumerate(AUGMENTATION_RATIOS):
        if ratio_idx < len(all_results):
            ratio_results = all_results[ratio_idx]
        else:
            ratio_results = {'ratio': ratio, 'runs': []}
            all_results.append(ratio_results)

        for run_id in range(1, N_RUNS + 1):
            run_key = [ratio, run_id]
            if tuple(run_key) in completed_runs:
                continue

            seed = ratio * 100 + run_id
            done_so_far = len(completed_runs)
            print(f"\n{'#' * 50}")
            print(f"[{done_so_far + 1}/{total_runs}] Ratio={ratio}, Run={run_id}/{N_RUNS}")
            print(f"{'#' * 50}")

            if ratio == 0:
                X_train, y_train = X_train_orig, y_train_orig
            else:
                X_aug, y_aug = subsample_synth(all_synth, ratio, seed)
                X_train = np.concatenate([X_train_orig, X_aug], axis=0)
                y_train = np.concatenate([y_train_orig, y_aug], axis=0)
                perm = np.random.permutation(len(X_train))
                X_train, y_train = X_train[perm], y_train[perm]

            X_train_norm = X_train.astype('float32') / 255.0

            t0 = time.time()
            result = train_one_run(X_train_norm, y_train, X_val_norm, y_val, seed)
            elapsed = time.time() - t0

            ratio_results['runs'].append(result)
            completed_runs.add(tuple(run_key))

            # Save checkpoint
            cp_data = {
                'completed': [list(r) for r in completed_runs],
                'results': [{
                    'ratio': r['ratio'],
                    'runs': r['runs'],
                } for r in all_results]
            }
            with open(progress_path, 'w') as f:
                json.dump(cp_data, f, indent=2)

            print(f"  acc={result['best_val_acc']:.4f} loss={result['best_val_loss']:.4f} "
                  f"ep={result['best_epoch']} | {elapsed:.1f}s")

            # Release memory after each run
            del X_train, y_train, X_train_norm, result
            tf.keras.backend.clear_session()
            gc.collect()

    if todo == 0:
        print("\nAll runs already completed. Generating plots from checkpoint...")

    plot_results(all_results)
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
