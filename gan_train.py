"""
MalGAN -- Conditional DCGAN Training
=====================================
Full GAN training loop with adaptive D/G ratio, label smoothing, gradient
clipping, LR decay, and NaN detection.
"""

import json
import time
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from config import (
    GAN_DIR,
    GAN_EPOCHS,
    GAN_BATCH_SIZE,
    GAN_INITIAL_LR,
    GAN_MIN_LR,
    GAN_LR_DECAY_EPOCHS,
    GAN_LABEL_SMOOTH_REAL,
    GAN_LABEL_SMOOTH_FAKE,
    GAN_D_TARGET_ACC_MIN,
    GAN_D_TARGET_ACC_MAX,
    GAN_RUNNING_AVG_WINDOW,
    GAN_SAVE_INTERVAL,
    GAN_CHECKPOINT_INTERVAL,
    GAN_CLIPNORM,
    GAN_INPUT_SHAPE,
    LATENT_DIM,
    NUM_CLASSES,
    ensure_dir,
)
from load_balanced_data import create_balanced_subset
from gan_arch import build_generator, build_discriminator, build_gan


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def normalize_images(images):
    """[0,255] or [0,1] -> [-1,1] (matches tanh generator output)."""
    images = tf.cast(images, tf.float32)
    if images.shape[-1] is not None and tf.reduce_max(images) > 1.0:
        return (images / 127.5) - 1.0
    return images * 2.0 - 1.0


def denormalize_images(images):
    """[-1,1] -> [0,1]."""
    return (images + 1.0) / 2.0


def get_real_labels(batch_size, smooth=True):
    val = GAN_LABEL_SMOOTH_REAL if smooth else 1.0
    return np.ones((batch_size, 1), dtype=np.float32) * val


def get_fake_labels(batch_size, smooth=True):
    val = GAN_LABEL_SMOOTH_FAKE if smooth else 0.0
    return np.ones((batch_size, 1), dtype=np.float32) * val


def generate_noise(batch_size, latent_dim=LATENT_DIM):
    return tf.convert_to_tensor(
        np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32))


def generate_class_labels(batch_size, num_classes=NUM_CLASSES):
    return tf.convert_to_tensor(
        np.random.randint(0, num_classes, (batch_size, 1)), dtype=tf.int32)


def check_for_nan(*losses):
    for loss in losses:
        if isinstance(loss, (list, tuple)):
            if any(np.isnan(x) for x in loss):
                return True
        elif np.isnan(loss):
            return True
    return False


def update_learning_rate(epoch, optimizer, initial_lr, min_lr, decay_after, total_epochs):
    if epoch < decay_after:
        return initial_lr
    decay_per = (initial_lr - min_lr) / (total_epochs - decay_after)
    new_lr = max(min_lr, initial_lr - decay_per * (epoch - decay_after))
    optimizer.learning_rate.assign(new_lr)
    return new_lr


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def save_generated_images(generator, epoch, latent_dim, num_classes,
                          class_names, save_dir):
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    noise = generate_noise(num_classes, latent_dim)
    labels = tf.convert_to_tensor(np.arange(num_classes).reshape(-1, 1), dtype=tf.int32)
    gen_imgs = generator([noise, labels], training=False)

    for i in range(num_classes):
        img = denormalize_images(gen_imgs[i].numpy())
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[i], fontsize=10)
        axes[i].axis("off")

    plt.suptitle(f"Generated Images - Epoch {epoch}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = f"{save_dir}/generated_epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(history, save_dir):
    if not history["d_loss"]:
        return
    epochs_range = range(1, len(history["d_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(epochs_range, history["d_loss"], "o-", label="D Loss", linewidth=2, markersize=3)
    axes[0, 0].plot(epochs_range, history["g_loss"], "s-", label="G Loss", linewidth=2, markersize=3)
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("GAN Losses"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs_range, history["d_acc"], "o-", color="green", linewidth=2, markersize=3)
    axes[0, 1].axhline(GAN_D_TARGET_ACC_MIN, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].axhline(GAN_D_TARGET_ACC_MAX, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].axhspan(GAN_D_TARGET_ACC_MIN, GAN_D_TARGET_ACC_MAX, alpha=0.2, color="green")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Discriminator Accuracy"); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs_range, history["learning_rates"], "o-", color="purple", linewidth=2, markersize=3)
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_title("Learning Rate Decay"); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale("log")

    axes[1, 1].plot(epochs_range, history["epoch_times"], "o-", color="orange", linewidth=2, markersize=3)
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].set_title("Epoch Duration"); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    return path


def plot_final_samples(generator, num_classes, class_names, save_dir):
    fig, axes = plt.subplots(num_classes, 5, figsize=(15, num_classes * 3))
    for class_idx in range(num_classes):
        noise = generate_noise(5, LATENT_DIM)
        labels = tf.convert_to_tensor(np.full((5, 1), class_idx), dtype=tf.int32)
        gen_imgs = generator([noise, labels], training=False)
        for sample_idx in range(5):
            img = denormalize_images(gen_imgs[sample_idx].numpy())
            img = np.clip(img, 0, 1)
            axes[class_idx, sample_idx].imshow(img)
            axes[class_idx, sample_idx].axis("off")
            if sample_idx == 0:
                axes[class_idx, sample_idx].set_ylabel(class_names[class_idx],
                                                       fontsize=12, fontweight="bold",
                                                       rotation=0, ha="right", va="center")
    plt.suptitle("Final Generated Samples (5 per class)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/final_samples_grid.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    return path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_gan():
    ensure_dir(GAN_DIR)
    samples_dir = ensure_dir(GAN_DIR / "samples")
    checkpoints_dir = ensure_dir(GAN_DIR / "checkpoints")

    # -- Load & resize data ------------------------------------------------
    print("=" * 80)
    print("LOADING & PREPARING DATA (64x64)")
    print("=" * 80)

    X_train, y_train, _, _, class_names = create_balanced_subset()

    if X_train.shape[1] == 224:
        print(f"Resizing from {X_train.shape[1]}x{X_train.shape[2]} -> 64x64 ...")
        resized = [cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                   for img in X_train]
        X_train = np.array(resized)

    print(f"Train shape: {X_train.shape}  Classes: {len(class_names)}")

    # -- tf.data pipeline --------------------------------------------------
    dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(lambda x, y: (normalize_images(x), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(GAN_BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # -- Build models ------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUILDING GAN MODELS")
    print("=" * 80)

    generator = build_generator(LATENT_DIM, NUM_CLASSES)
    discriminator = build_discriminator(GAN_INPUT_SHAPE, NUM_CLASSES)

    discriminator.compile(
        optimizer=Adam(learning_rate=GAN_INITIAL_LR, beta_1=0.5,
                       clipnorm=GAN_CLIPNORM),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    gan = build_gan(generator, discriminator, LATENT_DIM)
    gan.compile(
        optimizer=Adam(learning_rate=GAN_INITIAL_LR, beta_1=0.5,
                       clipnorm=GAN_CLIPNORM),
        loss="binary_crossentropy",
    )

    print(f"Generator:      {sum(tf.size(w).numpy() for w in generator.weights):,} params")
    print(f"Discriminator:  {sum(tf.size(w).numpy() for w in discriminator.weights):,} params")

    # -- Training state ----------------------------------------------------
    history = {
        "d_loss": [], "d_acc": [], "g_loss": [],
        "epoch_times": [], "learning_rates": [],
    }

    d_train_ratio, g_train_ratio = 1, 1
    d_acc_window = deque(maxlen=GAN_RUNNING_AVG_WINDOW)
    failed = False
    total_start = time.time()

    print("\n" + "=" * 80)
    print(f"TRAINING: {GAN_EPOCHS} epochs  |  batch_size={GAN_BATCH_SIZE}  |  "
          f"lr={GAN_INITIAL_LR}  |  clipnorm={GAN_CLIPNORM}")
    print("=" * 80)

    # -- Per-epoch loop ----------------------------------------------------
    for epoch in range(GAN_EPOCHS):
        if failed:
            break

        epoch_start = time.time()
        current_lr_d = update_learning_rate(
            epoch, discriminator.optimizer,
            GAN_INITIAL_LR, GAN_MIN_LR,
            GAN_LR_DECAY_EPOCHS, GAN_EPOCHS,
        )
        update_learning_rate(epoch, gan.optimizer,
                             GAN_INITIAL_LR, GAN_MIN_LR,
                             GAN_LR_DECAY_EPOCHS, GAN_EPOCHS)

        d_losses, d_accs, g_losses = [], [], []

        for batch_idx, (real_imgs, real_labels) in enumerate(dataset):
            bs = real_imgs.shape[0]
            real_labels = tf.cast(tf.reshape(real_labels, (-1, 1)), tf.int32)

            # ----- Train Discriminator (combined batch) -------------------
            for _ in range(d_train_ratio):
                noise = generate_noise(bs)
                fake_labels = generate_class_labels(bs)
                fake_imgs = generator([noise, fake_labels], training=True)

                combined_imgs = tf.concat([real_imgs, fake_imgs], axis=0)
                combined_labels = tf.concat([real_labels, fake_labels], axis=0)

                real_y = get_real_labels(bs, smooth=True)
                fake_y = get_fake_labels(bs, smooth=True)
                combined_y = np.vstack([real_y, fake_y])

                discriminator.train_on_batch(
                    [combined_imgs, combined_labels], combined_y)

                # Manual accuracy (label smoothing skews metric)
                preds = discriminator([combined_imgs, combined_labels], training=False)
                real_ok = tf.reduce_sum(tf.cast(preds[:bs] > 0.5, tf.float32))
                fake_ok = tf.reduce_sum(tf.cast(preds[bs:] < 0.5, tf.float32))
                d_acc = float((real_ok + fake_ok) / (bs * 2))

                d_losses.append(0.0)   # placeholder (train_on_batch loss unreliable combined)
                d_accs.append(d_acc)
                d_acc_window.append(d_acc)

                if check_for_nan(d_acc):
                    print(f"NaN in Discriminator @ epoch {epoch + 1}, batch {batch_idx}")
                    failed = True
                    break

            # ----- Train Generator ----------------------------------------
            for _ in range(g_train_ratio):
                noise = generate_noise(bs)
                gen_labels = generate_class_labels(bs)
                valid_y = get_real_labels(bs, smooth=True)
                g_loss = gan.train_on_batch([noise, gen_labels], valid_y)
                g_losses.append(g_loss)

                if check_for_nan(g_loss):
                    print(f"NaN in Generator @ epoch {epoch + 1}, batch {batch_idx}")
                    failed = True
                    break

            # ----- Adaptive strategy --------------------------------------
            if len(d_acc_window) >= GAN_RUNNING_AVG_WINDOW:
                avg = np.mean(d_acc_window)
                if avg < GAN_D_TARGET_ACC_MIN:
                    d_train_ratio, g_train_ratio = 2, 1
                elif avg > GAN_D_TARGET_ACC_MAX:
                    d_train_ratio, g_train_ratio = 1, 2
                else:
                    d_train_ratio, g_train_ratio = 1, 1

        if failed:
            break

        # -- Epoch summary -------------------------------------------------
        epoch_time = time.time() - epoch_start
        avg_d_loss = np.mean(d_losses) if d_losses else 0
        avg_d_acc = np.mean(d_accs) if d_accs else 0
        avg_g_loss = np.mean(g_losses) if g_losses else 0

        history["d_loss"].append(float(avg_d_loss))
        history["d_acc"].append(float(avg_d_acc))
        history["g_loss"].append(float(avg_g_loss))
        history["epoch_times"].append(float(epoch_time))
        history["learning_rates"].append(float(current_lr_d))

        # Balance status
        if avg_d_acc < GAN_D_TARGET_ACC_MIN:
            status = "D weak -> training D more"
        elif avg_d_acc > GAN_D_TARGET_ACC_MAX:
            status = "D strong -> training G more"
        else:
            status = "balanced"

        elapsed = time.time() - total_start
        bar_len = 50
        filled = int(bar_len * (epoch + 1) / GAN_EPOCHS)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{GAN_EPOCHS}")
        print(f"Progress: [{bar}] {(epoch + 1) / GAN_EPOCHS * 100:.1f}%")
        print(f"D Loss: {avg_d_loss:.4f}  D Acc: {avg_d_acc:.4f}")
        print(f"G Loss: {avg_g_loss:.4f}")
        print(f"LR: {current_lr_d:.6f}  Epoch time: {epoch_time:.1f}s  Total: {elapsed / 60:.1f}m")
        print(f"Status: {status}")

        # Warnings
        if epoch > 10:
            recent_g = history["g_loss"][-5:]
            recent_d = history["d_acc"][-5:]
            if np.mean(recent_g) < 0.01:
                print("WARNING: possible mode collapse (G loss -> 0)")
            if 0.48 < np.mean(recent_d) < 0.52:
                print("WARNING: discriminator at random guessing (50%)")
            if avg_d_acc > 0.95:
                print("WARNING: discriminator too strong (>95%)")

        # -- Save samples --------------------------------------------------
        if (epoch + 1) % GAN_SAVE_INTERVAL == 0 or epoch == 0:
            save_generated_images(generator, epoch + 1, LATENT_DIM,
                                  NUM_CLASSES, class_names, str(samples_dir))

        # -- Checkpoints ---------------------------------------------------
        if (epoch + 1) % GAN_CHECKPOINT_INTERVAL == 0:
            generator.save(str(checkpoints_dir / f"generator_epoch_{epoch + 1:04d}.h5"))
            discriminator.save(str(checkpoints_dir / f"discriminator_epoch_{epoch + 1:04d}.h5"))

    # ======================================================================
    # POST-TRAINING
    # ======================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    if failed:
        print("TRAINING STOPPED (INSTABILITY)")
    else:
        print("GAN TRAINING COMPLETE")
    print(f"Total time: {total_time / 3600:.2f}h")

    # -- Save final models ------------------------------------------------
    generator.save(str(checkpoints_dir / "generator_final.h5"))
    discriminator.save(str(checkpoints_dir / "discriminator_final.h5"))
    gan.save(str(checkpoints_dir / "gan_final.h5"))

    # -- Save history ------------------------------------------------------
    with open(str(GAN_DIR / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # -- Plots -------------------------------------------------------------
    plot_training_curves(history, str(GAN_DIR))
    plot_final_samples(generator, NUM_CLASSES, class_names, str(GAN_DIR))

    print(f"\nOutput saved to: {GAN_DIR}")
    return generator, discriminator, history


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_gan()
