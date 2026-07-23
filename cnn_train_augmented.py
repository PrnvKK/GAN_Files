"""
MalGAN -- Augmented CNN Training
=================================
Trains an identical ResNet50 classifier on the original balanced subset
combined with GAN-generated synthetic images, then compares performance
against the baseline.
"""

import json
import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from config import (
    AUGMENTED_DIR,
    SYNTHETIC_DATA_DIR,
    CNN_EPOCHS,
    CNN_BATCH_SIZE,
    CNN_EARLY_STOPPING_PATIENCE,
    CNN_LR_PATIENCE,
    CNN_MIN_LR,
    NUM_CLASSES,
    ensure_dir,
)
from load_balanced_data import create_balanced_subset
from cnn_arch import get_compiled_model


# ---------------------------------------------------------------------------
# Training monitor (same as baseline for fair comparison)
# ---------------------------------------------------------------------------
class TrainingMonitor(Callback):
    def __init__(self, total_epochs, label="AUGMENTED"):
        super().__init__()
        self.total_epochs = total_epochs
        self.label = label
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'=' * 80}")
        print(f"{self.label} CNN TRAINING STARTED")
        print(f"{'=' * 80}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        elapsed = time.time() - self.start_time

        train_loss = logs.get("loss")
        train_acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")
        lr = float(self.model.optimizer.learning_rate.numpy())

        is_best = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.epochs_no_improve = 0
            is_best = True
        else:
            self.epochs_no_improve += 1

        best_marker = " *** BEST" if is_best else ""
        bar_len = 50
        filled = int(bar_len * (epoch + 1) / self.total_epochs)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        print(f"\n{'─' * 80}")
        print(f"Epoch {epoch + 1}/{self.total_epochs}{best_marker}")
        print(f"Progress: [{bar}] {(epoch + 1) / self.total_epochs * 100:.1f}%")
        print(f"Training   -> Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Validation -> Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"LR: {lr:.2e}  Epoch time: {epoch_time:.1f}s  Total: {elapsed / 60:.1f}m")
        print(f"Best Acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch}  "
              f"No improvement: {self.epochs_no_improve}/{CNN_EARLY_STOPPING_PATIENCE}")

        if train_acc - val_acc > 0.15:
            print("WARNING: possible overfitting (train-val gap > 15%)")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print(f"{self.label} CNN TRAINING COMPLETE")
        print(f"Total time: {total_time / 60:.2f} min")
        print(f"Best Val Acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch}")
        print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
def load_synthetic_data(synth_dir, class_names):
    """Load synthetic images from per-class folders, return (X, y)."""
    X_list, y_list = [], []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    if not os.path.exists(synth_dir):
        raise FileNotFoundError(f"Synthetic data not found at {synth_dir}. "
                                "Run gan_augment.py first.")

    total = 0
    for class_name in class_names:
        class_dir = os.path.join(synth_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"WARNING: no synthetic data for '{class_name}'")
            continue

        class_idx = class_to_idx[class_name]
        count = 0
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            X_list.append(img)
            y_list.append(class_idx)
            count += 1

        print(f"  {class_name}: {count} synthetic images")
        total += count

    if total == 0:
        raise RuntimeError("No synthetic images loaded.")

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
def train_augmented(synth_dir=None):
    ensure_dir(AUGMENTED_DIR)
    synth_dir = synth_dir or str(SYNTHETIC_DATA_DIR)

    # -- Original data -----------------------------------------------------
    print("=" * 60)
    print("LOADING ORIGINAL BALANCED DATASET")
    print("=" * 60)
    X_train_orig, y_train_orig, X_val, y_val, class_names = \
        create_balanced_subset()

    # -- Synthetic data ----------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING GAN-AUGMENTED DATA")
    print("=" * 60)
    X_train_synth, y_train_synth = load_synthetic_data(synth_dir, class_names)

    # -- Combine & shuffle -------------------------------------------------
    X_train = np.concatenate([X_train_orig, X_train_synth], axis=0)
    y_train = np.concatenate([y_train_orig, y_train_synth], axis=0)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    print(f"\nCombined training set: {X_train.shape[0]} samples "
          f"({X_train_orig.shape[0]} original + {X_train_synth.shape[0]} synthetic)")

    # -- Normalize ---------------------------------------------------------
    X_train_norm = X_train.astype("float32") / 255.0
    X_val_norm = X_val.astype("float32") / 255.0

    # -- Model -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING RESNET50 CLASSIFIER (AUGMENTED)")
    print("=" * 60)
    model = get_compiled_model("resnet", num_classes=NUM_CLASSES)

    # -- Callbacks ---------------------------------------------------------
    monitor = TrainingMonitor(CNN_EPOCHS, label="AUGMENTED")
    checkpoint_path = str(AUGMENTED_DIR / "augmented_cnn_best.h5")

    callbacks = [
        monitor,
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                        save_best_only=True, mode="max", verbose=0),
        EarlyStopping(monitor="val_accuracy", patience=CNN_EARLY_STOPPING_PATIENCE,
                      mode="max", restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=CNN_LR_PATIENCE, min_lr=CNN_MIN_LR, verbose=0),
    ]

    # -- Train -------------------------------------------------------------
    t0 = time.time()
    history = model.fit(
        X_train_norm, y_train,
        batch_size=CNN_BATCH_SIZE,
        epochs=CNN_EPOCHS,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=0,
    )
    train_time = (time.time() - t0) / 60

    # -- Save history ------------------------------------------------------
    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "training_time_minutes": train_time,
        "epochs_completed": len(history.history["loss"]),
        "best_val_accuracy": float(monitor.best_val_acc),
        "best_val_loss": float(monitor.best_val_loss),
        "best_epoch": int(monitor.best_epoch),
    }
    history_path = AUGMENTED_DIR / "augmented_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    # -- Evaluation --------------------------------------------------------
    val_loss, val_acc = model.evaluate(X_val_norm, y_val, batch_size=CNN_BATCH_SIZE,
                                       verbose=0)
    print(f"\nFinal Val Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Final Val Loss:     {val_loss:.4f}")

    # -- Predictions -------------------------------------------------------
    y_pred = np.argmax(model.predict(X_val_norm, batch_size=CNN_BATCH_SIZE, verbose=0), axis=1)
    np.save(AUGMENTED_DIR / "augmented_val_predictions.npy", y_pred)

    print("\nAugmented training complete.")
    return model, history_dict


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_augmented()
