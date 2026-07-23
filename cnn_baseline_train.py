"""
MalGAN -- Baseline CNN Training
================================
Trains a ResNet50 classifier on the 6-class balanced subset and saves the
model, training history, plots, and validation predictions.
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from config import (
    OUTPUT_DIR,
    BASELINE_DIR,
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
# Custom callback -- clean per-epoch progress display
# ---------------------------------------------------------------------------
class TrainingMonitor(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\n" + "=" * 80)
        print("BASELINE CNN TRAINING STARTED")
        print("=" * 80)

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
        print("\n" + "=" * 80)
        print("BASELINE CNN TRAINING COMPLETE")
        print(f"Total time: {total_time / 60:.2f} min")
        print(f"Best Val Acc: {self.best_val_acc:.4f} @ epoch {self.best_epoch}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 80)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_baseline():
    ensure_dir(BASELINE_DIR)

    # -- Data --------------------------------------------------------------
    print("=" * 60)
    print("LOADING BALANCED DATASET")
    print("=" * 60)
    X_train, y_train, X_val, y_val, class_names = create_balanced_subset()

    X_train_norm = X_train.astype("float32") / 255.0
    X_val_norm = X_val.astype("float32") / 255.0

    # -- Model -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILDING RESNET50 CLASSIFIER")
    print("=" * 60)
    model = get_compiled_model("resnet", num_classes=NUM_CLASSES)
    model.summary()

    # -- Callbacks ---------------------------------------------------------
    monitor = TrainingMonitor(CNN_EPOCHS)

    checkpoint_path = str(BASELINE_DIR / "baseline_cnn_best.h5")
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
    print(f"\nEpochs: {CNN_EPOCHS}  Batch size: {CNN_BATCH_SIZE}")
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
    history_path = BASELINE_DIR / "baseline_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"History saved -> {history_path}")

    # -- Evaluation --------------------------------------------------------
    val_loss, val_acc = model.evaluate(X_val_norm, y_val, batch_size=CNN_BATCH_SIZE,
                                       verbose=0)
    print(f"\nFinal Val Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Final Val Loss:     {val_loss:.4f}")

    # -- Predictions -------------------------------------------------------
    y_pred_probs = model.predict(X_val_norm, batch_size=CNN_BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    np.save(BASELINE_DIR / "baseline_val_predictions.npy", y_pred)
    np.save(BASELINE_DIR / "baseline_val_true_labels.npy", y_val)

    # -- Plots -------------------------------------------------------------
    _plot_history(history, monitor, BASELINE_DIR)

    print("\nBaseline training complete.")
    return model, history_dict


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_history(history, monitor, out_dir):
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    best_epoch = monitor.best_epoch

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs_range, history.history["accuracy"], "o-",
                 label="Training Acc", linewidth=2, markersize=4)
    axes[0].plot(epochs_range, history.history["val_accuracy"], "s-",
                 label="Validation Acc", linewidth=2, markersize=4)
    axes[0].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({best_epoch})")
    axes[0].scatter([best_epoch], [monitor.best_val_acc], color="red", s=200,
                    marker="*", edgecolors="black", linewidths=1.5, zorder=5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Model Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    axes[1].plot(epochs_range, history.history["loss"], "o-",
                 label="Training Loss", linewidth=2, markersize=4)
    axes[1].plot(epochs_range, history.history["val_loss"], "s-",
                 label="Validation Loss", linewidth=2, markersize=4)
    axes[1].axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7,
                    label=f"Best epoch ({best_epoch})")
    axes[1].scatter([best_epoch], [monitor.best_val_loss], color="red", s=200,
                    marker="*", edgecolors="black", linewidths=1.5, zorder=5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Model Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "baseline_training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plots saved -> {plot_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_baseline()
