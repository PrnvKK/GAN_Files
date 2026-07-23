"""
MalGAN -- CNN Architectures
============================
Transfer-learning ResNet50 and a lightweight custom CNN for malware
family classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

from config import (
    CNN_INPUT_SHAPE,
    CNN_LEARNING_RATE,
    CNN_DROPOUT_1,
    CNN_DROPOUT_2,
    CNN_DENSE_UNITS,
    CNN_RESNET_TRAINABLE_LAYERS,
)


def create_resnet_model(input_shape=None, num_classes=6):
    """
    ResNet50 with transfer learning.

    Freezes all but the last *k* convolutional layers, then appends a
    custom classification head with global average pooling, batch-norm,
    dropout, and a 256-unit dense layer.
    """
    input_shape = input_shape or CNN_INPUT_SHAPE

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers[:-CNN_RESNET_TRAINABLE_LAYERS]:
        layer.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(CNN_DROPOUT_1),
            layers.Dense(CNN_DENSE_UNITS, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(CNN_DROPOUT_2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="MalwareResNet50",
    )
    return model


def create_custom_cnn(input_shape=None, num_classes=6):
    """
    Lightweight custom CNN (no pre-trained weights).
    """
    input_shape = input_shape or CNN_INPUT_SHAPE

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                          input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="MalwareCustomCNN",
    )
    return model


def get_compiled_model(model_type="resnet", num_classes=6):
    """
    Build and compile a model.

    Parameters
    ----------
    model_type : str
        ``"resnet"`` or ``"custom"``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    tf.keras.Model
    """
    if model_type == "resnet":
        model = create_resnet_model(num_classes=num_classes)
    else:
        model = create_custom_cnn(num_classes=num_classes)

    model.compile(
        optimizer=Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MALGAN -- CNN ARCHITECTURE CHECK")
    print("=" * 60)

    for arch in ("resnet", "custom"):
        m = get_compiled_model(arch)
        trainable = sum(tf.size(w).numpy() for w in m.trainable_weights)
        total = sum(tf.size(w).numpy() for w in m.weights)
        print(f"\n{arch.upper()}: {total:,} params ({trainable:,} trainable)")
