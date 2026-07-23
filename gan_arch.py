"""
MalGAN -- Conditional DCGAN Architecture
=========================================
Generator and Discriminator definitions for 64x64 malware image generation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(latent_dim=100, num_classes=6):
    """
    Conditional DCGAN Generator.

    100-d noise + 50-d label embedding -> Dense(8x8x256) -> 3 x Conv2DTranspose -> 64x64x3 (tanh)
    """
    noise_input = layers.Input(shape=(latent_dim,), name="noise")
    label_input = layers.Input(shape=(1,), dtype="int32", name="label")

    label_emb = layers.Embedding(num_classes, 50)(label_input)
    label_emb = layers.Flatten()(label_emb)

    x = layers.Concatenate()([noise_input, label_emb])
    x = layers.Dense(8 * 8 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)

    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # 32x32 -> 64x64
    output = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding="same", activation="tanh")(x)

    return models.Model([noise_input, label_input], output, name="Generator_64x64")


def build_discriminator(img_shape=(64, 64, 3), num_classes=6):
    """
    Conditional DCGAN Discriminator.

    64x64 image + spatially-tiled label -> 3 x Conv2D down-sample -> sigmoid
    """
    img_input = layers.Input(shape=img_shape, name="image")
    label_input = layers.Input(shape=(1,), dtype="int32", name="label")

    label_emb = layers.Embedding(num_classes, 50)(label_input)
    label_emb = layers.Flatten()(label_emb)
    label_emb = layers.Dense(img_shape[0] * img_shape[1])(label_emb)
    label_emb = layers.Reshape((*img_shape[:2], 1))(label_emb)

    x = layers.Concatenate()([img_input, label_emb])

    # 64 -> 32
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # 32 -> 16
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # 16 -> 8
    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return models.Model([img_input, label_input], output, name="Discriminator_64x64")


def build_gan(generator, discriminator, latent_dim=100):
    """
    Combine generator + frozen discriminator into the full GAN.
    """
    discriminator.trainable = False

    noise = layers.Input(shape=(latent_dim,), name="gan_noise")
    label = layers.Input(shape=(1,), dtype="int32", name="gan_label")

    fake_img = generator([noise, label])
    validity = discriminator([fake_img, label])

    return models.Model([noise, label], validity, name="GAN")
