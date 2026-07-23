"""
MalGAN -- GAN Augmentation (Synthetic Image Generator)
=======================================================
Loads a trained generator, produces synthetic images per class, upscales
them to 224x224, and writes them to disk for downstream CNN training.
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from config import (
    SYNTHETIC_DATA_DIR,
    GAN_DIR,
    LATENT_DIM,
    SELECTED_FAMILIES,
    SYNTHETIC_IMAGES_PER_CLASS,
    CNN_IMG_SIZE,
    ensure_dir,
)


def generate_synthetic_images(
    model_path=None,
    output_dir=None,
    num_per_class=None,
    class_names=None,
    latent_dim=None,
    target_size=None,
):
    """
    Generate synthetic malware images from a trained GAN generator.

    Parameters
    ----------
    model_path : str, optional
        Path to the generator ``.h5`` file.
        Default: ``output/gan/checkpoints/generator_final.h5``
    output_dir : str, optional
        Root directory for per-class subfolders.
    num_per_class : int
        Number of images to generate per malware family.
    class_names : list[str]
    latent_dim : int
    target_size : tuple
        Upscale target ``(height, width)``.
    """
    model_path = model_path or str(GAN_DIR / "checkpoints" / "generator_final.h5")
    output_dir = output_dir or SYNTHETIC_DATA_DIR
    num_per_class = num_per_class or SYNTHETIC_IMAGES_PER_CLASS
    class_names = class_names or SELECTED_FAMILIES
    latent_dim = latent_dim or LATENT_DIM
    target_size = target_size or CNN_IMG_SIZE

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Generator not found at {model_path}. Train the GAN first.")

    print(f"Loading generator from {model_path} ...")
    generator = tf.keras.models.load_model(model_path, compile=False)

    ensure_dir(output_dir)
    total = 0

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(str(output_dir), class_name)
        os.makedirs(class_dir, exist_ok=True)

        print(f"Generating {num_per_class} images for '{class_name}' ...")
        remaining = num_per_class

        while remaining > 0:
            batch = min(32, remaining)
            noise = tf.random.normal(shape=(batch, latent_dim))
            labels = tf.convert_to_tensor(
                np.full((batch, 1), class_idx), dtype=tf.int32)

            gen_imgs = generator([noise, labels], training=False).numpy()

            for i in range(batch):
                img = ((gen_imgs[i] + 1.0) * 127.5).astype(np.uint8)
                img_224 = cv2.resize(img, target_size,
                                     interpolation=cv2.INTER_CUBIC)
                img_bgr = cv2.cvtColor(img_224, cv2.COLOR_RGB2BGR)
                fname = f"syn_{class_name}_{total + i:04d}.png"
                cv2.imwrite(os.path.join(class_dir, fname), img_bgr)

            remaining -= batch
            total += batch

    print(f"Done. {total} synthetic images saved to {output_dir}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_synthetic_images()
