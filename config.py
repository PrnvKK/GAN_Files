"""
MalGAN -- Central Configuration
================================
All paths, hyperparameters, and constants for the GAN-based malware image
classification pipeline.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (directory containing this config file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data paths (update these for your environment)
# ---------------------------------------------------------------------------
# Path to the MaleVis dataset (pre-split into train/val folders)
MALEVIS_DATA_DIR = os.environ.get(
    "MALEVIS_DATA_DIR",
    str(PROJECT_ROOT / "malevis_data" / "malevis_train_val_300x300"),
)

# ---------------------------------------------------------------------------
# Output directory -- everything lands here
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("MALGAN_OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# ---------------------------------------------------------------------------
# Sub-directories (created automatically)
# ---------------------------------------------------------------------------
BASELINE_DIR = OUTPUT_DIR / "baseline"
GAN_DIR = OUTPUT_DIR / "gan"
AUGMENTED_DIR = OUTPUT_DIR / "augmented"
SYNTHETIC_DATA_DIR = OUTPUT_DIR / "synthetic_data"
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"

# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------
SELECTED_FAMILIES = ["Androm", "Elex", "Expiro", "HackKMS", "Hlux", "Sality"]
NUM_CLASSES = len(SELECTED_FAMILIES)

# ---------------------------------------------------------------------------
# Image dimensions
# ---------------------------------------------------------------------------
CNN_IMG_SIZE = (224, 224)   # Input size for ResNet classifier
GAN_IMG_SIZE = (64, 64)     # Input size for DCGAN
IMG_CHANNELS = 3

CNN_INPUT_SHAPE = (*CNN_IMG_SIZE, IMG_CHANNELS)
GAN_INPUT_SHAPE = (*GAN_IMG_SIZE, IMG_CHANNELS)

# ---------------------------------------------------------------------------
# GAN hyperparameters
# ---------------------------------------------------------------------------
LATENT_DIM = 100
GAN_EPOCHS = 100
GAN_BATCH_SIZE = 64
GAN_INITIAL_LR = 0.0002
GAN_MIN_LR = 0.00002
GAN_LR_DECAY_EPOCHS = 50
GAN_LABEL_SMOOTH_REAL = 0.9
GAN_LABEL_SMOOTH_FAKE = 0.1
GAN_D_TARGET_ACC_MIN = 0.65
GAN_D_TARGET_ACC_MAX = 0.80
GAN_RUNNING_AVG_WINDOW = 10
GAN_SAVE_INTERVAL = 5
GAN_CHECKPOINT_INTERVAL = 10
GAN_CLIPNORM = 5.0

# ---------------------------------------------------------------------------
# CNN classifier hyperparameters
# ---------------------------------------------------------------------------
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.0001
CNN_EARLY_STOPPING_PATIENCE = 10
CNN_LR_PATIENCE = 5
CNN_MIN_LR = 1e-7
CNN_DROPOUT_1 = 0.5
CNN_DROPOUT_2 = 0.3
CNN_DENSE_UNITS = 256
CNN_RESNET_TRAINABLE_LAYERS = 20

# ---------------------------------------------------------------------------
# GAN augmentation
# ---------------------------------------------------------------------------
SYNTHETIC_IMAGES_PER_CLASS = 100
AUGMENTATION_RATIOS = [0, 25, 50, 100, 200]
N_MULTI_RUNS = 3

# ---------------------------------------------------------------------------
# Helper -- ensure directory exists
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_output_dirs() -> None:
    """Create the full output directory tree."""
    for d in [
        BASELINE_DIR,
        GAN_DIR / "checkpoints",
        GAN_DIR / "samples",
        AUGMENTED_DIR,
        SYNTHETIC_DATA_DIR,
        EXPERIMENTS_DIR,
    ]:
        ensure_dir(d)
