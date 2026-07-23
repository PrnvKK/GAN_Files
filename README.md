# MalGAN &mdash; GAN-Based Data Augmentation for Malware Image Classification

**MalGAN** trains a Conditional DCGAN to generate synthetic malware images and
uses them to augment a ResNet50 classifier, improving malware family
recognition on the [MaleVis](https://web.cs.hacettepe.edu.tr/~selman/malevis/) dataset.

## Pipeline Overview

```
MaleVis Dataset (RGB images)
        |
        v
 [1] load_data.py          -- Read train/val splits into NumPy arrays
        |
        v
 [2] load_balanced_data.py -- Select 6 malware families, create balanced subset
        |
        +--> [3] cnn_baseline_train.py  -- Train baseline ResNet50 (no aug.)
        |         |
        |         `--> output/baseline/   (model, history, predictions, plots)
        |
        +--> [4] gan_train.py            -- Train Conditional DCGAN (64x64)
        |         |
        |         `--> output/gan/        (models, checkpoints, samples, plots)
        |
        +--> [5] gan_augment.py          -- Generate synthetic images
        |         |
        |         `--> output/synthetic_data/  (per-class PNG folders)
        |
        +--> [6] cnn_train_augmented.py -- Retrain ResNet50 on original + synth
        |         |
        |         `--> output/augmented/  (model, history, predictions)
        |
        `--> [7] Experiments
                  |
                  +--> augmentation_ratios_experiment.py  -- Compare ratios
                  `--> multi_run_experiment.py            -- Mean +/- std (N=3)
```

## Dataset

Uses the **MaleVis** malware image dataset (pre-split into `train/` and `val/`
folders, one subfolder per family).  Set the data path via an environment
variable:

```bash
export MALEVIS_DATA_DIR=/path/to/malevis_data/malevis_train_val_300x300
```

If the variable is not set, the scripts look for `malevis_data/malevis_train_val_300x300/`
inside the project root.

**Target families** (6 classes):

| # | Family  |
|---|---------|
| 0 | Androm  |
| 1 | Elex    |
| 2 | Expiro  |
| 3 | HackKMS |
| 4 | Hlux    |
| 5 | Sality  |

## Installation

```bash
# Clone
git clone https://github.com/PrnvKK/MalGAN.git
cd MalGAN

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install tensorflow numpy matplotlib opencv-python tqdm
```

## Configuration

All paths, hyperparameters, and constants live in **`config.py`**.  Key settings
you may want to override via environment variables:

| Variable            | Default                                    | Description                     |
|---------------------|--------------------------------------------|---------------------------------|
| `MALEVIS_DATA_DIR`  | `./malevis_data/malevis_train_val_300x300` | Path to the MaleVis dataset     |
| `MALGAN_OUTPUT_DIR` | `./output`                                 | Root directory for all outputs  |

## Usage

Run scripts in order.  Each is a standalone entry point (`python <script>.py`).

### 1. Baseline CNN

```bash
python cnn_baseline_train.py
```

Trains a ResNet50 (ImageNet weights, last 20 layers trainable) on the balanced
6-class subset.  Outputs go to `output/baseline/`:

- `baseline_cnn_best.h5` &mdash; best model checkpoint
- `baseline_training_history.json` &mdash; per-epoch metrics
- `baseline_training_curves.png` &mdash; accuracy & loss plots
- `baseline_val_predictions.npy` &mdash; validation predictions
- `baseline_val_true_labels.npy` &mdash; validation ground truth

### 2. Train the GAN

```bash
python gan_train.py
```

Trains a Conditional DCGAN (100 epochs) with:
- **Label smoothing** (targets 0.9 / 0.1)
- **Adaptive D/G ratio** &mdash; adjusts training frequency based on a running
  accuracy window
- **Gradient clipping** (clipnorm = 5.0)
- **Learning-rate decay** (linear after epoch 50)
- **NaN detection** with automatic early-stop

Outputs go to `output/gan/`:

- `checkpoints/generator_final.h5` &mdash; final generator
- `checkpoints/discriminator_final.h5` &mdash; final discriminator
- `samples/` &mdash; per-epoch generated image grids
- `training_history.json` &mdash; per-epoch metrics
- `training_curves.png` &mdash; D/G loss, D accuracy, LR, epoch times
- `final_samples_grid.png` &mdash; 5 generated images per class

### 3. Generate Synthetic Images

```bash
python gan_augment.py
```

Loads the trained generator and produces 100 synthetic images per malware
family (upscaled from 64&times;64 to 224&times;224).  Output lands in
`output/synthetic_data/<FamilyName>/`.

### 4. Augmented CNN

```bash
python cnn_train_augmented.py
```

Trains an **identical** ResNet50 architecture on the original data *plus* the
GAN-generated images.  Results go to `output/augmented/`.

### 5. Experiments

```bash
python augmentation_ratios_experiment.py    # Single run per ratio
python multi_run_experiment.py              # 3 runs per ratio with error bars
```

Compare validation accuracy across augmentation ratios (0, 25, 50, 100, 200
images/class).  Results land in `output/experiments/`.

## Output Directory Structure

```
output/
|
+-- subset_samples.png               # One sample per class (from load_balanced_data)
|
+-- baseline/                        # Baseline ResNet50
|   +-- baseline_cnn_best.h5
|   +-- baseline_training_history.json
|   +-- baseline_training_curves.png
|   +-- baseline_val_predictions.npy
|   +-- baseline_val_true_labels.npy
|
+-- gan/                             # Trained Conditional DCGAN
|   +-- checkpoints/
|   |   +-- generator_epoch_xxxx.h5
|   |   +-- discriminator_epoch_xxxx.h5
|   |   +-- generator_final.h5
|   |   +-- discriminator_final.h5
|   |   +-- gan_final.h5
|   +-- samples/
|   |   +-- generated_epoch_xxxx.png
|   +-- training_history.json
|   +-- training_curves.png
|   +-- final_samples_grid.png
|
+-- synthetic_data/                  # GAN-generated images (224x224)
|   +-- Androm/
|   +-- Elex/
|   +-- Expiro/
|   +-- HackKMS/
|   +-- Hlux/
|   +-- Sality/
|
+-- augmented/                       # Augmented ResNet50
|   +-- augmented_cnn_best.h5
|   +-- augmented_training_history.json
|   +-- augmented_val_predictions.npy
|
+-- experiments/
    +-- augmentation_ratios/
    |   +-- ratio_000/
    |   +-- ratio_025/
    |   +-- ...
    |   +-- augmentation_ratios_summary.png
    |   +-- augmentation_ratios_results.csv
    |   +-- all_results.json
    |
    +-- multi_run/
        +-- multi_run_summary.png
        +-- multi_run_summary.json
        +-- _checkpoint.json
```

## Model Architecture

### Classifier

| Component       | Detail                              |
|-----------------|-------------------------------------|
| Backbone        | ResNet50 (ImageNet, 20 layers unfrozen) |
| Head            | GAP -> BN -> Dropout(0.5) -> Dense(256) -> BN -> Dropout(0.3) -> Dense(6, softmax) |
| Optimizer       | Adam (lr = 1e-4)                    |
| Loss            | Sparse categorical cross-entropy    |
| Regularisation  | EarlyStopping (patience=10), ReduceLROnPlateau |

### Generator (Conditional DCGAN)

| Layer        | Output Shape | Activation |
|--------------|-------------|------------|
| Noise (100) + Label Embed (50) | 150 | &mdash; |
| Dense + BN   | 8&times;8&times;256 | LeakyReLU(0.2) |
| Conv2DT (128, stride 2) | 16&times;16&times;128 | LeakyReLU(0.2) |
| Conv2DT (64, stride 2)  | 32&times;32&times;64  | LeakyReLU(0.2) |
| Conv2DT (3, stride 2)   | 64&times;64&times;3   | tanh |

### Discriminator

| Layer        | Output Shape | Activation |
|--------------|-------------|------------|
| Image (64&times;64&times;3) + Label tile (64&times;64&times;1) | 64&times;64&times;4 | &mdash; |
| Conv2D (64, stride 2)  | 32&times;32&times;64  | LeakyReLU(0.2) |
| Conv2D (128, stride 2) | 16&times;16&times;128 | LeakyReLU(0.2) |
| Conv2D (256, stride 2) | 8&times;8&times;256   | LeakyReLU(0.2) |
| Flatten + Dense(1)     | 1                     | sigmoid |

## Key Hyperparameters

| Parameter              | Value    |
|------------------------|----------|
| CNN input size         | 224&times;224 |
| GAN input size         | 64&times;64   |
| Latent dimension       | 100     |
| GAN epochs             | 100     |
| GAN batch size         | 64      |
| GAN initial LR         | 2e-4    |
| GAN min LR             | 2e-5    |
| GAN clipnorm           | 5.0     |
| Label smoothing        | 0.9 / 0.1 |
| CNN epochs             | 50      |
| CNN batch size         | 32      |
| CNN learning rate      | 1e-4    |
| Synth images per class | 100     |

## Licence

This project is provided for academic and research purposes.
