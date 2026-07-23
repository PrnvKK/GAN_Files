# MalGAN

## Generative Adversarial Network-Based Data Augmentation for Malware Image Classification

MalGAN is a reproducible deep-learning pipeline that uses a **Conditional DCGAN** to synthesize class-conditioned malware images from the [MaleVis](https://web.cs.hacettepe.edu.tr/~selman/malevis/) dataset, then augments a **ResNet50** classifier with the generated samples to study the effect of GAN-based data augmentation on malware family classification.

---

## Results

| Model | Validation Accuracy | Validation Loss |
|:---|:---:|:---:|
| Baseline ResNet50 | 96.11% | 0.1990 |
| Augmented ResNet50 (+100 synth/class) | **96.33%** | **0.1679** |

While the accuracy gain is modest (+0.22%), the validation loss dropped by approximately **15.6%** — indicating that GAN-generated synthetic images act primarily as a **regularizer** in this high-accuracy regime, improving model calibration and reducing overfitting to idiosyncratic features of the finite training set.

---

## Architecture

### Baseline Classifier — ResNet50

![ResNet50 Baseline Architecture](assets/baseline_model.png)

The classifier is built on an **ImageNet-pretrained ResNet50** backbone (`include_top=False`) with the last 20 convolutional layers unfrozen for domain-specific fine-tuning. A custom classification head is appended:

| Stage | Operation | Output Shape | Notes |
|:---|:---|:---|:---|
| Input | — | 224 × 224 × 3 | RGB malware image |
| Backbone | ResNet50 (20 layers trainable) | 7 × 7 × 2048 | Transfer learning |
| Head 1 | GlobalAveragePooling2D | 2048 | Spatial invariance |
| Head 2 | BatchNormalization + Dropout(0.5) | 2048 | Regularization |
| Head 3 | Dense(256, ReLU) | 256 | Feature projection |
| Head 4 | BatchNormalization + Dropout(0.3) | 256 | Lighter regularization |
| Output | Dense(6, Softmax) | 6 | Family probability distribution |

**Compilation:** Adam optimizer (lr = 1e-4), sparse categorical cross-entropy loss, accuracy metric.

---

### Conditional DCGAN

![Conditional DCGAN Architecture](assets/gan_architecture.png)

The generator and discriminator are conditioned on class labels via embedding layers, enabling class-specific image generation.

#### Generator

| Stage | Operation | Output Shape | Activation |
|:---|:---|:---:|:---:|
| Noise Input | — | 100 | — |
| Label Embedding | Embedding(6 → 50) + Flatten | 50 | — |
| Concatenation | Noise (100) + Label Embed (50) | 150 | — |
| Projection | Dense(8×8×256) + BatchNorm | 8 × 8 × 256 | LeakyReLU(0.2) |
| Upsample 1 | Conv2DTranspose(128, 4×4, stride 2) | 16 × 16 × 128 | LeakyReLU(0.2) |
| Upsample 2 | Conv2DTranspose(64, 4×4, stride 2) | 32 × 32 × 64 | LeakyReLU(0.2) |
| Output | Conv2DTranspose(3, 4×4, stride 2) | **64 × 64 × 3** | tanh |

#### Discriminator

The class label is embedded, projected to 64×64 = 4,096 units, reshaped into a 64×64×1 spatial map, and concatenated with the image as an additional channel — producing a 64×64×4 tensor.

| Stage | Operation | Output Shape | Activation |
|:---|:---|:---:|:---:|
| Image Input | — | 64 × 64 × 3 | — |
| Label Map | Embedding → Dense(4096) → Reshape | 64 × 64 × 1 | — |
| Concatenation | Image + Label Map | 64 × 64 × 4 | — |
| Downsample 1 | Conv2D(64, 4×4, stride 2) + Dropout(0.3) | 32 × 32 × 64 | LeakyReLU(0.2) |
| Downsample 2 | Conv2D(128, 4×4, stride 2) + Dropout(0.3) | 16 × 16 × 128 | LeakyReLU(0.2) |
| Downsample 3 | Conv2D(256, 4×4, stride 2) + Dropout(0.3) | 8 × 8 × 256 | LeakyReLU(0.2) |
| Output | Flatten + Dense(1) | 1 | sigmoid |

**Compilation:** Both networks use Adam (lr = 2e-4, β₁ = 0.5) with binary cross-entropy loss.

---

## Training Protocol

GAN training is an inherently unstable minimax game. MalGAN incorporates a suite of stabilization techniques drawn from the GAN literature to maintain a productive adversarial equilibrium throughout the 100-epoch run.

### Stabilization Techniques

| Technique | Implementation | Purpose |
|:---|:---|:---:|
| **Label Smoothing** | Real targets = 0.9, fake targets = 0.1 | Prevents discriminator overconfidence |
| **Combined Batch Updates** | Real + fake images fused into single `train_on_batch` call | More stable gradient estimates |
| **Adaptive D/G Ratio** | Running accuracy window (10 steps); if D acc < 65%, train D 2×; if D acc > 80%, train G 2× | Maintains adversarial equilibrium |
| **Gradient Clipping** | clipnorm = 5.0 on both optimizers | Prevents catastrophic parameter updates |
| **Learning Rate Decay** | Linear decay from 2e-4 to 2e-5 after epoch 50 | Fine-tunes equilibrium in later epochs |
| **NaN Detection** | Automatic loss/accuracy NaN checks with early stop | Prevents wasteful compute on irrecoverable collapse |
| **Manual Accuracy** | Threshold-based (predictions > 0.5 for real, < 0.5 for fake) | Reliable under label smoothing (Keras metric unreliable) |

### Data Pipeline

Training images are downsized from 224×224 to 64×64 using `cv2.resize` with `INTER_AREA` interpolation (anti-aliasing for downscaling). A `tf.data.Dataset` pipeline performs on-the-fly normalization from [0, 255] to [-1, 1] — matching the generator's tanh output range — with a shuffle buffer of 1,000, batch size of 64, and `AUTOTUNE` prefetch to overlap data preparation with GPU computation.

### Synthetic Image Generation

After GAN training, the generator produces **100 synthetic images per class** (600 total). Each 64×64 image undergoes:

1. **Denormalization:** tanh output [-1, 1] → uint8 [0, 255] via `(img + 1.0) × 127.5`
2. **Upscaling:** 64×64 → 224×224 via bicubic interpolation (`INTER_CUBIC`)

The synthetic images are structurally indistinguishable from real MaleVis images to the downstream loader — same PNG format, spatial dimensions, and RGB encoding — requiring no special handling in the augmented training pipeline.

---

## Pipeline

The pipeline is fully modular. Each script is a standalone entry point — run them in order to reproduce the full experiment.

| Step | Script | Description |
|:---:|:---|:---|
| 1 | `load_data.py` | Loads MaleVis train/val splits into NumPy arrays via OpenCV |
| 2 | `load_balanced_data.py` | Selects 6 malware families, remaps labels to 0–5, saves sample grid |
| 3 | `cnn_baseline_train.py` | Trains baseline ResNet50 (transfer learning) on 6-class subset |
| 4 | `gan_train.py` | Trains Conditional DCGAN (100 epochs) with adaptive stabilization |
| 5 | `gan_augment.py` | Generates 100 synthetic images/class, upscales to 224×224 |
| 6 | `cnn_train_augmented.py` | Retrains identical ResNet50 on real + synthetic data (fair comparison) |
| 7 | `augmentation_ratios_experiment.py` | Sweeps augmentation ratios [0, 25, 50, 100, 200] |
| 8 | `multi_run_experiment.py` | N=3 runs per ratio with mean ± std error bars |

To ensure a fair comparison, the baseline and augmented classifiers use **identical** ResNet50 architectures, optimizers, loss functions, callbacks, and hyperparameters. The only variable is the training data. The validation set is never augmented.

---

## Quickstart

### Prerequisites

```
tensorflow>=2.10
numpy
matplotlib
opencv-python
tqdm
```

### Installation

```bash
git clone https://github.com/PrnvKK/MalGAN.git
cd MalGAN
pip install tensorflow numpy matplotlib opencv-python tqdm
```

### Configure Paths

All paths and hyperparameters are centralized in [`config.py`](config.py). Set the dataset location via an environment variable:

```bash
export MALEVIS_DATA_DIR=/path/to/malevis_train_val_300x300
```

If unset, scripts default to `./malevis_data/malevis_train_val_300x300/` relative to the project root.

### Run the Pipeline

```bash
# 1. Train baseline classifier
python cnn_baseline_train.py

# 2. Train the GAN
python gan_train.py

# 3. Generate synthetic images
python gan_augment.py

# 4. Train augmented classifier
python cnn_train_augmented.py

# 5. Run experiments (optional)
python augmentation_ratios_experiment.py
python multi_run_experiment.py
```

---

## Dataset

This project uses the **MaleVis** (Malware Vision) dataset — an open-source collection of malware binary executables converted to grayscale images via byte-to-image transformation. Each byte value (0–255) is interpreted as a pixel intensity, and the byte stream is reshaped into a 2D array. Different executable sections (headers, code, data, resources) produce visually distinct texture patterns that CNNs can learn to differentiate.

A balanced **6-class subset** is constructed:

| Index | Family | Type |
|:---:|:---|:---|
| 0 | Androm | Trojan |
| 1 | Elex | Adware |
| 2 | Expiro | Virus |
| 3 | HackKMS | HackTool |
| 4 | Hlux | Rootkit |
| 5 | Sality | Virus |

The original MaleVis train/validation split is preserved — no validation sample appears in training.

---

## Configuration

All experiment settings are defined in [`config.py`](config.py) and can be overridden via environment variables:

| Variable | Default | Description |
|:---|:---|:---|
| `MALEVIS_DATA_DIR` | `./malevis_data/malevis_train_val_300x300` | Path to MaleVis dataset |
| `MALGAN_OUTPUT_DIR` | `./output` | Root directory for all outputs |

### Key Hyperparameters

| Parameter | Value |
|:---|:---:|
| CNN input size | 224 × 224 |
| GAN input size | 64 × 64 |
| Latent dimension | 100 |
| GAN epochs | 100 |
| GAN batch size | 64 |
| GAN learning rate | 2 × 10⁻⁴ |
| GAN gradient clipnorm | 5.0 |
| Label smoothing | 0.9 / 0.1 |
| CNN epochs | 50 |
| CNN batch size | 32 |
| CNN learning rate | 1 × 10⁻⁴ |
| ResNet50 trainable layers | 20 |
| Synthetic images per class | 100 |

---

## Output Structure

All outputs are written to `./output/` (configurable via `MALGAN_OUTPUT_DIR`):

```
output/
├── baseline/                           Baseline ResNet50
│   ├── baseline_cnn_best.h5
│   ├── baseline_training_history.json
│   ├── baseline_training_curves.png
│   ├── baseline_val_predictions.npy
│   └── baseline_val_true_labels.npy
├── gan/                               Conditional DCGAN
│   ├── checkpoints/
│   │   ├── generator_epoch_0010.h5
│   │   ├── ...
│   │   ├── generator_final.h5
│   │   ├── discriminator_final.h5
│   │   └── gan_final.h5
│   ├── samples/
│   │   └── generated_epoch_0005.png
│   ├── training_history.json
│   ├── training_curves.png
│   └── final_samples_grid.png
├── synthetic_data/                    600 GAN-generated images
│   ├── Androm/
│   ├── Elex/
│   ├── Expiro/
│   ├── HackKMS/
│   ├── Hlux/
│   └── Sality/
├── augmented/                         Augmented ResNet50
│   ├── augmented_cnn_best.h5
│   ├── augmented_training_history.json
│   └── augmented_val_predictions.npy
└── experiments/
    ├── augmentation_ratios/
    │   ├── ratio_000/
    │   ├── ratio_025/
    │   ├── ...
    │   ├── augmentation_ratios_summary.png
    │   ├── augmentation_ratios_results.csv
    │   └── all_results.json
    └── multi_run/
        ├── multi_run_summary.png
        ├── multi_run_summary.json
        └── _checkpoint.json
```

---

## Project Layout

```
MalGAN/
├── config.py                           Centralized configuration
├── load_data.py                        MaleVis dataset loader
├── load_balanced_data.py               6-class balanced subset creator
├── cnn_arch.py                         ResNet50 & custom CNN definitions
├── gan_arch.py                         Conditional DCGAN architecture
├── cnn_baseline_train.py              Baseline classifier training
├── gan_train.py                        GAN training loop
├── gan_augment.py                      Synthetic image generation
├── cnn_train_augmented.py              Augmented classifier training
├── augmentation_ratios_experiment.py   Ratio sweep experiment
├── multi_run_experiment.py             Multi-run experiment (N=3)
├── assets/
│   ├── baseline_model.png              Figure 1: ResNet50 architecture
│   └── gan_architecture.png            Figure 2: DCGAN architecture
└── README.md
```

---

## Discussion

The results demonstrate that GAN-based augmentation provides a measurable benefit for malware image classification **even when the baseline already performs at 96%+ accuracy**. The primary contribution of synthetic data in this regime is **regularization** rather than dramatic accuracy improvement — exposing the classifier to a wider variety of class-conditioned visual patterns during training reduces overfitting to the finite original training set and yields better-calibrated probability estimates (lower cross-entropy loss).

This finding suggests GAN augmentation may prove substantially more impactful when applied to harder problems: more classes, fewer real samples per class, or greater inter-class visual similarity. The generator-as-data-factory paradigm also addresses a structural bottleneck in malware research — once trained, a generator can produce unlimited labelled samples for any downstream task without transferring sensitive real binaries, facilitating collaboration and reproducibility in a domain where data sharing is frequently restricted.

---

## License

This project is provided for academic and research purposes.

## Author

**Pranav Krishnakumar** — Manipal Institute of Technology, Bengaluru