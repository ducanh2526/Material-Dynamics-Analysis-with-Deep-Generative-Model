
# Material Dynamics Analysis with Deep Generative Model

This repository contains a unified Python-based implementation of **Progressive Growing GANs (PGGAN)**, designed to support the research study:

**Title:** Material Dynamics Analysis with Deep Generative Model  
**Scope:** This work explores how deep generative models can infer intermediate transformations of material systems by training on experimental images that capture discrete snapshots of dynamic processes.

## Background and Purpose

Understanding material evolution at the nanoscale—including **phase transitions**, **structural deformations**, and **chemical reactions**—remains a fundamental challenge due to limited temporal resolution in imaging experiments. This repository provides tools for training **deep generative models**, with the goal of **reconstructing plausible intermediate transformation stages** between observed experimental images.

By treating the generative model as a probabilistic representation of the underlying material dynamics, the trained models are later used in **Monte Carlo simulations** to explore various transformation pathways. This work evaluates the framework's applicability through three representative case studies:

1. **Tantalum Test Chart Translation**
2. **Gold Nanoparticle Diffusion in Polyvinyl Alcohol (PVA) Solution**
3. **Copper Sulfidation in Heterogeneous Rubber/Brass Composites**

## 📂 Repository Structure

```
.
├── train_pggan.py           # Main training script for 2D/3D PGGAN
├── pgan_gen.py              # Script for generating images from trained models (used in downstream Monte Carlo simulation)
├── pg_gan.py                # Full PGGAN model definition with fade-in and stabilization stages
├── config/                  # YAML configuration files for each case study
├── utils/
│   ├── plot.py              # Visualization functions for 2D and 3D output samples
│   └── gan_dat_gen.py       # Data iterator for loading experimental image datasets
```

## Features

- Supports both **2D** and **3D image generation**
- Implements **Progressive Growing GAN (PGGAN)** evolving image resolution over training stages
- Designed for **later integration with Monte Carlo sampling pipelines** to infer transformation pathways
- Includes **visualization utilities** for monitoring training progression and inspecting generated samples

## Requirements

- Python 3.x
- TensorFlow 2.x
- numpy
- matplotlib
- pyyaml

Install all required packages:

```bash
pip install tensorflow numpy matplotlib pyyaml
```

## How to Train the Generative Model

### 1. Prepare Training Data

Place the experimental images (preprocessed into `.npy` format) into a target directory (e.g., `data/CXDI_testChart/` for the first case study).

### 2. Configure Model Parameters

Create or modify a YAML file under the `config/` directory. Example for the test chart case:

```yaml
name: CXDI_Chart
latent_dim: 128
filters: [512, 256, 128, 64]
filters_scale: 1
d_extra_steps: 1
n_epochs: 10
batch_size: [128, 64, 32, 16]
resolutions: [4, 8, 16, 32]
input_shape: [32, 32, 1]
train_dir: data/CXDI_testChart
```

### 3. Run Training

```bash
python train_pggan.py --config config/CXDI_testChart.yaml
```

Training will progress through each resolution level with automated **fade-in** and **stabilization** control.

## How to Generate Images After Training

After training and checkpoint saving, generate samples from the trained generator using implementation in pgan_gen.py.

This step is used for later **Monte Carlo trajectory sampling** during material dynamics analysis.

## Output Directory Structure

Model checkpoints, visualization samples, and configuration snapshots are automatically saved to:

```
../Visualize/<DATA_NAME>/<model_name>/
```

Contents include:

- Generator and discriminator weights (`.ckpt`)
- Sample images at each training stage
- Saved config YAML for reproducibility

## Citation

If you use this code or methodology, please cite our paper:

**Title:** Material Dynamics Analysis with Deep Generative Model  
(Include DOI or journal reference when available)

Additionally, cite the original PGGAN paper by Karras et al., for foundational GAN methodology.
