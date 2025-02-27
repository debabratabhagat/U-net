# U-net

## Overview

This repository contains an implementation of the U-Net model for salt segmentation using the TGS Salt Segmentation dataset available on Kaggle. The model is trained to predict salt deposits in seismic images using deep learning.

## Dataset

- The dataset consists of grayscale images and corresponding binary masks indicating the presence of salt.
- It is organized into:
  - `dataset/train/images/` (Training images)
  - `dataset/train/masks/` (Ground truth segmentation masks)

## Model Configuration

- **Architecture:** U-Net with 3 levels
- **Input:** 128x128 grayscale images
- **Output:** Binary mask prediction
- **Loss Function:** Binary Cross-Entropy with Dice Loss
- **Optimizer:** Adam

## Training Setup

- **Device:** CUDA (if available) or CPU
- **Batch Size:** 64
- **Learning Rate:** 0.001
- **Epochs:** 40
- **Workers:** 5 (for data loading)
- **Threshold:** 0.5 (for binary mask prediction)

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

### 3. Test the Model

```bash
python predict.py
```

## Output

- **Trained Model:** `output/unet_tgs_salt.pth`
- **Training Plot:** `output/plot.png`
- **Test Image Paths:** `output/test_paths.txt`
