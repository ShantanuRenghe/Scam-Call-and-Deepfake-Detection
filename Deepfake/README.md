# 🕵️ Deepfake Audio Detection

## Overview

This module detects AI-generated (synthetic) speech by treating audio analysis as an image classification problem. It converts raw audio into **Mel-spectrograms** and trains an **EfficientNetB0** CNN to distinguish between real human voice and deepfake artifacts.

## Features

* **Spectrogram Conversion:** Robust pipeline to convert audio to 128x128 Mel-spectrogram images.
* **Unified Pipeline:** Single-command data processing, splitting, and training.
* **Transfer Learning:** Uses pre-trained EfficientNetB0 (ImageNet weights) with a fine-tuning phase.
* **Visualization:** Generates ROC curves, Confusion Matrices, and sample prediction grids.

## Pipeline Workflow

| Step | Script | Description |
| :--- | :--- | :--- |
| **1. Process** | `detection.py` | Downloads datasets (Svarah vs. Orpheus TTS) and saves spectrograms as `.npy` files. |
| **2. Split** | `detection.py` | Creates stratified Train/Val/Test splits. |
| **3. Train** | `src/train.py` | Trains the classifier and saves the model. |

## How to Use

### 1. Data Preparation
Download datasets and convert audio to spectrograms:
```bash
python detection.py --process --max-samples 2000
```
Generate train/test splits:

```Bash
python detection.py --split
```
### 2. Training
Run the unified training pipeline (Initial training -> Fine-tuning -> Evaluation):
```
# Run from the root directory or ensure python path is set
python -m Deepfake.src.train --train --fine-tune --epochs 20
```
### 3. Evaluation Results
After training, evaluation metrics (ROC Curve, Accuracy, Reports) are automatically saved to Deepfake/evaluation/<timestamp>/.
