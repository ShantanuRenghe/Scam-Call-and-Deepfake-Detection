# 🗣️ Indian Accent ASR (Whisper Fine-tuning)

## Overview

This module focuses on adapting OpenAI's **Whisper Small** model to better understand Indian-accented English. Standard models often struggle with specific regional inflections; this project fine-tunes the model on the **Svarah** dataset to improve transcription accuracy for downstream analysis.

## Key Components

| Component | Role | File |
| :--- | :--- | :--- |
| **Training Script** | Hugging Face Trainer script for fine-tuning Whisper. | `src/train.py` |
| **Inference Script** | Standalone script to transcribe audio files. | `inference.py` |
| **Configuration** | Hyperparameters (Batch size, LR, paths). | `src/config.py` |

## How to Use

### 1. Training
To start fine-tuning the model (handles data download and preprocessing automatically):

```bash
python src/train.py
```
Checkpoints: Saved to ./checkpoints
Logs: Saved to ./logs

### 2. Inference
To transcribe a specific audio file using the fine-tuned model:
```
python inference.py path/to/audio.wav
```
### Configuration
You can adjust training parameters such as learning rate and batch size in src/config.py.
