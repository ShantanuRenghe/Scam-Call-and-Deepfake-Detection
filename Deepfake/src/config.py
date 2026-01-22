import os
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SPECTRO_DIR = ROOT / 'spectrograms'
REAL_DIR = SPECTRO_DIR / 'real'
FAKE_DIR = SPECTRO_DIR / 'fake'
CHECKPOINTS_DIR = ROOT / 'checkpoints'
LOGS_DIR = ROOT / 'logs'
EVALUATION_DIR = ROOT / 'evaluation'
FAKE_AUDIO_DIR = ROOT / 'fake_audio'

# Model / audio constants (kept in sync with detection.py)
IMG_SIZE = 128
TARGET_SAMPLE_RATE = 22050
FIXED_DURATION_SECONDS = 5.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)
