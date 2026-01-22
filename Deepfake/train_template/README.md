# Deepfake Training Template

This small template demonstrates how to run training with checkpointing and TensorBoard logs for the `Deepfake` project.

Structure
- `train.py` — training script. Saves checkpoints into `checkpoints/` and writes TensorBoard logs to `logs/` by default.

Quick start

1. Create / activate a Python environment and install requirements (see `requirements.txt`).

2. Generate spectrogram splits if you haven't already:

```bash
python -c "from Deepfake import detection; detection.build_and_save_splits()"
```

3. Train from scratch:

```bash
python Deepfake/train_template/train.py --epochs 10
```

4. Resume from latest checkpoint:

```bash
python Deepfake/train_template/train.py --resume-last
```

5. Resume from a specific checkpoint:

```bash
python Deepfake/train_template/train.py --resume /path/to/checkpoints/ckpt-3
```

Notes
- Checkpoints save model + optimizer + epoch (via `tf.train.Checkpoint`).
- TensorBoard logs are written to `logs/` and can be inspected with `tensorboard --logdir logs`.
- This script imports dataset helpers from `Deepfake/detection.py` to reuse preprocessing and split logic.
