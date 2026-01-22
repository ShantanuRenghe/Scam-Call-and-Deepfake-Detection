"""Training template for Deepfake project with checkpointing and resume support.

Features:
 - Creates checkpoints using `tf.train.Checkpoint` and `CheckpointManager` (saves model+optimizer+epoch)
 - Supports resuming from the latest checkpoint or a specific checkpoint path
 - Writes TensorBoard logs to `logs/` inside the project
 - Reuses dataset helpers from `detection.py` (splits, tf dataset creation)

Usage examples:
  python train.py --epochs 10                # train from scratch
  python train.py --resume-last              # resume from latest checkpoint in `checkpoints/`
  python train.py --resume /path/to/ckpt-3    # resume from specific checkpoint
  python train.py --checkpoint-dir ./checkpoints --log-dir ./logs
"""

import os
import argparse
import numpy as np
import tensorflow as tf

# Import helper functions from detection.py in the same folder
from .. import detection


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_model(img_size=detection.IMG_SIZE):
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_size, img_size, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def find_latest_checkpoint(checkpoint_dir):
    return tf.train.latest_checkpoint(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(PROJECT_DIR, 'checkpoints'))
    parser.add_argument('--log-dir', type=str, default=os.path.join(PROJECT_DIR, 'logs'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--resume-last', action='store_true', help='Resume from latest checkpoint in checkpoint dir')
    parser.add_argument('--resume', type=str, default=None, help='Path to specific checkpoint to resume from')
    parser.add_argument('--fine-tune-lr', type=float, default=5e-6)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Ensure splits exist
    splits_path = os.path.join(detection.SPECTRO_DIR, 'splits.npz')
    if not os.path.exists(splits_path):
        print('Splits not found; building splits...')
        detection.build_and_save_splits()

    data = np.load(splits_path, allow_pickle=True)
    train_paths = data['train_paths']
    train_labels = data['train_labels']
    val_paths = data['val_paths']
    val_labels = data['val_labels']

    train_ds = detection.create_tf_dataset_from_splits(train_paths, train_labels, batch_size=args.batch_size, img_size=detection.IMG_SIZE, shuffle=True)
    val_ds = detection.create_tf_dataset_from_splits(val_paths, val_labels, batch_size=args.batch_size, img_size=detection.IMG_SIZE, shuffle=False)

    model = build_model(detection.IMG_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Prepare checkpoint manager (saves model+optimizer+epoch)
    ckpt_epoch = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=ckpt_epoch)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.checkpoint_dir, max_to_keep=5)

    initial_epoch = 0
    # Restore if requested
    ckpt_to_restore = None
    if args.resume:
        ckpt_to_restore = args.resume
    elif args.resume_last:
        ckpt_to_restore = find_latest_checkpoint(args.checkpoint_dir)

    if ckpt_to_restore:
        print(f'Restoring checkpoint: {ckpt_to_restore}')
        checkpoint.restore(ckpt_to_restore).expect_partial()
        try:
            initial_epoch = int(checkpoint.epoch.numpy())
            print(f'Resuming from epoch {initial_epoch}')
        except Exception:
            initial_epoch = 0

    # TensorBoard
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)

    # Callback to save checkpoint manager at epoch end
    class ManagerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # epoch is relative to this fit call (starts at 0)
            global_epoch = initial_epoch + epoch + 1
            ckpt_epoch.assign(global_epoch)
            saved_path = manager.save()
            print(f'Checkpoint saved: {saved_path} (epoch={global_epoch})')

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    total_epochs = initial_epoch + args.epochs

    print(f'Starting training from epoch {initial_epoch} to {total_epochs}...')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=[tb_cb, ManagerCallback(), early_stopping]
    )

    print('Training complete.')


if __name__ == '__main__':
    main()
