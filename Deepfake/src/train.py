"""Unified training pipeline: process -> split -> train -> fine-tune.

This is the single entry point for all Deepfake training.
All logic is self-contained in src/ modules; detection.py is NOT imported.

Key sections:
  - build_model(): creates EfficientNetB0-based classifier.
  - train(): initial training with checkpointing and TensorBoard.
  - fine_tune(): unfreezes top layers and trains with lower learning rate.
  - evaluate_and_save(): evaluates on test set and saves graphs/stats with timestamped subfolder.
  - main(): CLI orchestrating the full pipeline.
"""
import os
import argparse
import json
from datetime import datetime
import tensorflow as tf
import numpy as np

from . import config
from . import data_utils
from . import visualize


def get_timestamp_dir(base_dir):
    """Create a timestamped subfolder for this training run.
    
    Format: YYYY-MM-DD_HH-MM-SS
    Returns the full path to the subfolder.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_model(img_size=config.IMG_SIZE):
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


def train(checkpoint_dir=None, log_dir=None, epochs=20, batch_size=None, learning_rate=None, resume_last=False, resume_path=None, test_ds=None):
    """Initial training phase: trains model from scratch or resumes from checkpoint.
    
    Args:
        checkpoint_dir: where to save checkpoints
        log_dir: where to save TensorBoard logs
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: initial learning rate
        resume_last: if True, resume from latest checkpoint
        resume_path: specific checkpoint path to resume from
        test_ds: optional test dataset for evaluation after training
    
    Returns:
        (model, checkpoint_manager, checkpoint, initial_epoch, test_paths, test_labels) for potential fine-tuning
    """
    checkpoint_dir = checkpoint_dir or str(config.CHECKPOINTS_DIR)
    log_dir = log_dir or str(config.LOGS_DIR)
    batch_size = batch_size or config.DEFAULT_BATCH_SIZE
    learning_rate = learning_rate or config.DEFAULT_LEARNING_RATE

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load splits
    splits_path = str(config.SPECTRO_DIR / 'splits.npz')
    if not os.path.exists(splits_path):
        print('Splits not found; building splits...')
        data_utils.build_and_save_splits()

    data = data_utils.load_splits(splits_path)
    train_ds, val_ds, test_ds_ret = data_utils.make_tf_datasets_from_splits(data, batch_size=batch_size, img_size=config.IMG_SIZE, shuffle=True)
    test_paths = data.get('test_paths')
    test_labels = data.get('test_labels')

    model = build_model(config.IMG_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Checkpoints
    ckpt_epoch = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=ckpt_epoch)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

    initial_epoch = 0
    ckpt_to_restore = resume_path if resume_path else (tf.train.latest_checkpoint(checkpoint_dir) if resume_last else None)
    if ckpt_to_restore:
        print(f'Restoring checkpoint: {ckpt_to_restore}')
        checkpoint.restore(ckpt_to_restore).expect_partial()
        try:
            initial_epoch = int(checkpoint.epoch.numpy())
            print(f'Resuming from epoch {initial_epoch}')
        except Exception:
            initial_epoch = 0

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    class ManagerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global_epoch = initial_epoch + epoch + 1
            ckpt_epoch.assign(global_epoch)
            saved = manager.save()
            # also write metadata
            meta = {'epoch': int(ckpt_epoch.numpy()), 'phase': 'initial_training'}
            meta_path = os.path.join(checkpoint_dir, f"{os.path.basename(saved)}.meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            print(f'Checkpoint saved: {saved} (epoch={global_epoch})')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    total_epochs = initial_epoch + epochs
    print(f'Starting initial training from epoch {initial_epoch} to {total_epochs}...')
    model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, initial_epoch=initial_epoch, callbacks=[tb_cb, ManagerCallback(), early_stopping])

    return model, manager, checkpoint, initial_epoch, test_paths, test_labels


def evaluate_and_save(model, test_ds, test_paths, test_labels, evaluation_dir=None, phase_label='initial_training'):
    """Evaluate model on test set and save all graphs/stats in a timestamped subfolder.
    
    Args:
        model: trained Keras model
        test_ds: test tf.data.Dataset
        test_paths: paths to test spectrograms
        test_labels: true test labels
        evaluation_dir: base directory for evaluation (defaults to config.EVALUATION_DIR)
        phase_label: 'initial_training' or 'fine_tune' to organize results
    
    Returns:
        dict with evaluation metrics
    """
    evaluation_dir = evaluation_dir or str(config.EVALUATION_DIR)
    
    # Create timestamped subfolder for this run
    phase_dir = os.path.join(evaluation_dir, phase_label)
    run_dir = get_timestamp_dir(phase_dir)
    
    print('\n' + '='*60)
    print(f'EVALUATION ON TEST SET ({phase_label})')
    print(f'Results will be saved to: {run_dir}')
    print('='*60)
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Generate predictions
    y_true = []
    y_probs = []
    for xb, yb in test_ds:
        preds = model.predict(xb, verbose=0)
        y_probs.extend(preds.flatten().tolist())
        y_true.extend(yb.numpy().astype(int).tolist())
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred_classes = (y_probs > 0.5).astype(int)
    
    # Generate classification report
    report = visualize.save_classification_report(
        y_true, y_pred_classes, 
        save_path=os.path.join(run_dir, 'classification_report.json'),
        class_names=('real', 'fake')
    )
    print("\nClassification Report:")
    print(f"  Precision (real): {report['real']['precision']:.4f}")
    print(f"  Recall (real): {report['real']['recall']:.4f}")
    print(f"  Precision (fake): {report['fake']['precision']:.4f}")
    print(f"  Recall (fake): {report['fake']['recall']:.4f}")
    
    # Plot and save confusion matrix
    visualize.plot_confusion_matrix_and_save(
        y_true, y_pred_classes,
        save_path=os.path.join(run_dir, 'confusion_matrix.png')
    )
    
    # Plot and save ROC curve
    roc_auc = visualize.plot_roc_and_save(
        y_true, y_probs,
        save_path=os.path.join(run_dir, 'roc_curve.png')
    )
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Save sample predictions
    visualize.show_sample_predictions_and_save(
        test_paths, y_true, y_probs, y_pred_classes,
        save_path=os.path.join(run_dir, 'sample_predictions.png'),
        num_samples=10
    )
    
    # Save comprehensive evaluation summary
    summary = visualize.save_evaluation_summary(
        y_true, y_probs, y_pred_classes, loss, accuracy, roc_auc,
        save_path=os.path.join(run_dir, 'evaluation_summary.json')
    )
    
    print(f'\n✓ All evaluation results saved to {run_dir}')
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'roc_auc': roc_auc,
        'classification_report': report,
        'summary': summary
    }


def fine_tune(model, checkpoint_manager, initial_epoch, train_ds, val_ds, fine_tune_lr=5e-6, epochs=10, log_dir=None, checkpoint_dir=None, test_ds=None, test_paths=None, test_labels=None, evaluation_dir=None):
    """Fine-tune the model by unfreezing top layers of the base model.
    
    ==== FINE-TUNING HAPPENS HERE ====
    This function:
      1. Unfreezes the top 10 layers of the EfficientNetB0 base model (keeps bottom layers frozen).
      2. Recompiles with a lower learning rate (default 5e-6) to avoid catastrophic forgetting.
      3. Trains for additional epochs with checkpointing and early stopping.
      4. Saves updated checkpoints to track fine-tuning progress separately.
    
    Args:
        model: trained Keras model
        checkpoint_manager: tf.train.CheckpointManager
        initial_epoch: current epoch count
        train_ds: training tf.data.Dataset
        val_ds: validation tf.data.Dataset
        fine_tune_lr: learning rate for fine-tuning (lower than initial training)
        epochs: number of fine-tuning epochs
        log_dir: directory for TensorBoard logs
        checkpoint_dir: directory for checkpoints
    
    Returns:
        history: training history
    """
    log_dir = log_dir or str(config.LOGS_DIR)
    checkpoint_dir = checkpoint_dir or str(config.CHECKPOINTS_DIR)
    
    print("\n" + "="*60)
    print("FINE-TUNING PHASE: Unfreezing top layers of base model")
    print("="*60)
    
    # Unfreeze top layers of base model
    base_model = model  # EfficientNetB0 is the second layer (after Input)
    base_model.trainable = True
    
    # Keep bottom 70% frozen, unfreeze top 30% (roughly last 10 layers for EfficientNetB0)
    fine_tune_at = len(base_model.layers) - 10
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Freezing layers 0-{fine_tune_at-1}, unfreezing {fine_tune_at}-{len(base_model.layers)-1}")
    
    # Recompile with lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    print(f"Recompiled model with fine-tune learning rate: {fine_tune_lr}")
    model.summary()
    
    # TensorBoard for fine-tuning phase
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # Callback to save checkpoints during fine-tuning
    ckpt_epoch = tf.Variable(initial_epoch, dtype=tf.int64)
    checkpoint = checkpoint_manager.checkpoint if hasattr(checkpoint_manager, 'checkpoint') else None
    
    class FineTuneManagerCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global_epoch = initial_epoch + epoch + 1
            ckpt_epoch.assign(global_epoch)
            if checkpoint:
                checkpoint.epoch.assign(global_epoch)
            saved = checkpoint_manager.save()
            meta = {'epoch': int(ckpt_epoch.numpy()), 'phase': 'fine_tune'}
            meta_path = os.path.join(checkpoint_dir, f"{os.path.basename(saved)}.meta.json")
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            print(f"Checkpoint saved: {saved} (epoch={global_epoch})")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    total_epochs = initial_epoch + epochs
    print(f"\nStarting fine-tuning from epoch {initial_epoch} to {total_epochs}...")
    history = model.fit(train_ds, validation_data=val_ds, 
                       epochs=total_epochs, 
                       initial_epoch=initial_epoch, 
                       callbacks=[tb_cb, FineTuneManagerCallback(), early_stopping])
    
    print("Fine-tuning complete.")
    return history


def main():
    """CLI entry point for the unified pipeline: process -> split -> train -> fine-tune."""
    parser = argparse.ArgumentParser(description='Deepfake Detection: Unified Training Pipeline')
    parser.add_argument('--process', action='store_true', help='Process HF datasets into spectrograms')
    parser.add_argument('--dataset-real', default='ai4bharat/Svarah', help='HF dataset for real audio')
    parser.add_argument('--dataset-fake', default='ar17to/orpheus_tts_english_indian_multispeaker', help='HF dataset for fake audio')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per dataset')
    
    parser.add_argument('--split', action='store_true', help='Build train/val/test splits')
    
    parser.add_argument('--train', action='store_true', help='Run initial training')
    parser.add_argument('--fine-tune', action='store_true', help='Run fine-tuning (requires trained model)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.DEFAULT_LEARNING_RATE, help='Initial learning rate')
    parser.add_argument('--fine-tune-lr', type=float, default=5e-6, help='Fine-tuning learning rate')
    parser.add_argument('--fine-tune-epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--resume-last', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Resume from specific checkpoint')
    
    args = parser.parse_args()

    if args.process:
        print('Processing datasets...')
        # Use the simple, direct dataset handlers
        data_utils.process_real_dataset(dataset_name=args.dataset_real, split = 'test', max_samples=args.max_samples)
        data_utils.process_fake_dataset()

    if args.split:
        print('Building splits...')
        data_utils.build_and_save_splits()

    if args.train:
        print('\n' + '='*60)
        print('INITIAL TRAINING PHASE')
        print('='*60)
        model, manager, checkpoint, initial_epoch, test_paths, test_labels = train(
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.lr, 
            resume_last=args.resume_last, 
            resume_path=args.resume
        )
        print('Initial training complete.')
        
        # Load test dataset for evaluation
        splits_path = str(config.SPECTRO_DIR / 'splits.npz')
        data = data_utils.load_splits(splits_path)
        _, _, test_ds = data_utils.make_tf_datasets_from_splits(data, batch_size=args.batch_size)
        
        # Evaluate after training
        if test_ds is not None:
            eval_results = evaluate_and_save(model, test_ds, test_paths, test_labels, 
                                            phase_label='initial_training')
        
        # Auto fine-tune if requested
        if args.fine_tune:
            # Reload splits to get datasets
            splits_path = str(config.SPECTRO_DIR / 'splits.npz')
            data = data_utils.load_splits(splits_path)
            train_ds, val_ds, test_ds = data_utils.make_tf_datasets_from_splits(data, batch_size=args.batch_size)
            
            fine_tune(
                model=model,
                checkpoint_manager=manager,
                initial_epoch=initial_epoch + args.epochs,
                train_ds=train_ds,
                val_ds=val_ds,
                fine_tune_lr=args.fine_tune_lr,
                epochs=args.fine_tune_epochs,
                test_ds=test_ds,
                test_paths=test_paths,
                test_labels=test_labels
            )
            
            # Evaluate after fine-tuning
            if test_ds is not None:
                eval_results_ft = evaluate_and_save(model, test_ds, test_paths, test_labels, 
                                                     phase_label='fine_tune')

    elif args.fine_tune:
        # Fine-tune without training first
        print("Fine-tuning requires a trained model. Use --train --fine-tune together or --resume-last --fine-tune to resume and fine-tune.")


if __name__ == '__main__':
    main()
