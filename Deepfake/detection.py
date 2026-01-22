"""deepfake_final.py

Single-file utility to:
 - (optionally) download/load datasets (Svarah, Deepfake) using Hugging Face
 - convert audio to mel-spectrograms and save as .npy in `spectrograms/real` and `spectrograms/fake`
 - build stratified train/val/test splits and save them to disk
 - provide TF Dataset creation helper to load spectrograms during training

This file is intended to be a consolidated, single-file version of the
dataset creation and preprocessing logic (derived from `deepfake.py` and
the Jupyter notebook). It intentionally keeps processing and splitting
separate so you can re-run only the steps you need.

Usage examples:
  python deepfake_final.py --process --max-samples 1000
  python deepfake_final.py --split --test-size 0.20

Dependencies: librosa, numpy, datasets, tqdm, scikit-learn, opencv-python
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# -------------------------
# Configuration (tweak as needed)
# -------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRO_DIR = os.path.join(PROJECT_DIR, "spectrograms")
REAL_DIR = os.path.join(SPECTRO_DIR, "real")
FAKE_DIR = os.path.join(SPECTRO_DIR, "fake")

IMG_SIZE = 128
TARGET_SAMPLE_RATE = 22050
FIXED_DURATION_SECONDS = 5.0
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def process_audio(audio_array, orig_sr, target_sr=TARGET_SAMPLE_RATE, target_duration=FIXED_DURATION_SECONDS,
                  n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Resample/pad/truncate audio and convert to log-mel spectrogram.

    Returns a 2D numpy array of shape (n_mels, time_frames).
    """
    if orig_sr != target_sr:
        audio = librosa.resample(audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    else:
        audio = audio_array.astype(np.float32)

    target_len = int(target_duration * target_sr)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel


def get_audio_from_example(example):
    """Robustly extract audio array and sampling rate from a dataset example.

    Supports several common shapes returned by Hugging Face Audio features:
      - example['audio'] -> {'array': ..., 'sampling_rate': ...}
      - example['audio_filepath'] -> path string
      - example['audio_path'] -> path string
      - example may already contain raw numpy array fields
    Returns (array, sr) or (None, None) if not found.
    """
    # audio dict
    if isinstance(example, dict):
        if 'audio' in example:
            audio = example['audio']
            if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                return np.asarray(audio['array']), int(audio['sampling_rate'])

        # direct filepath fields
        for key in ('audio_filepath', 'audio_path', 'path'):
            if key in example and isinstance(example[key], str):
                path = example[key]
                try:
                    arr, sr = sf.read(path)
                    return arr, sr
                except Exception:
                    try:
                        arr, sr = librosa.load(path, sr=None)
                        return arr, sr
                    except Exception:
                        pass

    # Otherwise unsupported
    return None, None


def save_spectrogram(spec, out_path, resize_to=IMG_SIZE):
    """Resize (if needed) and save spectrogram as .npy. Spec is 2D np.array."""
    import cv2

    h, w = spec.shape[:2]
    if (h, w) != (resize_to, resize_to):
        # cv2.resize expects float32 or uint8
        spec_resized = cv2.resize(spec.astype(np.float32), (resize_to, resize_to))
    else:
        spec_resized = spec

    # Normalize to [0,1]
    if np.max(spec_resized) != np.min(spec_resized):
        spec_resized = (spec_resized - np.min(spec_resized)) / (np.max(spec_resized) - np.min(spec_resized))

    np.save(out_path, spec_resized)


def process_dataset_and_save(dataset_name=None, split_name=None, out_dir=None, prefix='item', max_samples=None):
    """Load a HF dataset split and save spectrograms into out_dir.

    If dataset_name is None, the function does nothing. Returns number of saved files.
    """
    if load_dataset is None:
        raise RuntimeError('datasets library is not available. Install `datasets` to use this function.')

    ds = load_dataset(dataset_name, split=split_name) if split_name else load_dataset(dataset_name)
    n = len(ds)
    to_process = n if max_samples is None else min(n, max_samples)

    saved = 0
    for i in tqdm(range(to_process), desc=f'Processing {dataset_name}:{split_name}'):
        try:
            example = ds[i]
            audio_arr, sr = get_audio_from_example(example)
            if audio_arr is None:
                # try alternative keys used by some datasets
                # some datasets include 'audio' as a path or token; attempt a last try
                # skip if nothing usable
                continue

            spec = process_audio(audio_arr, sr)
            out_name = f"{prefix}_{i:05d}.npy"
            out_path = os.path.join(out_dir, out_name)
            save_spectrogram(spec, out_path)
            saved += 1
        except Exception:
            continue

    return saved


def build_and_save_splits(real_dir=REAL_DIR, fake_dir=FAKE_DIR, test_size=0.20, val_fraction=0.20, out_path=None, random_state=42):
    """Create stratified train/val/test splits from files in real_dir and fake_dir.

    Saves a compressed numpy file containing the lists and labels.
    Returns a dict with arrays.
    """
    real_files = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) if f.endswith('.npy')]
    fake_files = [os.path.join(fake_dir, f) for f in sorted(os.listdir(fake_dir)) if f.endswith('.npy')]

    filepaths = np.array(real_files + fake_files)
    labels = np.array([0] * len(real_files) + [1] * len(fake_files))

    if len(filepaths) == 0:
        raise RuntimeError('No spectrograms found in provided directories.')

    X_trainval, X_test, y_trainval, y_test = train_test_split(filepaths, labels, test_size=test_size, random_state=random_state, stratify=labels)
    val_size = val_fraction
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, random_state=random_state, stratify=y_trainval)

    result = {
        'train_paths': X_train, 'train_labels': y_train,
        'val_paths': X_val, 'val_labels': y_val,
        'test_paths': X_test, 'test_labels': y_test,
    }

    if out_path is None:
        out_path = os.path.join(SPECTRO_DIR, 'splits.npz')

    np.savez_compressed(out_path, **result)
    return result


def create_tf_dataset_from_splits(split_paths, split_labels, batch_size=32, img_size=IMG_SIZE, shuffle=True):
    """Create a tf.data.Dataset that loads spectrogram .npy files and yields (image, label).

    This mirrors the notebook approach using tf.py_function to load numpy arrays.
    """
    import tensorflow as tf

    def parse_function(filepath, label):
        def load_and_preprocess(path):
            path_str = path.numpy().decode('utf-8')
            arr = np.load(path_str)
            # Ensure shape and channels
            if arr.ndim == 2:
                spec = arr
            elif arr.ndim == 3:
                spec = arr[:, :, 0]
            else:
                spec = arr.reshape((img_size, img_size))

            # resize if necessary
            import cv2
            if spec.shape[:2] != (img_size, img_size):
                spec = cv2.resize(spec.astype(np.float32), (img_size, img_size))

            # normalize
            if np.max(spec) != np.min(spec):
                spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

            # expand to 3 channels
            spec = np.expand_dims(spec, -1)
            spec = np.repeat(spec, 3, axis=-1)
            return spec.astype(np.float32)

        image = tf.py_function(func=lambda p: load_and_preprocess(p), inp=[filepath], Tout=tf.float32)
        image.set_shape([img_size, img_size, 3])
        label = tf.cast(label, tf.int32)
        label.set_shape([])
        return image, label

    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((split_paths.astype('S'), split_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(split_paths))
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', action='store_true', help='Process HF datasets into spectrograms')
    parser.add_argument('--process-only-if-missing', dest='process_only_if_missing', action='store_true', help='Skip processing if spectrogram folders are non-empty (default)')
    parser.add_argument('--no-process-only-if-missing', dest='process_only_if_missing', action='store_false', help='Do not skip processing even if spectrogram folders contain files')
    parser.set_defaults(process_only_if_missing=True)
    parser.add_argument('--dataset-svarah', default='ai4bharat/Svarah', help='Hugging Face dataset for real audio')
    parser.add_argument('--dataset-fake', default='ar17to/orpheus_tts_english_indian_multispeaker', help='Hugging Face dataset for fake audio')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit number of samples per dataset')
    parser.add_argument('--split', action='store_true', help='Create and save train/val/test splits from spectrogram folders')
    parser.add_argument('--test-size', type=float, default=0.20)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    args = parser.parse_args()

    if args.process:
        # If folders already have files and --process-only-if-missing provided, skip
        exist_real = len([f for f in os.listdir(REAL_DIR) if f.endswith('.npy')])
        exist_fake = len([f for f in os.listdir(FAKE_DIR) if f.endswith('.npy')])
        if args.process_only_if_missing and exist_real + exist_fake > 0:
            print('Spectrogram folders already contain files; skipping processing.')
        else:
            print('Processing REAL dataset (Svarah) ...')
            try:
                n_saved_real = process_dataset_and_save(dataset_name=args.dataset_svarah, split_name=None, out_dir=REAL_DIR, prefix='real', max_samples=args.max_samples)
            except Exception as e:
                print(f'Could not process Svarah dataset: {e}')
                n_saved_real = 0

            print('Processing FAKE dataset (Deepfake TTS) ...')
            try:
                # many HF datasets use train/validation splits; try 'train' first
                n_saved_fake = process_dataset_and_save(dataset_name=args.dataset_fake, split_name='train', out_dir=FAKE_DIR, prefix='fake', max_samples=args.max_samples)
                if n_saved_fake == 0:
                    # fallback to root split
                    n_saved_fake = process_dataset_and_save(dataset_name=args.dataset_fake, split_name=None, out_dir=FAKE_DIR, prefix='fake', max_samples=args.max_samples)
            except Exception as e:
                print(f'Could not process fake dataset: {e}')
                n_saved_fake = 0

            print(f'Saved {n_saved_real} real and {n_saved_fake} fake spectrograms.')

    if args.split:
        print('Building and saving train/val/test splits...')
        result = build_and_save_splits(real_dir=REAL_DIR, fake_dir=FAKE_DIR, test_size=args.test_size, val_fraction=args.val_fraction)
        out_path = os.path.join(SPECTRO_DIR, 'splits.npz')
        print(f"Saved splits to {out_path}")

    # Training flags
    parser.add_argument('--train', action='store_true', help='Train model using saved splits')
    parser.add_argument('--initial-epochs', type=int, default=20, help='Initial training epochs')
    parser.add_argument('--fine-tune-epochs', type=int, default=20, help='Fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--fine-tune-lr', type=float, default=5e-6, help='Fine-tune learning rate')

    # Parse again to capture new training args
    args = parser.parse_args()

    if args.train:
        # Ensure TensorFlow is available
        try:
            import tensorflow as tf
        except Exception as e:
            raise RuntimeError('TensorFlow is required for training. Install tensorflow.')

        # Force CPU-only execution by default to match original script behavior
        import os as _os
        _os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

        # Load splits
        splits_path = os.path.join(SPECTRO_DIR, 'splits.npz')
        if not os.path.exists(splits_path):
            print('Splits file not found; building splits automatically...')
            build_and_save_splits(real_dir=REAL_DIR, fake_dir=FAKE_DIR, test_size=args.test_size, val_fraction=args.val_fraction)

        data = np.load(splits_path, allow_pickle=True)
        train_paths = data['train_paths']
        train_labels = data['train_labels']
        val_paths = data['val_paths']
        val_labels = data['val_labels']
        test_paths = data['test_paths']
        test_labels = data['test_labels']

        # Create tf.data datasets
        train_ds = create_tf_dataset_from_splits(train_paths, train_labels, batch_size=args.batch_size, img_size=IMG_SIZE, shuffle=True)
        val_ds = create_tf_dataset_from_splits(val_paths, val_labels, batch_size=args.batch_size, img_size=IMG_SIZE, shuffle=False)
        test_ds = create_tf_dataset_from_splits(test_paths, test_labels, batch_size=args.batch_size, img_size=IMG_SIZE, shuffle=False)

        # Build model (EfficientNetB0 backbone)
        print('🛠 Building model with EfficientNetB0 base...')
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.summary()

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Initial training
        print(f"\n🚂 Starting initial training for up to {args.initial_epochs} epochs...")
        history = model.fit(train_ds, epochs=args.initial_epochs, validation_data=val_ds, callbacks=[early_stopping])

        # Fine-tuning
        print('\n⚙ Preparing for fine-tuning...')
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 10
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(f'Fine-tuning from layer {fine_tune_at} onwards.')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        initial_epochs_trained = len(history.epoch)
        total_epochs = initial_epochs_trained + args.fine_tune_epochs

        early_stopping_ft = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print(f"\n🚂 Starting fine-tuning for up to {args.fine_tune_epochs} epochs...")
        history_fine = model.fit(train_ds, epochs=total_epochs, initial_epoch=initial_epochs_trained, validation_data=val_ds, callbacks=[early_stopping_ft])

        # Evaluation on test set
        print('\n📊 Evaluating the final model on the unseen test set...')
        loss, acc = model.evaluate(test_ds)
        print(f"\nFinal Test Accuracy: {acc:.4f}")
        print(f"Final Test Loss: {loss:.4f}")

        # Generate predictions by iterating test_ds
        y_true = []
        y_probs = []
        for xb, yb in test_ds:
            preds = model.predict(xb)
            y_probs.extend(preds.flatten().tolist())
            y_true.extend(yb.numpy().astype(int).tolist())

        y_pred_classes = (np.array(y_probs) > 0.5).astype(int)

        # Metrics and plots
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        import matplotlib.pyplot as plt
        import seaborn as sns

        print('\nClassification Report:')
        CLASSES = ['real', 'fake']
        print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={'size':16})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix on Test Data')
        plt.show()

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # Sample predictions
        import numpy as _np
        plt.figure(figsize=(15,10))
        num_samples_to_show = min(10, len(y_true))
        sample_indices = _np.random.choice(len(y_true), num_samples_to_show, replace=False)

        # We need raw images for display; load them from paths
        for i, idx in enumerate(sample_indices):
            path = test_paths[idx]
            spec = np.load(path)
            if spec.ndim == 3:
                spec_disp = spec[:, :, 0]
            else:
                spec_disp = spec

            plt.subplot(2, 5, i+1)
            plt.imshow(spec_disp, cmap='viridis')
            true_label = CLASSES[int(test_labels[idx])]
            pred_prob = float(y_probs[idx])
            pred_label = CLASSES[int(y_pred_classes[idx])]
            title_color = 'green' if true_label == pred_label else 'red'
            plt.title(f"True: {true_label}\nPred: {pred_label} ({pred_prob:.2f})", color=title_color, fontsize=10)
            plt.axis('off')

        plt.suptitle('Sample Test Predictions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


if __name__ == '__main__':
    main()
