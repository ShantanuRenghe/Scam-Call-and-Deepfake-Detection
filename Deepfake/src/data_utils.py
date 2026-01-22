import os
import torch
import torchaudio
import soundfile as sf

# ==============================================================================
# 🚑 EMERGENCY FIX: MONKEY PATCH TORCHAUDIO (Must be at the very top)
# ==============================================================================
# PyTorch 2.4 on Linux defaults to a broken FFmpeg backend that crashes.
# We intercept calls from the 'TTS' library and force them to use 'soundfile'.

print("🔧 Applying Torchaudio Monkey Patch for PyTorch 2.4+...")

# 1. Patch 'load' (Used to read audio)
# 1. Patch 'load'
_real_load = torchaudio.load
def _safe_load(*args, **kwargs):
    # If backend is missing OR explicit None, force 'soundfile'
    if 'backend' not in kwargs or kwargs['backend'] is None:
        kwargs['backend'] = 'soundfile'
    return _real_load(*args, **kwargs)
torchaudio.load = _safe_load

# 2. Patch 'info'
_real_info = torchaudio.info
def _safe_info(*args, **kwargs):
    if 'backend' not in kwargs or kwargs['backend'] is None:
        kwargs['backend'] = 'soundfile'
    return _real_info(*args, **kwargs)
torchaudio.info = _safe_info

print("✅ Torchaudio patched successfully (Force SoundFile).")
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import soundfile as sf
from TTS.api import TTS
import glob
from . import config

from datasets import load_dataset



# ============ Audio Processing Functions ============

def process_audio(audio_array, orig_sr, target_sr=config.TARGET_SAMPLE_RATE, target_duration=config.FIXED_DURATION_SECONDS,
                  n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH):
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


# def get_audio_from_example(example):
#     """Robustly extract audio array and sampling rate from a dataset example.
    
#     Supports several common shapes returned by Hugging Face Audio features:
#       - example['audio'] -> {'array': ..., 'sampling_rate': ...}
#       - example['audio_filepath'] -> path string
#       - example['audio_path'] -> path string
#     Returns (array, sr) or (None, None) if not found.
#     """
#     # Ensure example is a dict
#     if not isinstance(example, dict):
#         print('Example is not a dict:', type(example))
#         return None, None

#     # Case A: 'audio' field (common HF audio feature)
#     if 'audio' in example:
#         audio = example['audio']
#         # audio can be a dict with 'array' and 'sampling_rate'
#         if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
#             return np.asarray(audio['array']), int(audio['sampling_rate'])
#         # audio may also sometimes be an object with 'array'/'sampling_rate' under a different key

#     # Case B: fields like 'audio_filepath', 'audio_path', or 'path'
#     for key in ('audio_filepath', 'audio_path', 'path'):
#         if key in example:
#             val = example[key]
#             # Sometimes these fields are dicts containing array and sampling_rate
#             if isinstance(val, dict) and 'array' in val and 'sampling_rate' in val:
#                 return np.asarray(val['array']), int(val['sampling_rate'])
#             # Or they can be path strings
#             if isinstance(val, str):
#                 path = val
#                 try:
#                     arr, sr = sf.read(path)
#                     return arr, sr
#                 except Exception:
#                     try:
#                         arr, sr = librosa.load(path, sr=None)
#                         return arr, sr
#                     except Exception:
#                         pass

#     # Nothing matched
#     return None, None


def save_spectrogram(spec, out_path, resize_to=config.IMG_SIZE):
    """Resize (if needed) and save spectrogram as .npy."""
    import cv2

    h, w = spec.shape[:2]
    if (h, w) != (resize_to, resize_to):
        spec_resized = cv2.resize(spec.astype(np.float32), (resize_to, resize_to))
    else:
        spec_resized = spec

    # Normalize to [0,1]
    if np.max(spec_resized) != np.min(spec_resized):
        spec_resized = (spec_resized - np.min(spec_resized)) / (np.max(spec_resized) - np.min(spec_resized))

    np.save(out_path, spec_resized)


def process_real_dataset(dataset_name, split='test', out_dir=None, fake_audio_out_dir=None, max_samples=None):
    """
    1. Loads REAL audio -> Converts to Spectrogram -> Saves .npy
    2. Uses REAL audio as reference to Generate FAKE audio (Cloning) -> Saves .wav
    """
    # 1. Setup Directories
    if out_dir is None:
        out_dir = str(config.REAL_DIR)
    
    # We need a place to save the generated .wav files so the next function can read them
    if fake_audio_out_dir is None:
        fake_audio_out_dir = str(config.FAKE_AUDIO_DIR) # Ensure this exists in your config
    
    # We also need a temp folder for reference audios (TTS needs a file path on disk to clone)
    ref_audio_dir = os.path.join(fake_audio_out_dir, "reference_clips")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fake_audio_out_dir, exist_ok=True)
    os.makedirs(ref_audio_dir, exist_ok=True)

    # 2. Load Dataset
    print(f"\nProcessing REAL audio from '{dataset_name}'...")
    dataset_obj = load_dataset(dataset_name)
    
    # Select split
    if isinstance(dataset_obj, dict): # DatasetDict
        if split in dataset_obj:
            real_dataset = dataset_obj[split]
        else:
            real_dataset = dataset_obj[list(dataset_obj.keys())[0]]
    else:
        real_dataset = dataset_obj

    # 3. Initialize TTS Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" ⚡ Initializing XTTS Model on {device}...")
    # Loading XTTS v2 (Good for zero-shot cloning)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    num_to_process = len(real_dataset) if max_samples is None else min(len(real_dataset), max_samples)
    saved_specs = 0
    generated_fakes = 0

    # 4. Processing Loop
    for i in tqdm(range(num_to_process), desc=f"Real Specs & Fake Gen"):
        try:
            # --- A. PREPARE DATA ---
            # Svarah specific structure
            item = real_dataset[i]
            audio_example = item['audio_filepath'] 
            text_transcript = item['text']

            audio_array = np.array(audio_example["array"])
            original_sr = audio_example["sampling_rate"]
            
            # Skip very short audio (TTS struggles to clone < 1.0s)
            duration = len(audio_array) / original_sr
            if duration < 1.5:
                continue

            # --- B. PROCESS REAL SPECTROGRAM ---
            # Check if spec exists to skip calculation
            output_filename = f"real_{i:05d}"
            output_spec_path = os.path.join(out_dir, output_filename)
            
            if not os.path.exists(output_spec_path + '.npy'):
                spectrogram = process_audio(audio_array, original_sr)
                np.save(output_spec_path, spectrogram)
                saved_specs += 1

            # --- C. GENERATE FAKE AUDIO (TTS) ---
            fake_wav_path = os.path.join(fake_audio_out_dir, f"fake_{i:05d}.wav")
            
            # Only generate if it doesn't exist yet
            if not os.path.exists(fake_wav_path):
                # 1. Save Real Audio to disk (TTS needs a path to clone from)
                ref_path = os.path.join(ref_audio_dir, f"ref_{i:05d}.wav")
                sf.write(ref_path, audio_array, original_sr)

                # 2. Run Inference
                # split_sentences=False helps preserve prosody for short clips
                tts.tts_to_file(
                    text=text_transcript,
                    speaker_wav=ref_path,
                    language="en", 
                    file_path=fake_wav_path,
                    split_sentences=False
                )
                generated_fakes += 1
            
        except Exception as e:
            print(f" ⚠ Error at index {i}: {e}")
            continue
    
    print(f"✅ REAL Processing Complete.")
    print(f"   - Real Spectrograms Saved: {saved_specs}")
    print(f"   - Fake Audios Generated:   {generated_fakes}")
    return saved_specs


def process_fake_dataset(fake_audio_dir=None, out_dir=None):
    """
    Scans a directory of generated .wav files -> Converts to Spectrogram -> Saves .npy
    """
    if fake_audio_dir is None:
        fake_audio_dir = str(config.FAKE_AUDIO_DIR)
    
    if out_dir is None:
        out_dir = str(config.FAKE_DIR)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all .wav files in the directory
    wav_files = sorted(glob.glob(os.path.join(fake_audio_dir, "*.wav")))
    
    print(f"\nProcessing FAKE audio from directory: '{fake_audio_dir}'...")
    print(f"Found {len(wav_files)} generated files.")
    
    saved = 0
    
    for wav_path in tqdm(wav_files, desc="Processing Fake Wavs"):
        try:
            # Extract index from filename (assuming format "fake_00123.wav")
            filename = os.path.basename(wav_path)
            file_id = os.path.splitext(filename)[0] # e.g., "fake_00123"
            
            output_path = os.path.join(out_dir, file_id) # e.g., ".../fake_00123.npy"
            
            # Skip if already processed
            if os.path.exists(output_path + '.npy'):
                continue
            
            # Load Audio using soundfile or librosa
            audio_array, sr = sf.read(wav_path)
            
            # Process to spectrogram
            spectrogram = process_audio(audio_array, sr)
            
            # Save
            np.save(output_path, spectrogram)
            saved += 1
            
        except Exception as e:
            print(f" ⚠ Could not process {wav_path}. Error: {e}")
            continue
    
    print(f"✅ Finished processing FAKE spectrograms. Saved: {saved}")
    return saved



# ============ Dataset Splitting ============

def build_and_save_splits(real_dir=None, fake_dir=None, test_size=0.20, val_fraction=0.20, out_path=None, random_state=42):
    """Create stratified train/val/test splits from spectrogram files."""
    real_dir = real_dir or str(config.REAL_DIR)
    fake_dir = fake_dir or str(config.FAKE_DIR)
    
    real_files = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) if f.endswith('.npy')]
    fake_files = [os.path.join(fake_dir, f) for f in sorted(os.listdir(fake_dir)) if f.endswith('.npy')]

    filepaths = np.array(real_files + fake_files)
    labels = np.array([0] * len(real_files) + [1] * len(fake_files))

    if len(filepaths) == 0:
        raise RuntimeError('No spectrograms found in provided directories.')

    X_trainval, X_test, y_trainval, y_test = train_test_split(filepaths, labels, test_size=test_size, random_state=random_state, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_fraction, random_state=random_state, stratify=y_trainval)

    result = {
        'train_paths': X_train, 'train_labels': y_train,
        'val_paths': X_val, 'val_labels': y_val,
        'test_paths': X_test, 'test_labels': y_test,
    }

    if out_path is None:
        out_path = str(config.SPECTRO_DIR / 'splits.npz')

    np.savez_compressed(out_path, **result)
    return result


def load_splits(splits_path=None):
    """Load splits from a .npz file."""
    if splits_path is None:
        splits_path = str(config.SPECTRO_DIR / 'splits.npz')
    data = np.load(splits_path, allow_pickle=True)
    return data


# ============ TensorFlow Dataset Creation ============

def create_tf_dataset_from_splits(split_paths, split_labels, batch_size=config.DEFAULT_BATCH_SIZE, img_size=config.IMG_SIZE, shuffle=True):
    """Create a tf.data.Dataset that loads spectrogram .npy files and yields (image, label)."""
    import tensorflow as tf

    def parse_function(filepath, label):
        def load_and_preprocess(path):
            import cv2
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

    ds = tf.data.Dataset.from_tensor_slices((split_paths.astype('S'), split_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(split_paths))
    ds = ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_tf_datasets_from_splits(data, batch_size=config.DEFAULT_BATCH_SIZE, img_size=config.IMG_SIZE, shuffle=True):
    """Helper to create train/val/test datasets from loaded splits."""
    train_paths = data['train_paths']
    train_labels = data['train_labels']
    val_paths = data['val_paths']
    val_labels = data['val_labels']
    test_paths = data.get('test_paths')
    test_labels = data.get('test_labels')

    train_ds = create_tf_dataset_from_splits(train_paths, train_labels, batch_size=batch_size, img_size=img_size, shuffle=shuffle)
    val_ds = create_tf_dataset_from_splits(val_paths, val_labels, batch_size=batch_size, img_size=img_size, shuffle=False)
    test_ds = None
    if test_paths is not None:
        test_ds = create_tf_dataset_from_splits(test_paths, test_labels, batch_size=batch_size, img_size=img_size, shuffle=False)

    return train_ds, val_ds, test_ds


