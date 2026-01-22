import os
import torch
import soundfile as sf
import argparse
from datasets import load_dataset
from TTS.api import TTS

# --- 1. SETUP CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="Generate a Deepfake Dataset from Svarah")

parser.add_argument(
    "--count", 
    type=int, 
    default=None, 
    help="Number of NEW pairs to generate. If not set, processes the FULL dataset."
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    default="svarah_deepfake_dataset", 
    help="Directory to save the dataset."
)

args = parser.parse_args()
TARGET_COUNT = args.count
OUTPUT_DIR = args.output_dir

# --- 2. CONFIGURATION ---
DATASET_ID = "ai4bharat/Svarah"
SPLIT = "test"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directories
real_dir = os.path.join(OUTPUT_DIR, "real")
fake_dir = os.path.join(OUTPUT_DIR, "fake")
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

if TARGET_COUNT is None:
    print(f"Goal: Process ALL remaining samples in {DATASET_ID}.")
else:
    print(f"Goal: Generate {TARGET_COUNT} NEW pairs.")

print(f"Loading {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split=SPLIT)

print(f"Initializing XTTS-v2 on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# --- 3. GENERATION LOOP ---
generated_count = 0 

print("Starting generation...")

for i, sample in enumerate(dataset):
    
    # STOP CONDITION
    if TARGET_COUNT is not None and generated_count >= TARGET_COUNT:
        print(f"\n--- Target of {TARGET_COUNT} new pairs reached! ---")
        break

    try:
        # Define filenames early to check existence
        real_filename = f"svarah_{i}_real.wav"
        fake_filename = f"svarah_{i}_fake.wav"
        
        real_path = os.path.join(real_dir, real_filename)
        fake_path = os.path.join(fake_dir, fake_filename)

        # --- CHECK: DO FILES ALREADY EXIST? ---
        if os.path.exists(real_path) and os.path.exists(fake_path):
            # If they exist, we just skip silently (or print if you prefer verbose logs)
            # We do NOT increment 'generated_count' because we didn't do work.
            # print(f"Skipping index {i} (Files already exist)")
            continue

        # --- A. PREPARE INPUTS ---
        text = sample['text']
        audio_data = sample['audio_filepath']['array']
        sample_rate = sample['audio_filepath']['sampling_rate']
        
        # Filter: Skip short audios (< 2.0s)
        duration = sample.get('duration', len(audio_data)/sample_rate)
        if duration < 2.0:
            continue

        # --- B. SAVE REAL AUDIO ---
        sf.write(real_path, audio_data, sample_rate)

        # --- C. GENERATE DEEPFAKE AUDIO ---
        tts.tts_to_file(
            text=text,
            speaker_wav=real_path,
            language="en",
            file_path=fake_path,
            split_sentences=False
        )
        
        generated_count += 1
        
        # Dynamic print
        progress_msg = f"[{generated_count}]" if TARGET_COUNT is None else f"[{generated_count}/{TARGET_COUNT}]"
        print(f"{progress_msg} Generated pair for index {i}")

    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")

print(f"\nProcessing complete. New pairs generated: {generated_count}")
print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")