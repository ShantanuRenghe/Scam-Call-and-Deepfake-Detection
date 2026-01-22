import torch
import os
from datasets import load_dataset, Audio, load_from_disk
from transformers import WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from config import Config 

def load_and_prepare_dataset(processor: WhisperProcessor):
    """
    Checks if processed data exists. If yes, loads it.
    If no, loads Svarah, processes it, and saves it to disk.
    """
    # --- CACHING LOGIC ---
    if os.path.exists(Config.PROCESSED_DATA_DIR):
        print(f"Found cached dataset at {Config.PROCESSED_DATA_DIR}. Loading directly...")
        try:
            dataset = load_from_disk(Config.PROCESSED_DATA_DIR)
            print("Loaded successfully from cache!")
            return dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing dataset...")

    print(f"Cache not found. Loading raw dataset: {Config.DATASET_NAME}...")
    
    # Load dataset (Removed trust_remote_code based on your previous warning)
    raw_dataset = load_dataset(Config.DATASET_NAME, split=Config.DATASET_SPLIT)
    
    # Select columns
    column_names = raw_dataset.column_names
    audio_col = "audio_filepath" if "audio_filepath" in column_names else "audio"
    text_col = "text"
    
    raw_dataset = raw_dataset.select_columns([audio_col, text_col])
    
    # Split
    print(f"Splitting dataset (Test size: {Config.TEST_SIZE})...")
    dataset = raw_dataset.train_test_split(test_size=Config.TEST_SIZE)
    
    # Cast Audio
    print("Setting sampling rate to 16kHz...")
    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=Config.SAMPLING_RATE))
    
    # Filter function
    def is_audio_in_length_range(ds):
        audio_info = ds[audio_col]
        return len(audio_info["array"]) < Config.MAX_AUDIO_DURATION * Config.SAMPLING_RATE

    print("Filtering audio > 30 seconds...")
    dataset = dataset.filter(is_audio_in_length_range, num_proc=2)

    # Map function
    def prepare_dataset(batch):
        audio = batch[audio_col]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    print("Processing dataset (Spectrograms & Tokenization)...")
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names, 
        num_proc=4
    )
    
    # --- SAVE TO DISK FOR NEXT TIME ---
    print(f"Saving processed dataset to {Config.PROCESSED_DATA_DIR}...")
    dataset.save_to_disk(Config.PROCESSED_DATA_DIR)
    
    return dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch