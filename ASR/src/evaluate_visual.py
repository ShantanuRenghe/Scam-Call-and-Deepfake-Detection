import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_from_disk
from config import Config
import pandas as pd
from tqdm import tqdm
import jiwer

def evaluate_visual():
    # 1. Paths
    MODEL_PATH = f"{Config.OUTPUT_DIR}/final_model" # Or use a checkpoint folder
    CACHE_DIR = Config.PROCESSED_DATA_DIR
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to("cuda")
        processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did training finish? Try changing MODEL_PATH to a specific checkpoint (e.g., checkpoints/checkpoint-1000)")
        return

    # 2. Load Test Data
    print(f"Loading processed data from {CACHE_DIR}...")
    dataset = load_from_disk(CACHE_DIR)["test"]
    
    # Select a subset for quick visualization (e.g., first 50 examples)
    # Remove the [:50] slicing if you want to evaluate the WHOLE test set
    eval_dataset = dataset.select(range(min(100, len(dataset)))) 

    results = []
    print("Running inference...")

    model.eval()
    
    # 3. Inference Loop
    for i, item in enumerate(tqdm(eval_dataset)):
        # Move input to GPU
        input_features = torch.tensor(item["input_features"]).unsqueeze(0).to("cuda")
        
        # Generate token ids
        with torch.no_grad():
            generated_ids = model.generate(input_features, max_length=225)
        
        # Decode
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Decode Reference (Ground Truth)
        # We have to handle the label padding (-100) if present
        reference_ids = [token for token in item["labels"] if token != -100]
        reference = processor.decode(reference_ids, skip_special_tokens=True)
        
        # Calculate individual WER for this sentence
        wer = jiwer.wer(reference, transcription)
        
        results.append({
            "Reference": reference,
            "Prediction": transcription,
            "WER": round(wer, 4)
        })

    # 4. Save to CSV
    df = pd.DataFrame(results)
    csv_path = "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    # 5. Print Summary
    print("\n" + "="*30)
    print(" EVALUATION SUMMARY ")
    print("="*30)
    print(df.head(5))
    print(f"\nResults saved to {csv_path}")
    print(f"Average WER on this sample: {df['WER'].mean()*100:.2f}%")

if __name__ == "__main__":
    evaluate_visual()