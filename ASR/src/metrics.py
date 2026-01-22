import evaluate

# Load metric globally
wer_metric = evaluate.load("wer")

def compute_metrics_wrapper(processor):
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calculate WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        # --- NEW: PRINT SAMPLES TO CONSOLE ---
        print("\n" + "="*50)
        print(f"Global Step WER: {wer*100:.2f}%")
        print("Sample Predictions vs Ground Truth:")
        for i in range(min(3, len(pred_str))): # Print top 3
            print(f"\nRef:  {label_str[i]}")
            print(f"Pred: {pred_str[i]}")
        print("="*50 + "\n")
        # -------------------------------------

        return {"wer": 100 * wer}
    
    return compute_metrics