import torch
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from config import Config
from data_utils import load_and_prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding
from metrics import compute_metrics_wrapper

def main():
    print(f"--- Starting Training for {Config.MODEL_ID} ---")

    # 1. Initialize Processor
    processor = WhisperProcessor.from_pretrained(
        Config.MODEL_ID, 
        language=Config.LANGUAGE, 
        task=Config.TASK
    )

    # 2. Load Data (Will use cache if available)
    dataset = load_and_prepare_dataset(processor)

    # 3. Initialize Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Load Model
    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_ID)
    
    # Cleanup model config
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = Config.USE_CACHE 
    
    # --- CONDITIONAL GRADIENT CHECKPOINTING ---
    if Config.USE_GRADIENT_CHECKPOINTING:
        print("Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        print("Gradient Checkpointing DISABLED (More stable).")
    # ------------------------------------------
    
    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUMULATION,
        learning_rate=Config.LEARNING_RATE,
        warmup_steps=Config.WARMUP_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        # Only enable if Config says so
        gradient_checkpointing=Config.USE_GRADIENT_CHECKPOINTING,
        fp16=Config.FP16,
        eval_strategy="steps",
        per_device_eval_batch_size=Config.BATCH_SIZE, # Eval usually fits fine
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=Config.SAVE_STEPS,
        eval_steps=Config.EVAL_STEPS,
        logging_steps=Config.LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_dir=Config.LOGGING_DIR,
        push_to_hub=False,
        remove_unused_columns=False, 
    )

    # 6. Initialize Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper(processor),
        processing_class=processor.feature_extractor,
    )

    # 7. Train
    print("Starting training loop...")
    trainer.train()
    
    # 8. Save Final Model
    print("Saving final model...")
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{Config.OUTPUT_DIR}/final_model")
    print("Done!")

if __name__ == "__main__":
    main()