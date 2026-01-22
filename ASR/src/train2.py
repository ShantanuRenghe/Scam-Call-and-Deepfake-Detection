import torch
import sys
import os
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from transformers.trainer_utils import get_last_checkpoint
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

    # 2. Load Data
    dataset = load_and_prepare_dataset(processor)

    # 3. Initialize Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Load Model
    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_ID)
    
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
        gradient_checkpointing=Config.USE_GRADIENT_CHECKPOINTING,
        fp16=Config.FP16,
        eval_strategy="steps",
        per_device_eval_batch_size=Config.BATCH_SIZE,
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

    # 7. Checkpoint Logic
    checkpoint = None
    if len(sys.argv) > 1:
        # Case 1: User passed a specific checkpoint path
        checkpoint = sys.argv[1]
        print(f"Resuming from specific checkpoint: {checkpoint}")
    elif os.path.isdir(Config.OUTPUT_DIR):
        # Case 2: Auto-detect the latest checkpoint in output_dir
        last_checkpoint = get_last_checkpoint(Config.OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"Found valid checkpoint. Resuming from: {last_checkpoint}")
            checkpoint = last_checkpoint
        else:
            print("No existing checkpoint found. Starting fresh.")

    # 8. Train
    print("Starting training loop...")
    # Pass the 'checkpoint' variable (None = fresh start, Path = resume)
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # 9. Save Final Model
    print("Saving final model...")
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_model")
    processor.save_pretrained(f"{Config.OUTPUT_DIR}/final_model")
    print("Done!")

if __name__ == "__main__":
    main()