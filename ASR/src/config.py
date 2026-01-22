import os

class Config:
    # Model Parameters
    MODEL_ID = "openai/whisper-small"
    LANGUAGE = "English"
    TASK = "transcribe"
    
    # Data Parameters
    DATASET_NAME = "ai4bharat/Svarah"
    DATASET_SPLIT = "test"
    TEST_SIZE = 0.1
    MAX_AUDIO_DURATION = 30.0
    SAMPLING_RATE = 16000
    
    # CACHING
    PROCESSED_DATA_DIR = "./processed_data_cache"
    
    # Training Parameters
    OUTPUT_DIR = "./checkpoints"
    LOGGING_DIR = "./logs"
    
    # --- MEMORY OPTIMIZATION SETTINGS ---
    # Reduced batch size to fit in memory without checkpointing
    BATCH_SIZE = 8          
    # Increased accumulation to maintain effective batch size of 16
    GRAD_ACCUMULATION = 2   
    
    # Disable this to fix the RuntimeError
    USE_GRADIENT_CHECKPOINTING = False
    # ------------------------------------

    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    
    # Checkpointing
    SAVE_STEPS = 1000
    EVAL_STEPS = 1000
    LOGGING_STEPS = 25
    
    # Hardware
    FP16 = True
    USE_CACHE = False