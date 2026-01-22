import torch
from transformers import pipeline
import sys
import os
import subprocess

# Path to your trained model
# Adjust this if you want to use a specific checkpoint like "checkpoints/checkpoint-1000"
MODEL_PATH = "./checkpoints/final_model" 

def convert_to_wav(input_path):
    """
    Converts any audio file to 16kHz Mono WAV using ffmpeg.
    Returns the path to the converted file.
    """
    # Create a temporary filename to avoid overwriting the original
    # e.g., "recording.m4a" -> "recording_converted_temp.wav"
    filename, _ = os.path.splitext(input_path)
    output_path = f"{filename}_converted_temp.wav"
    
    print(f"Converting {input_path} to 16kHz WAV format...")
    
    # ffmpeg command:
    # -i input_path : Input file
    # -ar 16000     : Set audio rate to 16kHz (Whisper native requirement)
    # -ac 1         : Set audio channels to 1 (Mono)
    # -y            : Overwrite output if exists
    # -loglevel error : Suppress logs unless there is an error
    command = [
        "ffmpeg", "-i", input_path, 
        "-ar", "16000", 
        "-ac", "1", 
        "-y", 
        "-loglevel", "error", 
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install it with 'sudo apt install ffmpeg'")
        sys.exit(1)

def transcribe_file(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File {audio_path} does not exist.")
        return None

    # 1. Load Model
    # Check if model exists first
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Did training finish?")
        return

    print(f"Loading model from {MODEL_PATH}...")
    device = 0 if torch.cuda.is_available() else -1
    
    pipe = pipeline(
        "automatic-speech-recognition", 
        model=MODEL_PATH,
        tokenizer=MODEL_PATH, 
        chunk_length_s=30, 
        device=device
    )

    # 2. Convert Audio
    # We convert EVERYTHING to ensure stability. 
    # Even if it's already a WAV, this ensures it is 16kHz mono.
    processing_path = convert_to_wav(audio_path)

    prediction = None
    
    # 3. Run Inference
    try:
        print(f"Transcribing...")
        prediction = pipe(processing_path, batch_size=8)["text"]

        print("\n" + "="*20 + " TRANSCRIPTION " + "="*20)
        print(prediction)
        print("="*55 + "\n")
        
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        # 4. Cleanup
        # Remove the temporary file to keep folder clean
        if os.path.exists(processing_path) and processing_path != audio_path:
            print(f"Removing temporary file: {processing_path}")
            os.remove(processing_path)

    return prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file>")
    else:
        transcribe_file(sys.argv[1])