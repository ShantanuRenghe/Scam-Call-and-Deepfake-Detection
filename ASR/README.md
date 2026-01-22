Indian Accent ASR Fine-tuning (Whisper)

This project fine-tunes OpenAI's Whisper Small model on the Svarah dataset to improve Automatic Speech Recognition (ASR) for Indian-accented English.

Project Structure

src/train.py: Main training script using Hugging Face Trainer.

src/config.py: Hyperparameters (Batch size, learning rate, paths).

inference.py: Script to transcribe audio files using the fine-tuned model.

Installation

Clone the repository:

git clone <YOUR_REPO_URL>
cd svarah_whisper_finetune


Create a Virtual Environment:

python3 -m venv venv
source venv/bin/activate


Install Dependencies:

# Install system-level ffmpeg first (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg

# Install Python libraries
pip install -r requirements.txt


Usage

1. Training

To start fine-tuning the model. This handles data download, preprocessing, and training automatically.

python src/train.py


Training checkpoints are saved to ./checkpoints. Logs are saved to ./logs.

2. Inference (Testing)

To transcribe an audio file (supports .wav, .mp3, .m4a, etc.):

python inference.py path/to/your_audio.m4a


Configuration

You can adjust batch size, learning rate, or toggle Gradient Checkpointing in src/config.py.


### Step 3: Initialize and Commit (Terminal)

Open your terminal inside `svarah_whisper_finetune` and run:

1.  **Initialize Git:**
    ```bash
    git init
    ```

2.  **Add files (Staging):**
    This prepares the files for saving. Because of `.gitignore`, it will automatically skip the massive folders.
    ```bash
    git add .
    ```

3.  **Commit (Save):**
    ```bash
    git commit -m "Initial commit: Whisper fine-tuning pipeline"
    ```

4.  **Rename branch to main (Standard practice):**
    ```bash
    git branch -M main
    ```

### Step 4: Push to GitHub

1.  Go to **[github.com/new](https://github.com/new)**.
2.  **Repository name:** `whisper-indian-accent-finetune` (or whatever you like).
3.  **Public/Private:** Choose Public.
4.  **Important:** Do **NOT** check "Add a README" or "Add .gitignore" (we already made them).
5.  Click **Create repository**.

GitHub will show you a page with commands. Look for the section **"…or push an existing repository from the command line"**.

Copy those commands and run them in your terminal. They will look like this:

```bash
git remote add origin https://github.com/YOUR_USERNAME/whisper-indian-accent-finetune.git
git push -u origin main
