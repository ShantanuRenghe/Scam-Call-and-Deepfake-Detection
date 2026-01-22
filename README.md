# 🛡️ Scam Call and Deepfake Detection System
### Overview
This project implements a comprehensive end-to-end pipeline designed to detect malicious voice activity. It integrates three cutting-edge AI components to tackle different aspects of voice fraud:
1. ASR (Automatic Speech Recognition): A fine-tuned Whisper model optimized for Indian-accented English to accurately transcribe calls.
2. Deepfake Detection: A computer vision-based classifier that analyzes audio spectrograms to identify AI-generated speech.
3. RAG (Retrieval-Augmented Generation): A forensic psychological analyzer that uses Large Language Models (LLM) and a vector database of academic research to detect scam intents in transcripts.

### Features
* Accent-Aware Transcription: Customized OpenAI Whisper model fine-tuned on the Svarah dataset for robust handling of Indian accents.
* Visual Audio Forensics: Converts audio to Mel-spectrograms and uses an EfficientNetB0 Convolutional Neural Network (CNN) to detect deepfake artifacts.
* Psychological Intent Analysis: Uses a RAG pipeline (LangChain + ChromaDB) to cross-reference call transcripts with forensic psychology papers, identifying coercion, urgency, and other scam tactics.
* Unified Pipeline: A single orchestrator script connects the audio input to the final forensic verdict.

### Project Structure 
| Module | Description | Path |
| :--- | :--- | :--- |
| ASR | Speech-to-Text fine-tuning and inference. | /ASR |
| Deepfake | Audio preprocessing and fake audio classification. | /Deepfake |
| RAG | Vector database ingestion and LLM-based analysis. | /RAG |
| Orchestrator | Main script connecting ASR and RAG. | run_audio_rag.py

## Quick Start
### Prerequisites
* Python 3.8+
* FFmpeg (sudo apt install ffmpeg)
* GPU recommended for training and fast inference.

### Installation
Clone the repository:
```
git clone https://github.com/yourusername/scam-detection-pipeline.git
cd scam-detection-pipeline
```
Install dependencies: (It is recommended to use a virtual environment)
```
pip install -r ASR/requirements.txt
pip install -r Deepfake/requirements.txt
pip install -r RAG/requirements.txt 
```
Environment Setup: Create a .env file in the RAG/ directory containing your Groq API key:
```
snippetGROQ_API_KEY=your_groq_api_key_here
```

### Usage
Run the Full Pipeline:To process an audio file through the ASR and RAG systems:
```
python run_audio_rag.py path/to/audio/sample.wav
```
This will:
* Transcribe the audio using the local ASR model.
* Feed the transcript into the RAG analyzer.
* Output a verdict (SCAM/GENUINE) with psychological reasoning.
