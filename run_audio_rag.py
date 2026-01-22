import sys
from pathlib import Path

# Make local modules importable
ROOT = Path(__file__).resolve().parent
ASR_DIR = ROOT / "ASR"
RAG_DIR = ROOT / "RAG"
if str(ASR_DIR) not in sys.path:
    sys.path.append(str(ASR_DIR))
if str(RAG_DIR) not in sys.path:
    sys.path.append(str(RAG_DIR))

from inference import transcribe_file  # type: ignore
from main import load_chain, analyze_transcript  # type: ignore


def process_audio_file(audio_path: str):
    """Run ASR on the audio file and feed the transcript into the RAG analyzer."""
    transcript = transcribe_file(audio_path)
    if not transcript:
        print("No transcript produced; skipping RAG analysis.")
        return None

    print("Running RAG analysis...")
    vector_db, llm = load_chain()
    result = analyze_transcript(transcript, vector_db, llm)

    print("\n" + "=" * 20 + " RAG ANALYSIS " + "=" * 20)
    print(result)
    print("=" * 60 + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_audio_rag.py <path_to_audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    process_audio_file(audio_file)
