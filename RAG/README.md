# 🧠 RAG-Based Scam Analysis

## Overview

The Retrieval-Augmented Generation (RAG) module acts as an automated **Forensic Psychologist**. It analyzes call transcripts to determine "Scam" or "Genuine" intent. Instead of relying solely on the LLM's internal knowledge, it retrieves specific context from a curated vector database of academic papers on deception, fraud, and psychology.

## Key Components

| Component | Role | File |
| :--- | :--- | :--- |
| **Ingestion** | Loads PDF research papers, chunks text, and creates vector embeddings. | `ingest.py` |
| **Vector DB** | Local ChromaDB instance storing knowledge embeddings. | `vectorstore/` |
| **Analyzer** | Queries the DB and prompts the Groq LLM for a verdict. | `main.py` |

## Tech Stack
* **LLM:** Llama-3.3-70b (via Groq API)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** ChromaDB
* **Framework:** LangChain

## How to Use

### 1. Build Knowledge Base
Place your academic PDF papers in the `data/` folder and run:
```bash
python ingest.py
```
This will create/update the persistent vector database in vectorstore/.

### 2. Run Analysis (Standalone)
To test the analyzer with hardcoded text or specific inputs:

```Bash
python main.py
```
### 3. Integration
This module is primarily designed to be called by the root run_audio_rag.py script, which passes the ASR transcript directly to the analyze_transcript function.
