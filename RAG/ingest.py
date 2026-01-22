import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "data"
DB_PATH = "vectorstore"

def create_knowledge_base():
    print("Loading PDF papers...")
    # 1. Load PDF Data
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    # 2. Split Text into Chunks
    # We use a chunk size of 1000 characters with some overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} text chunks.")

    # 3. Create Embeddings
    # We use a free, high-quality local model from HuggingFace
    print("Creating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Save to Vector Store (ChromaDB)
    print("Saving to VectorStore...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"Success! Knowledge base saved to {DB_PATH}")

if __name__ == "__main__":
    create_knowledge_base()