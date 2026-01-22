import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Configuration
DB_PATH = "vectorstore"
# GROQ_MODEL = "llama-3.1-8b-instant" # Fast and free-tier friendly
GROQ_MODEL = "llama-3.3-70b-versatile" # better for psychological analysis

def load_chain():
    """Initializes the RAG chain components"""
    
    # 1. Load the Vector Database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 2. Initialize Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0, # Low temperature for factual/analytical responses
        max_retries=2,
    )

    return vector_db, llm

def analyze_transcript(transcript_text, vector_db, llm):
    """
    Analyzes a transcript by retrieving psychological concepts and asking the LLM.
    """
    
    # 1. Retrieve relevant research
    # We ask the DB for concepts related to the transcript's content
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(transcript_text)
    
    # Combine the content of the retrieved papers
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Define the Forensic Psychologist Prompt
    PROMPT_TEMPLATE = """
    You are an expert Forensic Psychologist and Fraud Detection Analyst.
    
    Your task is to analyze the following call transcript to determine if it is a SCAM or GENUINE.
    
    You must base your analysis on the provided Context (Academic Research on Deception).
    
    ### CONTEXT (Research Findings):
    {context}
    
    ### TRANSCRIPT TO ANALYZE:
    {question}
    
    ### INSTRUCTIONS:
    1. Identify specific linguistic cues of deception mentioned in the research (e.g., lack of 'I' pronouns, urgency, authority bias).
    2. Identify specific persuasion tactics (e.g., scarcity principle, visceral influence).
    3. Compare the transcript against these academic findings.
    
    ### OUTPUT FORMAT:
    **Classification:** [SCAM / GENUINE]
    **Confidence Score:** [0-100%]
    
    **Psychological Analysis:**
    * **Tactic 1:** [Explain tactic found in transcript using the research context]
    * **Tactic 2:** [Explain tactic found in transcript using the research context]
    
    **Verdict Reasoning:**
    [Brief summary of why this decision was made]
    """

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # 3. Generate Response
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": transcript_text})
    
    return response.content

if __name__ == "__main__":
    # --- SETUP ---
    print("Initializing Forensic Model...")
    db, llm = load_chain()
    
    # --- TEST CASES ---
    scam_text = """
Hello. Hi. Hi, this is Amy from the Compliance Department. I record show your business has outstanding documentation issues. What kind of issues? New government regulations require updated verification for all businesses. If not completed today, penalties will apply. I haven't heard of this before. It's urgent, ma'am. You need to submit your business registration number and financial statements to avoid fines. Can I confirm this with the government website? There's actually no time. Just email your documents now and I'll process the update for you. Alright, I'll send them. Yep. Yeah, I just received them. Thank you so much for your cooperation, ma'am. I'll take it from here and I'll update you soon about it. Have a good day.
    """
    
    # --- RUN ANALYSIS ---
    print("\n" + "="*50)
    print("ANALYZING TRANSCRIPT 1...")
    print("="*50)
    result_scam = analyze_transcript(scam_text, db, llm)
    print(result_scam)
