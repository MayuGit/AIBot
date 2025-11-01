# Doc_QA.py

import os
import pypdf

# --- CORRECTED IMPORTS FOR LATEST LANGCHAIN ---

# 1. Chain Logic (RetrievalQA is back in the main `langchain` package, though deprecated in favor of LCEL)
from langchain_classic.chains import RetrievalQA

# 2. Third-Party Integrations (Must come from `langchain_community` or specific integration packages)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# 3. Text Splitters (Moved to its own package)
from langchain_text_splitters import RecursiveCharacterTextSplitter 


# Text splitter (moved to 'langchain_text_splitters')
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# 1. Ollama Host & Model (Your running container setup)
OLLAMA_BASE_URL = 'http://host.docker.internal:11434'
# 'default' model points to your GPU-accelerated gemma3n:latest
MODEL_NAME = 'default' 

# 2. Embedding Model (Used to vectorize your document text)
# We will use 'nomic-embed-text' as it's excellent and optimized for Ollama.
EMBEDDING_MODEL = "nomic-embed-text" 
# NOTE: You MUST download this model into your Ollama container once! (See Step 3)

# 3. Path to your document
PRIVATE_DOC_PATH = "D:\\testAI.pdf" 
VECTOR_DB_DIR = "./chroma_db"

def setup_rag_pipeline(doc_path: str, vector_db_dir: str):
    """Sets up the RAG pipeline: Load -> Chunk -> Embed -> Store -> Retriever."""
    
    # 1. Load the Document
    print(f"Loading document: {os.path.basename(doc_path)}")
    loader = PyPDFLoader(doc_path)
    data = loader.load()

    # 2. Split the Document (Chunking)
    # Recommended settings for LLM context window size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(data)
    print(f"Split document into {len(all_splits)} chunks.")
    
    # 3. Define Embeddings and LLM (using your local Ollama instance)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    llm = Ollama(
        model=MODEL_NAME, 
        base_url=OLLAMA_BASE_URL
    )


    # 4. Create/Load Vector Store (Chroma)
    print(f"Creating vector store in {vector_db_dir}...")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=vector_db_dir
    )
    
    # 5. Create the Retriever Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Puts retrieved chunks directly into the prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}), # Retrieve top 4 relevant chunks
        return_source_documents=False
    )
    
    return qa_chain

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Check for document existence ---
    if not os.path.exists(PRIVATE_DOC_PATH):
        print(f"ERROR: Document not found at path: {PRIVATE_DOC_PATH}")
        print("Please update the PRIVATE_DOC_PATH variable in the script.")
        exit()

    # --- Setup ---
    try:
        # This will load and embed the document (slow the first time)
        qa_pipeline = setup_rag_pipeline(PRIVATE_DOC_PATH, VECTOR_DB_DIR)

        # --- Interactive Query Loop ---
        while True:
            question = input("\nAsk a question about your document (or type 'exit'):\n> ")
            if question.lower() == 'exit':
                print("Exiting RAG system.")
                break
            
            print("Thinking...")
            
            # --- Query the Model ---
            # The pipeline automatically fetches relevant chunks and feeds them to gemma3n
            result = qa_pipeline.invoke({"query": question})
            
            print("\n" + "="*20 + " GEMMA3N ANSWER " + "="*20)
            print(result['result'])
            print("="*56)
            
    except Exception as e:
        print(f"\n❌ An error occurred during RAG execution: {e}")
        print("Possible fixes: Ensure Ollama is running, and run Step 3 command below.")