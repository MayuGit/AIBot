import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import chromadb # Import the core Chroma library

# --- Configuration (Must match Doc_QA.py) ---
OLLAMA_BASE_URL = 'http://host.docker.internal:11434'
EMBEDDING_MODEL = "nomic-embed-text" 
VECTOR_DB_DIR = "./chroma_db"
COLLECTION_NAME = "langchain" # This is Chroma's default collection name when created via LangChain

def inspect_chroma_collection():
    """Accesses the persistent ChromaDB collection and prints its contents."""

    # 1. Initialize the same embedding function used for indexing
    #    (Chroma needs this to recognize the collection structure)
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # 2. Re-instantiate the Chroma store using the persistent directory
    #    We use the 'chromadb.PersistentClient' to avoid dependency on the LangChain wrapper.
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    # 3. Get the collection object
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        print("Ensure the chroma_db folder exists and contains indexed data.")
        return

    print(f"✅ Successfully loaded collection: '{COLLECTION_NAME}'")
    print(f"Total indexed chunks: {collection.count()}")
    print("-" * 50)
    
    # 4. Use the .get() method to retrieve the documents and metadata
    #    By not providing an ID or filter, it retrieves all documents (up to the limit).
    #    We set limit to 1000 to see all of your 143 chunks.
    indexed_data = collection.get(
        include=['metadatas', 'documents', 'embeddings'],
        limit=1000 # Retrieve up to 1000 chunks
    )
    
    # 5. Print the indexed content
    for i in range(len(indexed_data['ids'])):
        # The 'documents' field holds the text chunk
        chunk_text = indexed_data['documents'][i]
        
        # The 'metadatas' field holds the source file info (page number, source path)
        metadata = indexed_data['metadatas'][i]
        
        print(f"--- CHUNK {i+1} ---")
        print(f"Source: {metadata.get('source', 'N/A')} (Page: {metadata.get('page', 'N/A')})")
        print("Content Snippet:")
        # Print a short snippet of the chunk content
        print(f"> {chunk_text[:300].replace('\n', ' ')}...\n")


if __name__ == "__main__":
    inspect_chroma_collection()