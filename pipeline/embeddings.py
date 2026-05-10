"""
Embeddings and Vector Database
------------------------------
Handles storing and searching document chunks using ChromaDB.
"""
import chromadb
from chromadb.utils import embedding_functions

# Initialize a local persistent Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a Multilingual model that understands Danish!
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# Create or load the collection
collection = chroma_client.get_or_create_collection(
    name="legal_documents",
    embedding_function=embedding_func
)

# Build Vector DB
def build_vector_db(chunks):
    """Takes chunks from data_ingestion.py and puts them in ChromaDB"""
    print(f"🧠 Embedding {len(chunks)} chunks into ChromaDB...")
    
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"title": chunk["title"], "source": chunk["id"]} for chunk in chunks]
    ids = [f"{chunk['id']}_chunk_{i}" for i, chunk in enumerate(chunks)]

    # Add to database (Chroma automatically handles the vectorization)
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Vector database populated!")

# Search the Database
def search_db(query: str, n_results: int = 3):
    """Searches the database for the most relevant chunks"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Format results nicely
    formatted_results = []
    if results['documents'][0]:
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "text": results['documents'][0][i],
                "title": results['metadatas'][0][i]["title"],
                "source": results['metadatas'][0][i]["source"]
            })
    return formatted_results