"""
Embeddings + Simple Vector Store
--------------------------------
"""

from openai import OpenAI
import json
import numpy as np

client = OpenAI()

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

def load_chunks():

    from pipeline.data_ingestion import run_pipeline
    return run_pipeline()

# ---------------------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------------------

def create_embeddings(chunks):

    embedded_data = []

    for i, chunk in enumerate(chunks):

        print(f"Embedding {i+1}/{len(chunks)}")

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk["text"]
        )

        embedded_data.append({
            "text": chunk["text"],
            "title": chunk["title"],
            "embedding": response.data[0].embedding
        })

    return embedded_data

# ---------------------------------------------------------------------
# SAVE EMBEDDINGS
# ---------------------------------------------------------------------

def save_embeddings(data, filename="embeddings.json"):

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------

def search(query, embedded_data, top_k=3):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    scores = []

    for item in embedded_data:

        score = cosine_similarity(query_embedding, item["embedding"])
        scores.append((score, item))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scores[:top_k]]
