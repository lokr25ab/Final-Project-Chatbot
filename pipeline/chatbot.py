"""
RAG Chatbot (Multi-model comparison)
-----------------------------------

Purpose:
- Take a user query
- Retrieve relevant chunks
- Generate answers using multiple LLMs
- Compare model outputs

Final step in the pipeline.
"""

from openai import OpenAI
import json
import time

# Import search from embeddings module
from pipeline.embeddings import search

client = OpenAI()

# Models to compare
MODELS = ["gpt-5-nano", "gpt-5-mini"]


# ---------------------------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------------------------

def load_embeddings(filename="embeddings.json"):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# BUILD CONTEXT
# ---------------------------------------------------------------------

def build_context(results):
    return "\n\n".join([r["text"] for r in results])


# ---------------------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------------------

def generate_answer(query, context, model):

    response = client.chat.completions.create(
        model=model,
       
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. "
                    "Answer ONLY using the provided context. "
                    "If the answer is not in the context, say you don't know."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("Loading embeddings...")
    data = load_embeddings()

    while True:

        query = input("\nAsk a legal question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        print("Searching relevant documents...")
        results = search(query, data)

        context = build_context(results)

        # Optional: show retrieved sources
        print("\n--- SOURCES ---")

        for i, r in enumerate(results):
            print(f"\nSource {i+1}: {r['title']}")
            print(r["text"][:200])

        print("\nGenerating answers...\n")

        # Compare models
        for model in MODELS:

            start = time.time()

            answer = generate_answer(query, context, model)

            end = time.time()

            print(f"\n===== MODEL: {model} ({round(end - start, 2)}s) =====\n")

            print(answer)
