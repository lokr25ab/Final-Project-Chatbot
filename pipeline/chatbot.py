"""
RAG Chatbot
-----------

Purpose:
- Take a user query
- Retrieve relevant chunks
- Generate answer using LLM

Final step in the pipeline.
"""

from openai import OpenAI
import json

# Import search from embeddings module
from pipeline.embeddings import search

client = OpenAI()


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
    context = "\n\n".join([r["text"] for r in results])
    return context


# ---------------------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------------------

def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a legal assistant. Answer based ONLY on the provided context."
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

        print("Generating answer...\n")
        answer = generate_answer(query, context)

        print("Answer:\n")
        print(answer)
