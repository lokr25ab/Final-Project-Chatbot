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

# Choose model (you HAVE access to these)
MODEL = "gpt-5-mini"   # or "gpt-5-nano"


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

def generate_answer(query, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. Answer ONLY using the provided context. "
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

        print("Generating answer...\n")
        answer = generate_answer(query, context)

        print("Answer:\n")
        print(answer)
