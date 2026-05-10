"""
RAG Chatbot (Multi-model comparison)
-----------------------------------
"""

from openai import OpenAI
import json

from pipeline.embeddings import search

client = OpenAI()

# Models to compare
MODELS = [
    "gpt-5-nano",
    "gpt-5-mini"
]

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
# GENERATE ANSWERS
# ---------------------------------------------------------------------

def generate_answers(query, context):

    answers = {}

    for model in MODELS:

        print(f"Generating answer with {model}...")

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

        answers[model] = response.choices[0].message.content

    return answers
