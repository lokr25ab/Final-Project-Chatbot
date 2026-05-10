"""
RAG Chatbot 
-----------------------------------
Connects User Query -> ChromaDB -> OpenAI
"""
from openai import OpenAI
import os

# Import the search function from our new embeddings file
from pipeline.embeddings import search_db

# Ensure you have your OPENAI_API_KEY set in your environment variables
client = OpenAI()

def build_context(results):
    """Formats the retrieved chunks into a single context string"""
    context_str = ""
    for i, r in enumerate(results):
        context_str += f"--- Source {i+1}: {r['title']} ---\n{r['text']}\n\n"
    return context_str

def ask_chatbot(query: str, model="gpt-5-nano"):
    print(f"\n🔎 Searching database for: '{query}'")
    
    # 1. Retrieve relevant documents
    retrieved_chunks = search_db(query, n_results=4)
    
    if not retrieved_chunks:
        return "I couldn't find any relevant legal documents for that query."

    # 2. Build the context string
    context = build_context(retrieved_chunks)
    
    # 3. Generate answer using OpenAI
    print(f"🤖 Generating answer with {model}...\n")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du er en dansk juridisk assistent (You are a Danish legal assistant). "
                    "Besvar KUN spørgsmålet baseret på den medfølgende kontekst. "
                    "Hvis svaret ikke findes i konteksten, skal du sige 'Jeg ved det ikke baseret på de tilgængelige dokumenter.'"
                )
            },
            {
                "role": "user",
                "content": f"Kontekst:\n{context}\n\nSpørgsmål: {query}"
            }
        ]
    )

    # 1. Get the LLM's text answer
    final_answer = response.choices[0].message.content
    
    # 2. Extract unique sources and format them as URLs
    HIVE_ID = "6dfa19d8-18cc-47d6-b4c4-3bd07bc15ec0"
    unique_urls = set()
    
    for chunk in retrieved_chunks:
        # Construct the official Vidensbasen URL using the Hive ID and Document ID
        doc_id = chunk['source']
        url = f"https://vidensbasen.anklagemyndigheden.dk/h/{HIVE_ID}/{doc_id}"
        
        # We also grab the title to make the link look nice!
        title = chunk['title']
        unique_urls.add(f"- [{title}]({url})")
        
    # 3. Append the sources to the bottom of the answer
    final_answer += "\n\n---\n**Kilder (Sources):**\n" + "\n".join(unique_urls)

    return final_answer
    # return response.choices[0].message.content

# ---------------------------------------------------------------------
# MAIN SCRIPT TO RUN THE WHOLE PIPELINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: If you haven't built the database yet, uncomment the lines below:
    # from pipeline.data_ingestion import run_pipeline
    # from pipeline.embeddings import build_vector_db
    # chunks = run_pipeline()
    # build_vector_db(chunks)

    # Test the chatbot
    user_question = "Hvad er retningslinjerne for vold?" # What are the guidelines for violence?
    answer = ask_chatbot(user_question)
    
    print("\n" + "="*50)
    print("SVAR (ANSWER):")
    print("="*50)
    print(answer)