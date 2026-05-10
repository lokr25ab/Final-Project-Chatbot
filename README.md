# ⚖️ Anklagemyndigheden Chatbot (RAG Pipeline)
# Final-Project-Chatbot
Final project in AI & Machine Learning at CBS

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on the Danish legal knowledge base, *Vidensbasen*. It leverages a local vector database for semantic search, an OpenAI Language Model for response generation, and includes a full evaluation suite to scientifically measure performance.

## ✨ Key Features
* **Automated Data Ingestion:** Connects directly to the Vidensbasen Swagger API to download and parse semantic HTML legal documents.
* **Multilingual Semantic Search:** Uses HuggingFace's `paraphrase-multilingual-MiniLM-L12-v2` to accurately embed and search Danish text locally.
* **Persistent Vector Storage:** Powered by ChromaDB for fast, local document retrieval.
* **Source Citations:** The chatbot automatically appends verifiable, clickable URLs to the original legal documents for every answer.
* **Scientific Evaluation:** Built-in test suite using the `Ragas` framework to measure output Faithfulness and Answer Relevancy, utilizing custom wrapper logic to handle strict LLM temperature constraints.
* **Interactive UI:** A modern web interface built with Streamlit.

---

## 🛠️ Installation & Setup

**1. Prerequisites**
Ensure you have Python 3.9+ installed. If you are on a Mac, you may need to install the Xcode Command Line tools (`xcode-select --install`).

**2. Install Dependencies**
Install all required packages using the provided requirements file. 
*(Note: A specific version of NumPy is required for macOS compatibility with sentence-transformers).*
```bash
python3 -m pip install -r requirements.txt

###3. Set up your OpenAI API Key
The generation and evaluation scripts require an OpenAI API key. Export it in your terminal before running any scripts:
```bash
export OPENAI_API_KEY="{sk-your-api-key-here}"
```

---

## 🚀 Usage Guide

**Step 1. Build the Vector Database (First Time Only)**
Before you can chat, you must download the documents and build the "brain".

1. Open 
```
pipeline/chatbot.py.
```

2. Scroll to the bottom and uncomment the 4 lines under 
```
if __name__ == "__main__":
```
that build the database.

3. Run the script:
```bash
python3 -m pipeline.chatbot
```
4. Wait for the success message: ```✅ Vector database populated!.```

5. **Re-comment** those 4 lines so you don't rebuild the database every time you chat.

**Step 2. Launch the Web App**

To start the chatbot user interface, run:
```bash
streamlit run app.py
```
This will open a browser window where you can ask questions in Danish (e.g., "Hvad er retningslinjerne for vold?").

**Step 3: Run the Evaluation Suite**
To scientifically evaluate the chatbot's performance (measuring Hallucinations and Relevancy), run the evaluation script:
```bash
python3 -m pipeline.evaluate
This will run a predefined test suite and output a Pandas DataFrame with the scores.
```

## 📂 Project Structure
Final-Project-Chatbot/
│
├── app.py                  # Streamlit Web User Interface
├── api_demo.py             # Sandbox API extraction tool
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── pipeline/
    ├── __init__.py         
    ├── data_ingestion.py   # API connection and LangChain chunking
    ├── embeddings.py       # ChromaDB setup and HuggingFace integration
    ├── chatbot.py          # RAG retrieval logic and OpenAI generation
    └── evaluate.py         # Ragas testing suite and custom LLM wrappers

## High-Level Architecture

### Pipeline Overview

The system follows a Retrieval-Augmented Generation (RAG) pipeline designed to process legal data and generate accurate, context-aware responses.

1. **Data ingestion** (API → raw JSON)
2. **Document normalization** (JSON → clean text documents)
3. **Chunking & metadata enrichment**
4. **Embedding & indexing**
5. **Retrieval-Augmented Generation (RAG)**
6. **Optional agentic enhancements**
7. **Evaluation & benchmarking**
8. **Application interface** (chat UI)


#### The central swagger-endpoints we need to use

1. GET /documents → data retrieval
2. Chunking → text segments
3. POST /v1/embeddings → vector representations
4. POST /vectors/upsert → storage
5. POST /vectors/query → retrieval
6. POST /v1/chat/completions → answer generation
