# Final-Project-Chatbot
Final project in AI & Machine Learning at CBS

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
