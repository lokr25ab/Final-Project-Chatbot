"""
Data Ingestion + Preprocessing Pipeline
--------------------------------------

Purpose:
- Fetch documents from Anklagemyndighedens API
- Extract HTML content
- Convert HTML → clean text
- Chunk text into smaller segments (for embeddings)

This is step 1+2 in the RAG pipeline.
"""

import requests
import time
import re
from typing import List, Dict

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_URL = "https://vidensbasen.anklagemyndigheden.dk"

PORTAL_ID = "6e302527-f0b3-4a5e-889a-668aa67e5491"
HIVE_ID = "6dfa19d8-18cc-47d6-b4c4-3bd07bc15ec0"

SEARCH_URL = f"{BASE_URL}/api/v1/portals/{PORTAL_ID}/search"
HTML_URL = f"{BASE_URL}/api/Portals({PORTAL_ID})/Documents/LocalPrimaryVariant/{HIVE_ID}"

HEADERS = {"Content-Type": "application/json"}

# Number of documents to fetch (start small!)
MAX_DOCS = 10

# Chunk size (characters)
CHUNK_SIZE = 500


# ---------------------------------------------------------------------
# STEP 1: SEARCH DOCUMENTS
# ---------------------------------------------------------------------

def search_documents(query: str = None) -> List[Dict]:
    """
    Calls the search endpoint to retrieve document metadata.
    """
    body = {
        "criteria": [
            {"data": "Effective", "groupOperator": 0, "parameterKey": "status"}
        ],
        "fieldSetName": "ListFields",
        "skip": 0,
        "take": MAX_DOCS,
        "sortOrder": {"descending": True, "fieldName": "Dato"},
        "resultViewName": "Liste",
        "page": 1
    }

    # Optional query
    if query:
        body["criteria"].append({
            "data": query,
            "groupOperator": 1,
            "parameterKey": "terms"
        })

    response = requests.post(SEARCH_URL, json=body, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    return data["Results"]


# ---------------------------------------------------------------------
# STEP 2: FETCH HTML CONTENT
# ---------------------------------------------------------------------

def get_document_html(full_name: str) -> str:
    """
    Fetch HTML version of a document.
    """
    url = f"{HTML_URL}/{full_name}"

    response = requests.get(url)
    response.raise_for_status()

    return response.text


# ---------------------------------------------------------------------
# STEP 3: CLEAN HTML → TEXT
# ---------------------------------------------------------------------

def clean_html(html: str) -> str:
    """
    Very simple HTML → clean text conversion.
    (Can later be replaced with markdown conversion)
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------
# STEP 4: CHUNKING
# ---------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into smaller chunks for embedding.
    """
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def run_pipeline():
    """
    Full ingestion pipeline:
    search → HTML → clean → chunk
    """
    print("🔍 Fetching documents...")

    results = search_documents(query="vold")

    all_chunks = []

    for doc in results:
        full_name = doc["FullName"]
        title = doc.get("Titel", "Unknown")

        print(f"📄 Processing: {title}")

        try:
            html = get_document_html(full_name)
            text = clean_html(html)
            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "title": title,
                    "text": chunk
                })

        except Exception as e:
            print(f"⚠️ Error processing document: {e}")

        # Be polite to API
        time.sleep(0.2)

    print(f"\n✅ Total chunks created: {len(all_chunks)}")

    return all_chunks


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    chunks = run_pipeline()

    # Print sample
    print("\n--- SAMPLE CHUNK ---\n")
    print(chunks[0]["text"][:500])
