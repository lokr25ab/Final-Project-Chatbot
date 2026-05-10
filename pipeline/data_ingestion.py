"""
Data Ingestion + Preprocessing Pipeline
--------------------------------------
"""
import requests
import time
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_URL = "https://vidensbasen.anklagemyndigheden.dk"
PORTAL_ID = "6e302527-f0b3-4a5e-889a-668aa67e5491"
HIVE_ID = "6dfa19d8-18cc-47d6-b4c4-3bd07bc15ec0"

SEARCH_URL = f"{BASE_URL}/api/v1/portals/{PORTAL_ID}/search"
HTML_URL = f"{BASE_URL}/api/Portals({PORTAL_ID})/Documents/LocalPrimaryVariant/{HIVE_ID}"
HEADERS = {"Content-Type": "application/json"}
MAX_DOCS = 50

# ---------------------------------------------------------------------
# STEP 1: SEARCH DOCUMENTS
# ---------------------------------------------------------------------
def search_documents(query: str = None):
    body = {
        "criteria": [{"data": "Effective", "groupOperator": 0, "parameterKey": "status"}],
        "fieldSetName": "ListFields", "skip": 0, "take": MAX_DOCS,
        "sortOrder": {"descending": True, "fieldName": "Dato"},
        "resultViewName": "Liste", "page": 1
    }
    if query:
        body["criteria"].append({"data": query, "groupOperator": 1, "parameterKey": "terms"})
    response = requests.post(SEARCH_URL, json=body, headers=HEADERS)
    response.raise_for_status()
    return response.json()["Results"]

# ---------------------------------------------------------------------
# STEP 2: FETCH HTML CONTENT
# ---------------------------------------------------------------------

def get_document_html(full_name: str) -> str:
    response = requests.get(f"{HTML_URL}/{full_name}")
    response.raise_for_status()
    return response.text

# ---------------------------------------------------------------------
# STEP 3: CLEAN HTML → TEXT
# ---------------------------------------------------------------------
def clean_html(html: str) -> str:
    """Use BeautifulSoup to extract clean text from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

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
def chunk_text(text: str):
    """Use LangChain for context-aware chunking with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


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
    print("🔍 Fetching documents...")
    results = search_documents()
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
                all_chunks.append({"title": title, "text": chunk, "id": full_name})
        except Exception as e:
            print(f"⚠️ Error: {e}")
        time.sleep(0.2)

    print(f"\n✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = run_pipeline()