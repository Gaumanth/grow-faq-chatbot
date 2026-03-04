"""
Custom RAG Engine (no LangChain)
Uses Google Gemini for embeddings + generation, FAISS for vector search.
"""

import os
import re
import json
import time
import numpy as np
import faiss
import google.generativeai as genai

SCRAPED_DIR = "scraped_data"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"
EMBED_MODEL = "models/gemini-embedding-001"
LLM_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
EMBED_BATCH_SIZE = 10

# ── PII Detection ────────────────────────────────────────────────

PII_PATTERNS = {
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "Aadhaar": r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b",
    "Email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}",
    "Phone": r"(?:\+91[\s\-]?)?[6-9]\d{9}\b",
}


def contains_pii(text: str) -> bool:
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True
    return False


# ── Opinion / Advice Detection ───────────────────────────────────

OPINION_KEYWORDS = [
    "should i buy", "should i sell", "should i invest",
    "is it good", "is it bad", "recommend", "best fund",
    "worst fund", "which is better", "what should i do",
    "give advice", "your opinion", "suggest me",
    "which fund should", "portfolio allocation",
    "will it go up", "will it go down", "predict",
    "guaranteed return", "safe to invest",
]


def is_opinion_request(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in OPINION_KEYWORDS)


# ── Text Chunking ────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            break_point = text.rfind(". ", start, end)
            if break_point > start + chunk_size // 2:
                end = break_point + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 30]


# ── Document Loading ─────────────────────────────────────────────

def load_scraped_documents(data_dir: str = SCRAPED_DIR) -> list[dict]:
    """Load all .txt files from scraped_data/ and return list of {text, source, title}."""
    documents = []
    if not os.path.isdir(data_dir):
        return documents

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        source_match = re.search(r"Source:\s*(https?://\S+)", content)
        title_match = re.search(r"Title:\s*(.+)", content)

        documents.append({
            "text": content,
            "source": source_match.group(1) if source_match else "Unknown",
            "title": title_match.group(1).strip() if title_match else filename,
        })

    return documents


# ── Embedding ────────────────────────────────────────────────────

def _embed_with_retry(batch: list[str], task_type: str, max_retries: int = 5) -> list:
    """Call embed_content with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 10 * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after max retries due to rate limits.")


def get_embeddings(texts: list[str], task_type: str = "retrieval_document") -> np.ndarray:
    """Get embeddings from Gemini API in batches with rate-limit handling."""
    all_embeddings = []
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_num = i // EMBED_BATCH_SIZE + 1
        batch = texts[i : i + EMBED_BATCH_SIZE]
        print(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        embeddings = _embed_with_retry(batch, task_type)
        all_embeddings.extend(embeddings)

        if batch_num < total_batches:
            time.sleep(1.5)

    return np.array(all_embeddings, dtype="float32")


# ── Index Building & Loading ─────────────────────────────────────

def build_index(api_key: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Build FAISS index from scraped documents. Returns (index, chunk_metadata)."""
    genai.configure(api_key=api_key)

    documents = load_scraped_documents()
    if not documents:
        raise FileNotFoundError(
            f"No documents found in '{SCRAPED_DIR}/'. Run scraper.py first."
        )

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "title": doc["title"],
            })

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

    texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(texts, task_type="retrieval_document")

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"Index saved: {INDEX_FILE} ({index.ntotal} vectors, dim={dimension})")
    return index, all_chunks


def load_index(api_key: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load existing FAISS index and chunk metadata from disk."""
    genai.configure(api_key=api_key)

    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def get_or_build_index(api_key: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load index if it exists, otherwise build from scraped data."""
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        return load_index(api_key)
    return build_index(api_key)


# ── Search ───────────────────────────────────────────────────────

def search(query: str, index: faiss.IndexFlatIP, chunks: list[dict], k: int = 4) -> list[dict]:
    """Return top-k most relevant chunks for a query."""
    query_embedding = get_embeddings([query], task_type="retrieval_query")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({**chunks[idx], "score": float(score)})
    return results


# ── Answer Generation ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a facts-only mutual fund FAQ assistant for the Groww platform, focused on HDFC Mutual Fund schemes.

STRICT RULES:
1. Answer ONLY using the provided context. Never make up facts.
2. Keep answers to 3 sentences or fewer.
3. Include one source URL from the context as a citation.
4. End every answer with: "Last updated from sources: March 2025"
5. If the context does not contain the answer, say: "I don't have this information in my current knowledge base. Please check the official HDFC Mutual Fund website at https://www.hdfcfund.com for the latest details."
6. NEVER give investment advice, opinions, buy/sell recommendations, or performance comparisons.
7. NEVER compute or compare returns. If asked, link to the official factsheet."""


def generate_answer(question: str, retrieved_chunks: list[dict], api_key: str) -> str:
    """Generate an answer using Gemini with retrieved context."""
    genai.configure(api_key=api_key)

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

User Question: {question}

Answer:"""

    last_error = None
    for model_name in LLM_MODELS:
        model = genai.GenerativeModel(model_name)
        for attempt in range(3):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=300,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = e
                err = str(e)
                if "404" in err:
                    print(f"  Model {model_name} not found, trying next...")
                    break
                if "429" in err or "quota" in err.lower():
                    wait = 15 * (2 ** attempt)
                    print(f"  {model_name} rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    return (
        "I'm temporarily unable to generate an answer due to API rate limits. "
        "Please wait a minute and try again. "
        f"(Last error: {last_error})"
    )


# ── Main Query Pipeline ──────────────────────────────────────────

PII_RESPONSE = (
    "I cannot process queries containing personal information "
    "(PAN, Aadhaar, account numbers, email, or phone numbers). "
    "Please remove any personal details and try again."
)

OPINION_RESPONSE = (
    "I'm a facts-only assistant and cannot provide investment advice, "
    "opinions, or buy/sell recommendations. For guidance, please consult "
    "a SEBI-registered financial advisor.\n\n"
    "Learn more about mutual fund basics: "
    "https://www.amfiindia.com/investor-corner/knowledge-center/what-are-mutual-funds-new.html"
)


def query_rag(
    question: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    api_key: str,
) -> dict:
    """
    Full RAG pipeline: PII check -> opinion check -> search -> generate.
    Returns {"answer": str, "sources": list[str]}
    """
    if contains_pii(question):
        return {"answer": PII_RESPONSE, "sources": []}

    if is_opinion_request(question):
        return {"answer": OPINION_RESPONSE, "sources": []}

    retrieved = search(question, index, chunks, k=4)
    source_urls = list(dict.fromkeys(r["source"] for r in retrieved))

    answer = generate_answer(question, retrieved, api_key)

    return {"answer": answer, "sources": source_urls[:3]}
