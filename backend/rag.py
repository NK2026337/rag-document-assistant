import io
import json
import os

import numpy as np
import ollama
from pypdf import PdfReader

# ---------- Settings ----------
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"      # use "llama3.2:1b" for low RAM, or "qwen2.5:7b" for better answers
MAX_PAGES = None             # testing limit; set to None to index every page
CHUNK_SIZE = 1000           # characters per chunk
CHUNK_OVERLAP = 200
TOP_K = 6                   # chunks sent to the model per question

# ---------- Storage ----------
chunks = []         # each item: {"text": ..., "source": ..., "page": ...}
embeddings = None   # numpy matrix, one row per chunk

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
EMB_FILE = os.path.join(DATA_DIR, "embeddings.npy")


def save_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    if embeddings is not None:
        np.save(EMB_FILE, embeddings)


def load_index():
    global embeddings
    if os.path.exists(CHUNKS_FILE) and os.path.exists(EMB_FILE):
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            chunks.extend(json.load(f))
        embeddings = np.load(EMB_FILE)


def remove_document(filename: str):
    """Drop old chunks of a file so re-uploading it doesn't create duplicates."""
    global embeddings
    keep = [i for i, c in enumerate(chunks) if c["source"] != filename]
    if len(keep) == len(chunks):
        return
    chunks[:] = [chunks[i] for i in keep]
    embeddings = embeddings[keep] if keep else None


load_index()  # runs at startup


# ---------- PDF reading and chunking ----------
def extract_pages(file_bytes: bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    parts, start = [], 0
    while start < len(text):
        part = text[start:start + size].strip()
        if part:
            parts.append(part)
        start += size - overlap
    return parts


# ---------- Embeddings ----------
def embed(texts, kind="document"):
    # nomic-embed-text expects these prefixes for good retrieval quality
    prefix = "search_document: " if kind == "document" else "search_query: "
    texts = [prefix + t for t in texts]
    vectors = []
    for i in range(0, len(texts), 32): 
        print(f"Embedding {min(i + 32, len(texts))} / {len(texts)}", flush=True) # small batches
        result = ollama.embed(model=EMBED_MODEL, input=texts[i:i + 32])
        vectors.extend(result["embeddings"])
    arr = np.array(vectors, dtype=float)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)  # normalize


# ---------- Indexing ----------
def index_document(filename: str, file_bytes: bytes) -> int:
    global embeddings

    pages = extract_pages(file_bytes)
    if MAX_PAGES:
        pages = pages[:MAX_PAGES]

    new_chunks = []
    for page_num, text in pages:
        for piece in split_text(text):
            new_chunks.append({"text": piece, "source": filename, "page": page_num})

    if not new_chunks:
        return 0  # likely a scanned PDF with no extractable text

    new_matrix = embed([c["text"] for c in new_chunks], kind="document")

    remove_document(filename)  # avoid duplicates if the same file is uploaded again
    embeddings = new_matrix if embeddings is None else np.vstack([embeddings, new_matrix])
    chunks.extend(new_chunks)
    save_index()
    return len(new_chunks)


# ---------- Retrieval and answering ----------
def retrieve(question: str, k: int = TOP_K):
    q = embed([question], kind="query")[0]
    scores = embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top]


def answer(question: str):
    if embeddings is None or not chunks:
        return {"answer": "Please upload a document first.", "sources": []}

    top_chunks = retrieve(question)
    context = "\n\n".join(
        f"[{n + 1}] ({c['source']}, page {c['page']})\n{c['text']}"
        for n, c in enumerate(top_chunks)
    )

    response = ollama.chat(
        model=LLM_MODEL,
        keep_alive="30m",
        options={"num_ctx": 4096, "temperature": 0.2},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about a document. "
                    "Use the numbered context passages to answer clearly and directly. "
                    "Cite passages like [1] or [2]. If the context only partly answers "
                    "the question, share what it does say. Only say the answer isn't "
                    "in the document if the context has nothing relevant."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    sources = [
        {"source": c["source"], "page": c["page"], "preview": c["text"][:200]}
        for c in top_chunks
    ]
    return {"answer": response["message"]["content"], "sources": sources}