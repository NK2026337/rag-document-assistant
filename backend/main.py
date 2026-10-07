import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class Question(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if os.getenv("DISABLE_UPLOAD") == "1":
        raise HTTPException(403, "Uploads are disabled on this deployment.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = file.file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, "File is too large (10 MB max).")
    count = rag.index_document(file.filename, data)
    if count == 0:
        raise HTTPException(400, "No text found. Scanned PDFs are not supported yet.")
    return {"filename": file.filename, "chunks_indexed": count}


@app.post("/ask")
def ask(body: Question):
    return rag.answer(body.question)


# Serve the frontend at /app (keep this at the bottom)
FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")