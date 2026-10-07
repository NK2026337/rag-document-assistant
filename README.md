# RAG Document Assistant

Ask questions about a PDF and get answers based on its content, with the pages the answer came from. Everything runs on your own computer, with no API keys and no paid services.

## Features

- Upload a PDF and ask questions in a simple chat page
- Answers include clickable citations, and each one links to the page and text it came from
- Runs fully locally using [Ollama](https://ollama.com), so your documents never leave your machine
- Saves the search index to disk, so you only index a document once
- Can be shared online for free with a Cloudflare Quick Tunnel

## How it works

1. **Read:** the PDF text is extracted page by page and split into overlapping chunks.
2. **Embed:** each chunk is turned into a vector with the `nomic-embed-text` model.
3. **Retrieve:** your question is embedded the same way, and the most similar chunks are found.
4. **Answer:** those chunks are sent to a local language model (`llama3.2`), which answers using only that context and cites its sources.

## Tech stack

- **Backend:** Python, FastAPI, pypdf, NumPy
- **Models:** Ollama (`llama3.2` for answers, `nomic-embed-text` for embeddings)
- **Frontend:** plain HTML, CSS and JavaScript (no build step)

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running
- About 8 GB of RAM or more (16 GB recommended for larger models)

## Setup

```powershell
# 1. Clone the project
git clone https://github.com/NK2026337/rag-document-assistant.git
cd rag-document-assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate

# 3. Install the dependencies
pip install -r backend/requirements.txt

# 4. Download the models (one time)
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run

Make sure the Ollama app is running, then:

```powershell
cd backend
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/app/** in your browser.

1. Upload a PDF with the upload box. Large documents can take several minutes to index, so keep the tab open.
2. Ask specific questions, for example "How do I create a table?" Specific questions work best, and broad ones like "summarize everything" work poorly.
3. Click a citation number or **Sources** to see the page and passage behind the answer.

The index is saved in `backend/data/`, so after the first upload you can restart the server without indexing again.

## Project structure

```
rag-document-assistant/
├── backend/
│   ├── main.py            # FastAPI app: /upload, /ask, serves the frontend
│   ├── rag.py             # chunking, embeddings, retrieval, answering
│   └── requirements.txt
├── frontend/
│   └── index.html         # chat interface
├── .gitignore
└── README.md
```

## API

| Method | Path      | Description                                  |
|--------|-----------|----------------------------------------------|
| GET    | `/`       | Health check                                 |
| POST   | `/upload` | Upload a PDF (form field `file`) and index it |
| POST   | `/ask`    | Ask a question: `{"question": "..."}`         |
| GET    | `/app/`   | The chat page                                |

## Settings

Edit the values at the top of `backend/rag.py`:

| Setting      | Meaning                                                      |
|--------------|--------------------------------------------------------------|
| `LLM_MODEL`  | Answering model. `llama3.2:1b` is faster, `qwen2.5:7b` is better |
| `MAX_PAGES`  | Limit pages indexed while testing. `None` indexes all pages   |
| `CHUNK_SIZE` | Characters per chunk                                         |
| `TOP_K`      | How many chunks are sent to the model per question           |

## Share it for free

You can share the running app through a Cloudflare Quick Tunnel. No account is needed, but the link only works while your computer is on.

```powershell
# Terminal 1: start the backend with uploads disabled
cd backend
$env:DISABLE_UPLOAD="1"; uvicorn main:app

# Terminal 2: create the tunnel
cloudflared tunnel --url http://localhost:8000
```

Open the `trycloudflare.com` link it prints, with `/app/` added at the end. Setting `DISABLE_UPLOAD=1` stops visitors from uploading files.

## Limitations

- Only PDFs with selectable text are supported. Scanned PDFs need OCR, which is not included.
- Local models are slower than hosted APIs, and answers can take from 10 to 60 seconds.
- All uploaded documents go into one shared index.
- Vectors are stored in memory and saved to simple files, which suits a few documents rather than a large library.