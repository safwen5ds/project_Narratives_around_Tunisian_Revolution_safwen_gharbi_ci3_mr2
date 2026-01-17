# Narratives Around the Tunisian Revolution (RAG)

Streamlit + FastAPI app for searching and answering questions over a multilingual archive
about the Tunisian Revolution. It ingests PDFs/HTML/TXT, builds a hybrid FAISS + BM25
index, then answers in Arabic, French, or English with citations and evaluation hooks.

## Features
- Multilingual RAG with hybrid retrieval (FAISS semantic + BM25 lexical)
- OCR pipeline for scanned PDFs (OCRmyPDF + Tesseract)
- Streamlit UI with provider switch (Ollama local or Groq hosted)
- FastAPI endpoint for programmatic access
- Domain gating and low-confidence handling

## Repository Layout
- `streamlit_app.py`: Streamlit UI
- `src/ingest.py`: Ingest documents and build the index
- `src/ask.py`: CLI question answering
- `src/ask_api.py`: FastAPI server
- `data/raw/`: Source documents (PDF/HTML/TXT)
- `data/index/`: Generated index artifacts
- `data/ontology.json`: Optional ontology for query expansion
- `logs/app.log`: Runtime logs

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add documents to `data/raw/`.
4. Build the index:
   ```bash
   python src/ingest.py
   ```
5. Run the UI:
   ```bash
   streamlit run streamlit_app.py
   ```

## Configuration
Use a `.env` file (do not commit secrets). Common settings:

```bash
RAG_LLM_PROVIDER=ollama            # ollama or groq
RAG_LLM_MODEL=gemma3               # e.g., gemma3 or qwen/qwen3-32b
RAG_OLLAMA_URL=http://localhost:11434
GROQ_API_KEY=your_key_here
RAG_ALLOW_FOREIGN_SERVICES=0       # set 1 for Groq
RAG_LLM_TIMEOUT_S=60
RAG_DOMAIN_SCORE_THRESHOLD=0.35    # optional override
```

Notes:
- The app refuses foreign providers unless `RAG_ALLOW_FOREIGN_SERVICES=1`.
- If you use Groq from the UI, the app sets `RAG_ALLOW_FOREIGN_SERVICES=1` automatically.

## Running the API
Start the FastAPI server with:

```bash
uvicorn src.ask_api:app --reload
```

Endpoints:
- `GET /health`: index and model status
- `POST /ask`: run a question (body: `{ "question": "..." }`)
- `POST /reload`: clear cached resources

## CLI Usage
```bash
python src/ask.py "What triggered the first protests in 2010?"
```

## OCR Requirements
For scanned PDFs, install these system tools and ensure they are on PATH:
- `ocrmypdf`
- `tesseract` with Arabic, French, and English language packs
- `ghostscript` (gs or gswin64c on Windows)
- `qpdf`

## Troubleshooting
- "Index not ingested": run `python src/ingest.py` and verify `data/index/` files exist.
- "Embedding dim mismatch": rebuild the index using the same embedding model.

