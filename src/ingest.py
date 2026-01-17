import os, glob, json, re, pickle, subprocess, shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup, FeatureNotFound
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
OCR_CACHE_DIR = PROJECT_ROOT / "data" / "ocr_cache"
LOG_DIR = PROJECT_ROOT / "logs"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OCR_LANGS = "ara+fra+eng"
OCR_FORCE_FLAGS = os.getenv("OCR_FORCE_FLAGS", "").split()
OCR_JOBS = max(2, (os.cpu_count() or 8) // 2)
FORCE_OCR = False

TEXT_MIN_CHARS_PER_PAGE = 40
TEXT_MIN_RATIO_OK = 0.60
TEXT_SAMPLE_PAGES = 20

CHUNK_WORDS = 450
CHUNK_OVERLAP = 80
MIN_CHUNK_LENGTH = 50
MIN_TEXT_LENGTH = 300

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE = 32

ALPHA_FAISS = 0.6
TOP_K_FAISS = 50
TOP_K_BM25 = 50
TOP_K_FINAL = 10

def is_e5_model(model_name: str) -> bool:
    return "e5" in (model_name or "").lower()

def format_query(text: str, model_name: str) -> str:
    return f"query: {text}" if is_e5_model(model_name) else text

def format_passage(text: str, model_name: str) -> str:
    return f"passage: {text}" if is_e5_model(model_name) else text

def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG_DIR / "app.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def clean_text_regex(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\u02da\u02db\u02dc\u02dd\u02c6]", "", text)
    text = "".join(c for c in text if not (ord(c) < 32 and c not in "\n\t\r"))
    return text.strip()

def chunk_text(text: str, chunk_words: int = 450, overlap: int = 80) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if words else []
    chunks = []
    step = max(1, chunk_words - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_words])
        if len(chunk.split()) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        i += step
    return chunks

def chunk_text_by_sentences(text: str, target_words: int = 450) -> List[str]:
    sentences = re.split(r"(?<=[.!?\u061f\u06d4])\s+", text)
    if not sentences or len(sentences) == 1:
        return chunk_text(text, chunk_words=target_words)
    chunks = []
    current_chunk = []
    current_words = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words_in_sentence = len(sentence.split())
        if current_words + words_in_sentence <= target_words:
            current_chunk.append(sentence)
            current_words += words_in_sentence
        else:
            if current_chunk:
                chunk_str = " ".join(current_chunk)
                if len(chunk_str.split()) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk_str)
            current_chunk = [sentence]
            current_words = words_in_sentence
    if current_chunk:
        chunk_str = " ".join(current_chunk)
        if len(chunk_str.split()) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk_str)
    return chunks

def tokenize_for_bm25(text: str) -> List[str]:
    text_normalized = text.lower()
    tokens = re.findall(r"\w+", text_normalized, re.UNICODE)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def read_html(path: Path) -> str:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[WARN] Error reading HTML {path.name}: {e}")
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "meta", "link"]):
        tag.extract()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n[ \t]*\n", "\n\n", text)
    return clean_text_regex(text)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log(f"[WARN] Error reading text {path.name}: {e}")
        return ""

def _pick_sample_pages(n_pages: int, k: int) -> List[int]:
    if n_pages <= 0:
        return []
    if k >= n_pages:
        return list(range(n_pages))
    idx = set()
    idx.add(0)
    idx.add(n_pages - 1)
    if n_pages > 1:
        idx.add(n_pages // 2)
    step = max(1, n_pages // k)
    for i in range(0, n_pages, step):
        idx.add(i)
        if len(idx) >= k:
            break
    return sorted(idx)[:k]

def pdf_has_text_layer(pdf_path: Path) -> bool:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log(f"[WARN] Cannot open PDF for text-check: {pdf_path.name} ({e})")
        return False
    n = doc.page_count
    if n == 0:
        doc.close()
        return False
    pages_to_check = _pick_sample_pages(n, TEXT_SAMPLE_PAGES)
    ok = 0
    for pno in pages_to_check:
        try:
            page = doc.load_page(pno)
            txt = page.get_text("text") or ""
            txt = clean_text_regex(txt)
            if len(txt) >= TEXT_MIN_CHARS_PER_PAGE:
                ok += 1
        except Exception:
            pass
    doc.close()
    ratio = ok / max(1, len(pages_to_check))
    return ratio >= TEXT_MIN_RATIO_OK

def require_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"'{name}' not found in PATH. Install it first.")

def require_ghostscript():
    if shutil.which("gs") is None and shutil.which("gswin64c") is None:
        raise RuntimeError("'gs' or 'gswin64c' not found in PATH. Install Ghostscript.")

def ocrmypdf_run(input_pdf: Path, output_pdf: Path):
    require_binary("ocrmypdf")
    require_binary("tesseract")
    require_ghostscript()
    require_binary("qpdf")
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--deskew",
        "--rotate-pages",
        "--optimize", "0",
        "--jobs", str(OCR_JOBS),
        "-l", OCR_LANGS,
        *OCR_FORCE_FLAGS,
        str(input_pdf),
        str(output_pdf),
    ]
    log(f"Running OCRmyPDF: {' '.join(cmd[:5])}... (see logs for full command)")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        if result.stdout.strip():
            log(f"[OCRmyPDF stdout] {result.stdout.strip()[:500]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"OCRmyPDF timeout for {input_pdf.name} (>10 min)")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()[:1000]
        raise RuntimeError(
            f"OCRmyPDF failed for {input_pdf.name}\n"
            f"Error: {err}\n"
            f"Ensure Tesseract is installed with Arabic+French+English support"
        )

def ensure_ocr_pdf(input_pdf: Path) -> Path:
    if not FORCE_OCR:
        if pdf_has_text_layer(input_pdf):
            log(f"PDF has text layer: {input_pdf.name}")
            return input_pdf
    output_pdf = OCR_CACHE_DIR / f"{input_pdf.stem}__ocr.pdf"
    if output_pdf.exists():
        try:
            if output_pdf.stat().st_mtime >= input_pdf.stat().st_mtime:
                log(f"Using cached OCR: {output_pdf.name}")
                return output_pdf
        except Exception:
            pass
    log(f"OCRing PDF: {input_pdf.name}")
    ocrmypdf_run(input_pdf, output_pdf)
    log(f"OCR complete: {output_pdf.name}")
    return output_pdf

def extract_pdf_pages_text(pdf_path: Path) -> List[Tuple[int, str]]:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        log(f"[ERROR] Cannot open PDF: {pdf_path.name} ({e})")
        return []
    pages = []
    for i in range(doc.page_count):
        try:
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            txt = clean_text_regex(txt)
            if txt and len(txt) >= 50:
                pages.append((i + 1, txt))
        except Exception as e:
            log(f"[WARN] Error extracting page {i+1}: {e}")
    doc.close()
    return pages

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score < 1e-6:
        return np.ones_like(scores) * 0.5
    return (scores - min_score) / (max_score - min_score)

def hybrid_search(
    query: str,
    embedder: SentenceTransformer,
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks: List[str],
    alpha: float = 0.6,
    k: int = 10,
) -> List[Tuple[int, float]]:
    query_emb = embedder.encode([format_query(query, EMBEDDING_MODEL)], normalize_embeddings=True)
    query_emb = np.asarray(query_emb, dtype=np.float32)

    distances, indices = faiss_index.search(query_emb, min(TOP_K_FAISS, len(chunks)))
    faiss_scores = distances[0]
    faiss_scores = normalize_scores(faiss_scores)

    query_tokens = tokenize_for_bm25(query)
    bm25_scores_all = np.array(bm25.get_scores(query_tokens))
    bm25_scores_all = normalize_scores(bm25_scores_all)

    top_bm25_indices = np.argsort(-bm25_scores_all)[:TOP_K_BM25]

    combined_scores = {}

    for idx, score in zip(indices[0], faiss_scores):
        if idx < 0:
            continue
        combined_scores[idx] = alpha * score

    for idx in top_bm25_indices:
        score = (1 - alpha) * bm25_scores_all[idx]
        combined_scores[idx] = combined_scores.get(idx, 0) + score

    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    return sorted_results

def process_file(file_path: Path) -> Tuple[List[str], List[Dict]]:
    chunks = []
    meta = []
    try:
        if file_path.suffix.lower() in [".html", ".htm"]:
            log(f"Reading HTML: {file_path.name}")
            text = read_html(file_path)
            text = clean_text_regex(text)
            if len(text) < MIN_TEXT_LENGTH:
                log(f"Skipping (too short): {file_path.name}")
                return chunks, meta
            file_chunks = chunk_text_by_sentences(text, target_words=CHUNK_WORDS)
            for j, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                meta.append({
                    "source_file": file_path.name,
                    "page": None,
                    "chunk_id": j,
                    "length": len(chunk.split())
                })
        elif file_path.suffix.lower() == ".pdf":
            log(f"Processing PDF: {file_path.name}")
            pdf_ready = ensure_ocr_pdf(file_path)
            pages = extract_pdf_pages_text(pdf_ready)
            if not pages:
                log(f"No text extracted: {file_path.name}")
                return chunks, meta
            log(f"  Extracted {len(pages)} pages")
            for page_no, page_text in pages:
                page_text = clean_text_regex(page_text)
                if len(page_text) < 100:
                    continue
                page_chunks = chunk_text_by_sentences(page_text, target_words=CHUNK_WORDS)
                for j, chunk in enumerate(page_chunks):
                    chunks.append(chunk)
                    meta.append({
                        "source_file": file_path.name,
                        "page": page_no,
                        "chunk_id": j,
                        "length": len(chunk.split())
                    })
        else:
            log(f"Reading TXT: {file_path.name}")
            text = read_text(file_path)
            text = clean_text_regex(text)
            if len(text) < MIN_TEXT_LENGTH:
                log(f"Skipping (too short): {file_path.name}")
                return chunks, meta
            file_chunks = chunk_text_by_sentences(text, target_words=CHUNK_WORDS)
            for j, chunk in enumerate(file_chunks):
                chunks.append(chunk)
                meta.append({
                    "source_file": file_path.name,
                    "page": None,
                    "chunk_id": j,
                    "length": len(chunk.split())
                })
    except Exception as e:
        log(f"[ERROR] Processing {file_path.name}: {e}")
        return [], []
    return chunks, meta

def main():
    log("=" * 70)
    log("START: Optimized Multilingual Document Ingestion")
    log("=" * 70)

    files = [Path(p) for p in glob.glob(str(RAW_DIR / "*"))]
    files = [f for f in files if f.suffix.lower() in [".pdf", ".txt", ".html", ".htm"]]

    if not files:
        log("No files found in data/raw")
        return

    log(f"Found {len(files)} files to process")

    all_chunks = []
    all_meta = []

    for i, file_path in enumerate(files, 1):
        log(f"\n[{i}/{len(files)}]")
        chunks, meta = process_file(file_path)
        all_chunks.extend(chunks)
        all_meta.extend(meta)

    log(f"\n{'='*70}")
    log(f"Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        log("No chunks generated. Exiting.")
        return

    log(f"\n{'='*70}")
    log(f"Loading embedding model: {EMBEDDING_MODEL}")

    device = "cuda" if (os.environ.get("USE_CUDA") and torch.cuda.is_available()) else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    log(f"Computing embeddings for {len(all_chunks)} chunks...")
    embeddings = embedder.encode(
        [format_passage(chunk, EMBEDDING_MODEL) for chunk in all_chunks],
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    log(f"Embeddings shape: {embeddings.shape}")

    log(f"\n{'='*70}")
    log("Building FAISS index...")

    embedding_dim = embeddings.shape[1]

    if len(all_chunks) > 10000:
        nlist = max(4, int(np.sqrt(len(all_chunks))))
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatIP(embedding_dim)

    index.add(embeddings)

    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = min(32, index.nlist)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    log(f"FAISS index saved ({len(all_chunks)} vectors)")

    log(f"\n{'='*70}")
    log("Building BM25 index...")

    tokenized_chunks = [tokenize_for_bm25(chunk) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    log("BM25 index saved")

    log(f"\n{'='*70}")
    log("Saving chunks and metadata...")

    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for meta, chunk in zip(all_meta, all_chunks):
            record = {**meta, "text": chunk}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        config = {
            "embedding_model": EMBEDDING_MODEL,
            "num_chunks": len(all_chunks),
            "chunking": {
                "method": "sentence-aware",
                "target_words": CHUNK_WORDS,
                "overlap": CHUNK_OVERLAP,
                "min_length": MIN_CHUNK_LENGTH,
            },
            "bm25": {
                "tokenizer": "unicode_word",
                "min_token_length": 2,
            },
            "hybrid_search": {
                "alpha": ALPHA_FAISS,
                "top_k_faiss": TOP_K_FAISS,
                "top_k_bm25": TOP_K_BM25,
                "top_k_final": TOP_K_FINAL,
            },
            "pdf_processing": {
                "method": "ocrmypdf+tesseract",
                "languages": OCR_LANGS,
                "jobs": OCR_JOBS,
                "force_ocr": FORCE_OCR,
            },
            "statistics": {
                "total_files": len(files),
                "total_chunks": len(all_chunks),
                "avg_chunk_words": int(np.mean([m["length"] for m in all_meta])) if all_meta else 0,
                "embedding_dim": embedding_dim,
            }
        }
        json.dump(config, f, ensure_ascii=False, indent=2)

    log("Chunks saved to chunks.jsonl")
    log("Config saved to meta.json")

    log(f"\n{'='*70}")
    log("INGEST COMPLETE")
    log(f"{'='*70}")

if __name__ == "__main__":
    main()
