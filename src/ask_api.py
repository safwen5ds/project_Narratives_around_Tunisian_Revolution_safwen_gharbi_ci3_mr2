import json, re, pickle, os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from pathlib import Path
import unicodedata
import threading
from typing import Any, Dict

import numpy as np
import faiss
import langid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
LOG_DIR = PROJECT_ROOT / "logs"
ONTOLOGY_PATH = PROJECT_ROOT / "data" / "ontology.json"
META_PATH = INDEX_DIR / "meta.json"
INDEX_REQUIRED_FILES = [
    INDEX_DIR / "chunks.jsonl",
    INDEX_DIR / "faiss.index",
    INDEX_DIR / "bm25.pkl",
]
INDEX_OPTIONAL_FILES = [META_PATH]

DEFAULT_INDEX_CONFIG = {
    "embedding_model": "intfloat/multilingual-e5-large",
    "hybrid_search": {
        "alpha": 0.6,
        "top_k_faiss": 50,
        "top_k_bm25": 50,
        "top_k_final": 10,
    },
}

_RESOURCE_CACHE = {
    "chunks": None,
    "index": None,
    "bm25": None,
    "model": None,
    "model_name": None,
    "config": None,
    "signature": None,
}

_CACHE_LOCK = threading.Lock()

LOG_DIR.mkdir(parents=True, exist_ok=True)

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_TATWEEL_RE = re.compile(r"\u0640")
_MATCH_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", re.UNICODE)
_MATCH_SPACE_RE = re.compile(r"\s+")

def strip_accents(text: str):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

def normalize_arabic(text: str):
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _ARABIC_TATWEEL_RE.sub("", text)
    return (
        text.replace("\u0622", "\u0627")
        .replace("\u0623", "\u0627")
        .replace("\u0625", "\u0627")
        .replace("\u0649", "\u064A")
    )

def normalize_text(text: str):
    if not text:
        return ""
    text = text.lower()
    text = strip_accents(text)
    return normalize_arabic(text)

def normalize_for_match(text: str):
    text = normalize_text(text)
    if not text:
        return ""
    text = _MATCH_PUNCT_RE.sub(" ", text)
    text = _MATCH_SPACE_RE.sub(" ", text).strip()
    return text

DOMAIN_KEYWORDS = [
    "tunis", "tunisie", "revolution", "revolution tunisienne", "sidi bouzid", "bouazizi", "kasbah",
    "protest camp", "protest camps", "sit-in", "sit-ins", "sitins",
    "youth testimony", "youth testimonies", "neighborhood stories",
    "media slogans", "slogans", "protest slogans", "chants",
    "\u062a\u0648\u0646\u0633", "\u0627\u0644\u062b\u0648\u0631\u0629", "\u062b\u0648\u0631\u0629",
    "\u0633\u064a\u062f\u064a \u0628\u0648\u0632\u064a\u062f", "\u0628\u0648\u0639\u0632\u064a\u0632\u064a",
    "\u0627\u0644\u0628\u0648\u0639\u0632\u064a\u0632\u064a", "\u0627\u0644\u0642\u0635\u0628\u0629",
    "\u0627\u0639\u062a\u0635\u0627\u0645", "\u0627\u0639\u062a\u0635\u0627\u0645\u0627\u062a",
    "\u0647\u062a\u0627\u0641\u0627\u062a", "\u0634\u0639\u0627\u0631\u0627\u062a", "\u0634\u0647\u0627\u062f\u0627\u062a",
]
DOMAIN_KEYWORDS_NORMALIZED = [normalize_for_match(k) for k in DOMAIN_KEYWORDS]

def log_event(event, payload):
    line = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with open(LOG_DIR / "app.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

def load_chunks():
    chunks = []
    with open(INDEX_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_index_config():
    if not META_PATH.exists():
        return DEFAULT_INDEX_CONFIG
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return DEFAULT_INDEX_CONFIG

    config = {
        "embedding_model": meta.get("embedding_model") or DEFAULT_INDEX_CONFIG["embedding_model"],
        "hybrid_search": {
            "alpha": meta.get("hybrid_search", {}).get("alpha", DEFAULT_INDEX_CONFIG["hybrid_search"]["alpha"]),
            "top_k_faiss": meta.get("hybrid_search", {}).get(
                "top_k_faiss", DEFAULT_INDEX_CONFIG["hybrid_search"]["top_k_faiss"]
            ),
            "top_k_bm25": meta.get("hybrid_search", {}).get(
                "top_k_bm25", DEFAULT_INDEX_CONFIG["hybrid_search"]["top_k_bm25"]
            ),
            "top_k_final": meta.get("hybrid_search", {}).get(
                "top_k_final", DEFAULT_INDEX_CONFIG["hybrid_search"]["top_k_final"]
            ),
        },
    }
    return config

def ensure_index_assets():
    missing = [p.name for p in INDEX_REQUIRED_FILES if not p.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Index not ingested; missing files: {missing_list}. Run ingestion first."
        )

def _file_signature(path: Path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (path.name, None, None)
    return (path.name, stat.st_mtime_ns, stat.st_size)

def get_index_signature():
    ensure_index_assets()
    signatures = [_file_signature(p) for p in INDEX_REQUIRED_FILES + INDEX_OPTIONAL_FILES]
    return tuple(signatures)

def reset_resource_cache():
    _RESOURCE_CACHE["chunks"] = None
    _RESOURCE_CACHE["index"] = None
    _RESOURCE_CACHE["bm25"] = None
    _RESOURCE_CACHE["model"] = None
    _RESOURCE_CACHE["model_name"] = None
    _RESOURCE_CACHE["config"] = None
    _RESOURCE_CACHE["signature"] = None

def load_retrieval_resources():
    with _CACHE_LOCK:
        current_signature = get_index_signature()
        if _RESOURCE_CACHE["signature"] != current_signature:
            reset_resource_cache()
            _RESOURCE_CACHE["signature"] = current_signature
        if _RESOURCE_CACHE["config"] is None:
            _RESOURCE_CACHE["config"] = load_index_config()

        if _RESOURCE_CACHE["chunks"] is None:
            ensure_index_assets()
            _RESOURCE_CACHE["chunks"] = load_chunks()
            if not _RESOURCE_CACHE["chunks"]:
                raise RuntimeError("Index not ingested: chunks.jsonl is empty.")

        if _RESOURCE_CACHE["index"] is None:
            _RESOURCE_CACHE["index"] = faiss.read_index(str(INDEX_DIR / "faiss.index"))

        if _RESOURCE_CACHE["bm25"] is None:
            with open(INDEX_DIR / "bm25.pkl", "rb") as f:
                _RESOURCE_CACHE["bm25"] = pickle.load(f)

        model_name = _RESOURCE_CACHE["config"]["embedding_model"]
        if _RESOURCE_CACHE["model"] is None or _RESOURCE_CACHE["model_name"] != model_name:
            _RESOURCE_CACHE["model"] = SentenceTransformer(model_name)
            _RESOURCE_CACHE["model_name"] = model_name

        return (
            _RESOURCE_CACHE["chunks"],
            _RESOURCE_CACHE["index"],
            _RESOURCE_CACHE["bm25"],
            _RESOURCE_CACHE["model"],
            _RESOURCE_CACHE["config"],
        )

def tokenize_for_bm25(text: str):
    text_normalized = normalize_text(text)
    tokens = re.findall(r"\w+", text_normalized, re.UNICODE)
    return [t for t in tokens if len(t) > 1]

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-6:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

_FAISS_DISTANCE_METRICS = {
    m for m in (
        getattr(faiss, "METRIC_L2", None),
        getattr(faiss, "METRIC_L1", None),
        getattr(faiss, "METRIC_Linf", None),
        getattr(faiss, "METRIC_Lp", None),
    ) if m is not None
}

def faiss_scores_to_similarity(scores, index):
    metric = getattr(index, "metric_type", None)
    if metric in _FAISS_DISTANCE_METRICS:
        return -scores
    return scores

def build_prompt(question, context_blocks, response_lang, low_confidence=False):
    lines = [
        "You are a helpful assistant. Answer using ONLY the context.",
        "If the answer is not in the context, say you don't know and ask for more details.",
        "Cite sources like [1], [2].",
        f"Answer in {response_lang}."
    ]
    if low_confidence:
        lines.append("The retrieved context may be weak; be conservative.")
    header = "\n".join(lines)
    context = "\n\n".join(context_blocks)
    return f"{header}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def build_translation_prompt(text, target_lang):
    return (
        f"Translate the following text to {target_lang}. "
        "Preserve names and locations. Return only the translation.\n\n"
        f"Text:\n{text}\n\nTranslation:"
    )

def is_supported_lang(lang):
    return lang in ("ar", "fr", "en")

def decide_response_language(lang):
    if lang == "ar":
        return "Arabic"
    if lang == "fr":
        return "French"
    if lang == "en":
        return "English"
    return "French"

def decide_response_lang_code(lang):
    return lang if is_supported_lang(lang) else "fr"

def load_ontology():
    if not ONTOLOGY_PATH.exists():
        return {}
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def find_ontology_hits(query, ontology):
    qn = normalize_for_match(query)
    hits = []
    expansions = []
    for term, data in ontology.items():
        aliases = data.get("aliases", [])
        for alias in aliases:
            an = normalize_for_match(alias)
            if an and an in qn:
                hits.append(term)
                expansions.extend(data.get("related", []))
                break
    seen = set()
    uniq_expansions = []
    for e in expansions:
        if e not in seen:
            uniq_expansions.append(e)
            seen.add(e)
    return hits, uniq_expansions

def is_in_domain(query):
    q = normalize_for_match(query)
    return any(k and k in q for k in DOMAIN_KEYWORDS_NORMALIZED)

def run_with_timeout(fn, timeout_s, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            raise TimeoutError(f"Timeout after {timeout_s}s")

def llm_generate_ollama(prompt, model, base_url, timeout_s=60):
    from ollama import chat

    prev_host = os.getenv("OLLAMA_HOST")
    if base_url:
        os.environ["OLLAMA_HOST"] = base_url
    try:
        response = run_with_timeout(
            chat,
            timeout_s,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    finally:
        if prev_host is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = prev_host

    if isinstance(response, dict):
        return response.get("message", {}).get("content", "").strip()
    return response.message.content.strip()

def strip_think_blocks(text: str):
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def llm_generate_groq(prompt, model, timeout_s=60):
    from groq import Groq
    client = Groq()
    completion = run_with_timeout(
        client.chat.completions.create,
        timeout_s,
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95,
        stream=False,
    )
    raw = completion.choices[0].message.content or ""
    return strip_think_blocks(raw)

def get_llm_config():
    provider = os.getenv("RAG_LLM_PROVIDER", "ollama").lower()
    model = os.getenv("RAG_LLM_MODEL", "gemma3")
    timeout_s = int(os.getenv("RAG_LLM_TIMEOUT_S", "60"))

    if provider == "ollama":
        base_url = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434")
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "timeout_s": timeout_s,
            "api_key": None,
        }
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return {
            "provider": provider,
            "model": model if model != "gemma3" else "qwen/qwen3-32b",
            "base_url": None,
            "timeout_s": timeout_s,
            "api_key": api_key,
        }

    raise RuntimeError(f"Unsupported RAG_LLM_PROVIDER: {provider}")

def is_foreign_provider(cfg):
    if cfg["provider"] == "groq":
        return True
    base_url = (cfg.get("base_url") or "").lower()
    local_prefixes = (
        "http://localhost",
        "http://127.0.0.1",
        "http://0.0.0.0",
        "https://localhost",
        "https://127.0.0.1",
        "https://0.0.0.0",
    )
    if base_url.startswith(local_prefixes):
        return False
    return True

def llm_generate(prompt, cfg):
    if cfg["provider"] == "ollama":
        return llm_generate_ollama(prompt, cfg["model"], cfg["base_url"], cfg["timeout_s"])
    if cfg["provider"] == "groq":
        return llm_generate_groq(prompt, cfg["model"], cfg["timeout_s"])
    raise RuntimeError(f"Unsupported RAG_LLM_PROVIDER: {cfg['provider']}")

def encode_query(model, text, model_name):
    name = (model_name or "").lower()
    prefix = "query: " if "e5" in name else ""
    return model.encode([prefix + text], normalize_embeddings=True).astype(np.float32)

def translate_text(text, target_lang, cfg):
    prompt = build_translation_prompt(text, target_lang)
    return llm_generate(prompt, cfg)

def answer_question(q: str, cfg: dict):
    q = q.strip()
    if not q:
        raise ValueError("Empty question.")

    lang, _ = langid.classify(q)
    response_lang_label = decide_response_language(lang)
    response_lang_code = decide_response_lang_code(lang)
    tool_trace = ["language_detect"]

    allow_foreign = os.getenv("RAG_ALLOW_FOREIGN_SERVICES", "0") == "1"
    foreign_used = is_foreign_provider(cfg)

    translation_used = False
    translation_error = None
    translated_query = None
    query_for_retrieval = q
    question_for_answer = q

    decision = "answer"
    decision_reason = "ok"
    final = ""

    if foreign_used and not allow_foreign:
        decision = "refuse"
        decision_reason = "foreign_provider_blocked"
        final = (
            "Refusal: configured LLM provider appears to be a foreign service. "
            "Switch to a local model or set RAG_ALLOW_FOREIGN_SERVICES=1."
        )
    elif not is_supported_lang(lang):
        translation_used = True
        tool_trace.append("translate")
        try:
            translated_query = translate_text(q, response_lang_label, cfg)
            query_for_retrieval = translated_query
            question_for_answer = translated_query
        except Exception as e:
            translation_error = str(e)
            decision = "refuse"
            decision_reason = "translation_failed"
            final = "Please ask in Arabic, French, or English."

    citations = []
    context_blocks = []
    low_conf = False
    best = 0.0
    ontology_hits = []
    expanded_terms = []
    expanded_query = query_for_retrieval

    if decision == "answer":
        ontology = load_ontology()
        ontology_hits, expanded_terms = find_ontology_hits(query_for_retrieval, ontology)
        if ontology_hits:
            tool_trace.append("ontology_lookup")
        if expanded_terms:
            expanded_query = query_for_retrieval + " " + " ".join(expanded_terms)

        try:
            chunks, index, bm25, model, index_cfg = load_retrieval_resources()
        except RuntimeError as e:
            decision = "refuse"
            decision_reason = "index_missing"
            final = str(e)
            chunks = []
            index = None
            bm25 = None
            model = None
            index_cfg = None

    if decision == "answer":
        hs_cfg = index_cfg["hybrid_search"]
        alpha = float(hs_cfg["alpha"])
        top_k_faiss = int(hs_cfg["top_k_faiss"])
        top_k_bm25 = int(hs_cfg["top_k_bm25"])
        top_k_final = int(hs_cfg["top_k_final"])

        model_name = index_cfg.get("embedding_model")
        qvec = encode_query(model, expanded_query, model_name)

        if getattr(index, "d", None) is not None and index.d != qvec.shape[1]:
            raise RuntimeError(
                f"Embedding dim mismatch: query dim={qvec.shape[1]} vs index dim={index.d}. "
                "Rebuild the index with the same embedding model."
            )

        faiss_k = min(top_k_faiss, len(chunks))
        scores_sem, ids_sem = index.search(qvec, faiss_k)

        ids_sem = ids_sem[0].tolist()
        scores_sem = scores_sem[0].tolist()

        pairs = [(i, s) for i, s in zip(ids_sem, scores_sem) if 0 <= i < len(chunks)]
        if pairs:
            ids_sem, scores_sem = zip(*pairs)
            ids_sem = list(ids_sem)
            scores_sem = list(scores_sem)
        else:
            ids_sem = []
            scores_sem = []

        scores_sem = faiss_scores_to_similarity(np.asarray(scores_sem, dtype=np.float32), index)
        scores_sem = normalize_scores(scores_sem.tolist())

        bm_scores_all = np.array(bm25.get_scores(tokenize_for_bm25(expanded_query)))
        bm_scores_all = normalize_scores(bm_scores_all)
        ids_lex = np.argsort(-bm_scores_all)[:top_k_bm25].tolist()

        sem_map = {i: float(s) for i, s in zip(ids_sem, scores_sem)}
        lex_map = {i: float(bm_scores_all[i]) for i in ids_lex}

        all_ids = list(dict.fromkeys(ids_sem + ids_lex))
        hybrid = []
        for i in all_ids:
            hs = alpha * sem_map.get(i, 0.0) + (1 - alpha) * lex_map.get(i, 0.0)
            hybrid.append((i, hs))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        top = hybrid[:top_k_final]

        min_conf = float(os.getenv("RAG_DOMAIN_SCORE_THRESHOLD", os.getenv("RAG_MIN_CONFIDENCE", "0.35")))
        best = top[0][1] if top else 0.0
        low_conf = best < min_conf
        tool_trace.append("retrieve_hybrid")

        for rank, (i, hs) in enumerate(top, 1):
            c = chunks[i] if 0 <= i < len(chunks) else {}
            if not isinstance(c, dict):
                c = {}
            source_file = c.get("source_file", "unknown")
            chunk_id = c.get("chunk_id")
            if chunk_id is None:
                chunk_id = i
            text = c.get("text", "")

            citations.append({
                "rank": rank,
                "hybrid_score": round(hs, 4),
                "source_file": source_file,
                "chunk_id": chunk_id,
                "snippet": text[:280].replace("\n", " ")
            })

            text_content = text[:1200]
            context_blocks.append(
                f"[{rank}] source_file={source_file} chunk_id={chunk_id}\n{text_content}"
            )

        if low_conf:
            decision = "clarify"
            decision_reason = "out_of_domain_or_low_confidence"
            final = (
                "I could not find strong evidence in the local archive. "
                "Please add more details or narrow the question."
            )
        else:
            prompt = build_prompt(question_for_answer, context_blocks, response_lang_label, low_confidence=False)
            try:
                final = llm_generate(prompt, cfg)
                tool_trace.append("llm_generate")
            except Exception as e:
                final = f"LLM error: {e}"

    evaluation_hooks = {
        "after_tool_selection": {
            "foreign_service_used": foreign_used,
            "translation_used": translation_used,
            "ontology_used": bool(ontology_hits),
        },
        "after_retrieval": {
            "best_hybrid_score": round(best, 4),
            "citations_count": len(citations),
        },
        "after_generate": {
            "answer_language": response_lang_label,
            "decision": decision,
        },
    }

    out = {
        "question": q,
        "detected_lang": lang,
        "response_lang_code": response_lang_code,
        "response_lang": response_lang_label,
        "translation_used": translation_used,
        "translation_error": translation_error,
        "translated_query": translated_query,
        "ontology_hits": ontology_hits,
        "expanded_terms": expanded_terms,
        "expanded_query": expanded_query if expanded_terms else None,
        "decision": decision,
        "decision_reason": decision_reason,
        "tool_trace": tool_trace,
        "top_citations": citations,
        "confidence_rule": "best_score < RAG_DOMAIN_SCORE_THRESHOLD => out_of_domain_or_low_confidence",
        "low_confidence": low_conf,
        "evaluation_hooks": evaluation_hooks,
        "final_answer": final
    }

    log_event("ask", {
        "question": q,
        "decision": decision,
        "decision_reason": decision_reason,
        "detected_lang": lang,
        "response_lang": response_lang_code,
        "translation_used": translation_used,
        "translated_query": translated_query,
        "ontology_hits": ontology_hits,
        "foreign_service_used": foreign_used,
        "best_hybrid_score": round(best, 4),
    })

    return out

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")

class AskResponse(BaseModel):
    data: Dict[str, Any]

app = FastAPI(
    title="RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    ok = True
    missing = []
    for p in [INDEX_DIR / "chunks.jsonl", INDEX_DIR / "faiss.index", INDEX_DIR / "bm25.pkl"]:
        if not p.exists():
            ok = False
            missing.append(p.name)

    provider = os.getenv("RAG_LLM_PROVIDER", "ollama")
    model = os.getenv("RAG_LLM_MODEL", "gemma3")

    return {
        "ok": ok,
        "index_dir": str(INDEX_DIR),
        "missing_index_files": missing,
        "llm_provider": provider,
        "llm_model": model,
    }

@app.post("/reload")
def reload_resources():
    with _CACHE_LOCK:
        reset_resource_cache()
    return {"ok": True, "message": "Cache cleared. Resources will reload on next /ask."}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        cfg = get_llm_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"LLM config error: {e}")

    try:
        out = answer_question(req.question, cfg)
        return {"data": out}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
