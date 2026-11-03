from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from scholaragent.utils import md5_short, slugify

load_dotenv()

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PDF_DIR = DATA_DIR / "pdfs"
PAPERS_PATH = DATA_DIR / "papers.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"
TRAIN_PAIRS_PATH = DATA_DIR / "train_pairs.jsonl"
RESULTS_DIR = Path("evaluation/results")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
MIN_CHUNK_WORDS = 30
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = _bool("USE_RERANKER", True)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "")
CONTEXT_K = int(os.getenv("CONTEXT_K", "6"))
CANDIDATES_PER_QUERY = int(os.getenv("CANDIDATES_PER_QUERY", "12"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "2"))

def collection_name() -> str:
    """One Chroma collection per embedding model, so baseline and fine-tuned indexes never mix."""
    return f"papers_{slugify(Path(EMBED_MODEL).name)}_{md5_short(EMBED_MODEL)}"