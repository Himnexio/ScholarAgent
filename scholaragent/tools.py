from __future__ import annotations
from functools import lru_cache
from langsmith import traceable
from scholaragent import config as cfg
from scholaragent.retrieval import Retriever
from scholaragent.utils import read_jsonl

@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever()

@lru_cache(maxsize=1)
def _papers() -> dict[str, dict]:
    return {p["paper_id"]: p for p in read_jsonl(cfg.PAPERS_PATH)}

def _snippet(text: str, words: int = 60) -> str:
    return " ".join(text.split()[:words])

@traceable(name="search_chunks", run_type="retriever")
def search_chunks(query: str, k: int = 6, mode: str = "hybrid", rerank: bool | None = None,
                  paper_id: str | None = None) -> list[dict]:
    return get_retriever().search(query, k=k, mode=mode, rerank=rerank, paper_id=paper_id)

def search_papers(query: str, k: int = 5) -> list[dict]:
    """Return the k most relevant *papers* (best passage per paper)."""
    best: dict[str, dict] = {}
    for c in search_chunks(query, k=max(k * 3, 10)):  # already sorted best-first
        if c["paper_id"] not in best:
            best[c["paper_id"]] = {
                "paper_id": c["paper_id"], "title": c["title"], "authors": c["authors"], "year": c["year"],
                "url": c["url"], "section": c["section"], "score": c["score"], "snippet": _snippet(c["text"]),
            }
    return list(best.values())[:k]

def get_paper(paper_id: str) -> dict | None:
    p = _papers().get(paper_id)
    return None if p is None else {k: p[k] for k in ("paper_id", "title", "authors", "year", "abstract", "url")}

def get_paper_evidence(paper_id: str, question: str, k: int = 3) -> list[dict]:
    """Top passages of ONE paper for a question (e.g. 'what datasets were used?')."""
    return [
        {"section": c["section"], "score": c["score"], "text": c["text"]}
        for c in search_chunks(question, k=k, paper_id=paper_id)
    ]

def compare_papers(paper_ids: list[str], aspect: str, k_per_paper: int = 3) -> dict:
    """Collect evidence about the same aspect from several papers; the caller's LLM does the comparing."""
    out = []
    for pid in paper_ids:
        meta = get_paper(pid)
        if meta is None:
            out.append({"paper_id": pid, "error": "paper not in index"})
            continue
        out.append({
            "paper_id": pid, "title": meta["title"], "year": meta["year"],
            "passages": get_paper_evidence(pid, aspect, k_per_paper),
        })
    return {"aspect": aspect, "papers": out}