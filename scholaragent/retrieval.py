from __future__ import annotations
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from scholaragent import config as cfg
from scholaragent.utils import contextual_text, read_jsonl, rrf, tokenize
log = logging.getLogger(__name__)

class Retriever:
    def __init__(self) -> None:
        self.chunks = read_jsonl(cfg.CHUNKS_PATH)
        if not self.chunks:
            raise RuntimeError(f"No chunks in {cfg.CHUNKS_PATH}. Run: python -m scholaragent.ingest")
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self._paper_ids = np.array([c["paper_id"] for c in self.chunks])
        self.bm25 = BM25Okapi([tokenize(contextual_text(c)) for c in self.chunks])
        self._index = None
        self._reranker = None

    @property
    def index(self):
        if self._index is None:
            from scholaragent.index import load_index
            self._index = load_index()
        return self._index

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(cfg.RERANKER_MODEL, max_length=512)
        return self._reranker

    def _dense(self, query: str, k: int, paper_id: str | None) -> list[str]:
        filters = None
        if paper_id:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
            filters = MetadataFilters(filters=[MetadataFilter(key="paper_id", value=paper_id)])
        nodes = self.index.as_retriever(similarity_top_k=k, filters=filters).retrieve(query)
        return [n.node.metadata["chunk_id"] for n in nodes]

    def _sparse(self, query: str, k: int, paper_id: str | None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        if paper_id:
            scores = np.where(self._paper_ids == paper_id, scores, -np.inf)
        top = np.argsort(scores)[::-1][:k]
        return [self.chunks[i]["chunk_id"] for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def search(self, query: str, k: int = 6, mode: str = "hybrid", rerank: bool | None = None,
               paper_id: str | None = None) -> list[dict]:
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        rerank = cfg.USE_RERANKER if rerank is None else rerank
        pool = max(k * 4, 20)
        rank_lists = []
        if mode in {"vector", "hybrid"}:
            rank_lists.append(self._dense(query, pool, paper_id))
        if mode in {"bm25", "hybrid"}:
            rank_lists.append(self._sparse(query, pool, paper_id))
        fused = rrf(rank_lists)[: max(k, 20) if rerank else k]
        docs = [{**self.by_id[cid], "score": score} for cid, score in fused]
        return self.rerank(query, docs, k) if rerank else docs

    def rerank(self, query: str, docs: list[dict], k: int) -> list[dict]:
        if not docs:
            return []
        scores = self.reranker.predict([(query, contextual_text(d)) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**d, "score": float(s)} for d, s in ranked]