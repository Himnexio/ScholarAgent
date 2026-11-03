from __future__ import annotations
import argparse
import logging
from functools import lru_cache
from scholaragent import config as cfg
from scholaragent.utils import read_jsonl
log = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embed_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name=cfg.EMBED_MODEL, query_instruction=cfg.QUERY_INSTRUCTION or None)

def _collection(rebuild: bool = False):
    import chromadb
    cfg.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    name = cfg.collection_name()
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name)

def _vector_store(collection):
    from llama_index.vector_stores.chroma import ChromaVectorStore
    return ChromaVectorStore(chroma_collection=collection)

def build_index(rebuild: bool = False) -> None:
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from scholaragent.utils import contextual_text

    chunks = read_jsonl(cfg.CHUNKS_PATH)
    if not chunks:
        raise SystemExit(f"{cfg.CHUNKS_PATH} is empty. Run: python -m scholaragent.ingest")
    collection = _collection(rebuild)
    if collection.count() > 0:
        log.info("collection %s already has %d vectors; use --rebuild to recreate", collection.name, collection.count())
        return

    nodes = []
    for c in chunks:
        meta = {"chunk_id": c["chunk_id"], "paper_id": c["paper_id"], "section": c["section"]}
        nodes.append(TextNode(
            id_=c["chunk_id"],
            text=contextual_text(c),
            metadata=meta,
            excluded_embed_metadata_keys=list(meta),
            excluded_llm_metadata_keys=list(meta),
        ))
    storage = StorageContext.from_defaults(vector_store=_vector_store(collection))
    VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model(), show_progress=True)
    log.info("indexed %d chunks into collection %s", len(nodes), collection.name)

def load_index():
    from llama_index.core import VectorStoreIndex
    collection = _collection()
    if collection.count() == 0:
        raise RuntimeError(f"Collection {collection.name} is empty. Run: python -m scholaragent.index")
    return VectorStoreIndex.from_vector_store(_vector_store(collection), embed_model=get_embed_model())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    build_index(rebuild=ap.parse_args().rebuild)