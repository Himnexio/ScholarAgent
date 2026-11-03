from __future__ import annotations
import argparse
import logging
import re
import time
from pathlib import Path
import arxiv
import fitz
import requests
from tqdm import tqdm
from scholaragent import config as cfg
from scholaragent.utils import (chunk_text, clean_text, safe_id, short_authors, split_sections, write_jsonl)

log = logging.getLogger(__name__)
DEFAULT_QUERY = 'abs:"LLM agents" AND (cat:cs.CL OR cat:cs.AI)'

def fetch_metadata(query: str, max_results: int) -> list[dict]:
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for r in client.results(search):
        pid = re.sub(r"v\d+$", "", r.get_short_id())
        papers.append({
            "paper_id": pid,
            "title": " ".join(r.title.split()),
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "abstract": " ".join(r.summary.split()),
            "pdf_url": r.pdf_url.replace("http://", "https://"),
            "url": f"https://arxiv.org/abs/{pid}",
        })
    return papers

def download_pdf(paper: dict, pdf_dir: Path) -> Path | None:
    path = pdf_dir / f"{safe_id(paper['paper_id'])}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    try:
        resp = requests.get(paper["pdf_url"], timeout=60, headers={"User-Agent": "scholaragent/0.1 (student project)"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("download failed for %s: %s", paper["paper_id"], exc)
        return None
    path.write_bytes(resp.content)
    time.sleep(3)
    return path

def extract_text(pdf_path: Path) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        log.warning("could not parse %s: %s", pdf_path.name, exc)
        return ""

def build_chunks(paper: dict, text: str) -> list[dict]:
    pid = paper["paper_id"]
    chunks: list[dict] = []

    def add(section: str, body: str) -> None:
        for piece in chunk_text(clean_text(body), cfg.CHUNK_WORDS, cfg.CHUNK_OVERLAP):
            if len(piece.split()) < cfg.MIN_CHUNK_WORDS:
                continue
            chunks.append({
                "chunk_id": f"{safe_id(pid)}_{len(chunks):04d}",
                "paper_id": pid,
                "title": paper["title"],
                "authors": short_authors(paper["authors"]),
                "year": paper["year"],
                "url": paper["url"],
                "section": section,
                "text": piece,
            })

    add("Abstract", paper["abstract"])
    sections = split_sections(text)
    if len(sections) > 1:
        sections = [s for s in sections if s[0] != "Front matter"]
    for name, body in sections:
        if name.lower() != "abstract":
            add(name, body)
    return chunks

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=DEFAULT_QUERY, help="raw arXiv API query string")
    ap.add_argument("--max", type=int, default=300, dest="max_results")
    args = ap.parse_args()

    cfg.PDF_DIR.mkdir(parents=True, exist_ok=True)
    papers = fetch_metadata(args.query, args.max_results)
    log.info("arXiv returned %d papers", len(papers))

    kept, all_chunks = [], []
    for paper in tqdm(papers, desc="download+parse"):
        pdf = download_pdf(paper, cfg.PDF_DIR)
        if pdf is None:
            continue
        chunks = build_chunks(paper, extract_text(pdf))
        if len(chunks) < 3:
            log.warning("skipping %s (only %d usable chunks)", paper["paper_id"], len(chunks))
            continue
        kept.append(paper)
        all_chunks.extend(chunks)

    write_jsonl(cfg.PAPERS_PATH, kept)
    write_jsonl(cfg.CHUNKS_PATH, all_chunks)
    log.info("saved %d papers and %d chunks to %s", len(kept), len(all_chunks), cfg.DATA_DIR)

if __name__ == "__main__":
    main()