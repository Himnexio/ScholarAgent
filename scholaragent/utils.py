from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries", "method", "methods",
    "methodology", "approach", "model", "experiments", "experimental setup", "results", "discussion",
    "analysis", "limitations", "conclusion", "conclusions", "future work", "acknowledgements",
    "acknowledgments", "references", "bibliography",
}
NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\.?\s+([A-Z][A-Za-z0-9 ,:&\-/()']{2,70})$")
ROMAN_HEADING = re.compile(r"^([IVX]{1,4})\.\s+([A-Z][A-Za-z ,:&\-/()']{2,70})$")
STOP_HEADINGS = {"references", "bibliography"}

def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def heading_of(line: str) -> str | None:
    s = line.strip()
    if not s or len(s) > 80:
        return None
    bare = s.rstrip(".:")
    if bare.lower() in KNOWN_HEADINGS:
        return bare.title()
    m = NUMBERED_HEADING.match(s)
    if m and not s.endswith((".", ",")):
        return f"{m.group(1)} {m.group(2).strip()}"
    m = ROMAN_HEADING.match(s)
    if m:
        return m.group(2).strip().title()
    return None

def is_stop_heading(name: str) -> bool:
    return re.sub(r"^\d+(?:\.\d+)?\s+", "", name).lower() in STOP_HEADINGS

def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current, buf = "Front matter", []
    for line in text.splitlines():
        h = heading_of(line)
        if h is None:
            buf.append(line)
            continue
        if buf:
            sections.append((current, "\n".join(buf)))
        buf = []
        if is_stop_heading(h):
            return sections
        current = h
    if buf:
        sections.append((current, "\n".join(buf)))
    return sections

def chunk_text(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def contextual_text(chunk: dict) -> str:
    return f"{chunk['title']} - {chunk['section']}\n{chunk['text']}"

def rrf(rank_lists: Iterable[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranks in rank_lists:
        for r, item in enumerate(ranks):
            scores[item] += 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in: {text[:200]!r}")
    return json.loads(text[start : end + 1])

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

def check_citations(answer: str, n_docs: int) -> tuple[str, list[int]]:
    used: list[int] = []
    def repl(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        valid = [n for n in nums if 1 <= n <= n_docs]
        for n in valid:
            if n not in used:
                used.append(n)
        return f"[{', '.join(map(str, valid))}]" if valid else ""

    cleaned = re.sub(r"[ \t]{2,}", " ", _CITE.sub(repl, answer))
    return cleaned, used

def hash_split(paper_id: str, test_mod: int = 5) -> str:
    return "test" if int(hashlib.md5(paper_id.encode()).hexdigest(), 16) % test_mod == 0 else "train"

def slugify(name: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_-")[:max_len] or "x"

def md5_short(s: str, n: int = 6) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:n]

def safe_id(paper_id: str) -> str:
    return paper_id.replace("/", "_")

def short_authors(authors: list[str], n: int = 3) -> str:
    return ", ".join(authors[:n]) + (" et al." if len(authors) > n else "")

def read_jsonl(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path | str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")