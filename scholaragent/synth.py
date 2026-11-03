from __future__ import annotations
import logging
import time
from pathlib import Path
from tqdm import tqdm
from scholaragent.llm import ask_json
from scholaragent.utils import append_jsonl, contextual_text, read_jsonl

log = logging.getLogger(__name__)
QA_PROMPT = """You are building a test set for a research-paper search assistant.

Passage from the paper "{title}" (section: {section}):
\"\"\"{text}\"\"\"
Write ONE specific question that can be answered using ONLY this passage. It must make sense on its own to someone
searching a large collection of papers: include the key technical terms and never say "the passage", "this paper"
or "the text". Then write the short, correct answer.
Return JSON: {{"question": "...", "answer": "..."}}"""

def generate_questions(chunks: list[dict], out_path: Path, sleep: float = 0.5) -> int:
    """Append {question, answer, passage, chunk_id, paper_id} rows to out_path. Resumable. Returns #new rows."""
    done = {r["chunk_id"] for r in read_jsonl(out_path)}
    added = 0
    for c in tqdm(chunks, desc="generating questions"):
        if c["chunk_id"] in done:
            continue
        try:
            qa = ask_json(QA_PROMPT.format(title=c["title"], section=c["section"], text=c["text"]))
        except Exception as exc:
            log.warning("skipping %s: %s", c["chunk_id"], exc)
            continue
        question, answer = str(qa.get("question", "")).strip(), str(qa.get("answer", "")).strip()
        if len(question.split()) < 5 or not answer:
            continue
        append_jsonl(out_path, {
            "question": question, "answer": answer, "chunk_id": c["chunk_id"], "paper_id": c["paper_id"],
            "passage": contextual_text(c),
        })
        added += 1
        time.sleep(sleep)
    return added