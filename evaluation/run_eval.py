"""Retrieval ablation (+ optional answer-level LLM judging).

    python -m evaluation.run_eval                      # retrieval table: vector / bm25 / hybrid / hybrid+rerank
    python -m evaluation.run_eval --answers 30         # also run the agent on 30 questions and judge them
    EMBED_MODEL=models/bge-small-scholar python -m evaluation.run_eval --label finetuned

Retrieval metrics are computed against the chunk the question was generated from (strict) and against its paper
(lenient). They need no LLM calls, so you can re-run ablations as often as you like.
"""
import argparse
import json
import logging

from tqdm import tqdm

from scholaragent import config as cfg
from scholaragent.llm import ask_json
from scholaragent.tools import get_retriever
from scholaragent.utils import read_jsonl, slugify

logging.basicConfig(level=logging.WARNING)

CONFIGS = {
    "vector": dict(mode="vector", rerank=False),
    "bm25": dict(mode="bm25", rerank=False),
    "hybrid": dict(mode="hybrid", rerank=False),
    "hybrid+rerank": dict(mode="hybrid", rerank=True),
}

JUDGE_PROMPT = """You are grading a research assistant's answer.

Question: {question}
Reference answer: {reference}

Retrieved context given to the assistant:
{context}

Assistant answer:
{answer}

Return JSON: {{"correct": 0 or 1, "faithful": 0 or 1, "reason": "one sentence"}}
correct = 1 if the answer matches the reference answer in substance.
faithful = 1 if every claim in the assistant answer is supported by the retrieved context."""


def retrieval_eval(rows: list[dict], k: int) -> dict:
    retriever = get_retriever()
    results = {}
    for name, kwargs in CONFIGS.items():
        chunk_hits = paper_hits = 0
        rr = 0.0
        for row in tqdm(rows, desc=name):
            docs = retriever.search(row["question"], k=k, **kwargs)
            ids = [d["chunk_id"] for d in docs]
            if row["chunk_id"] in ids:
                chunk_hits += 1
                rr += 1.0 / (ids.index(row["chunk_id"]) + 1)
            if row["paper_id"] in {d["paper_id"] for d in docs}:
                paper_hits += 1
        n = len(rows)
        results[name] = {f"chunk_hit@{k}": chunk_hits / n, f"paper_hit@{k}": paper_hits / n, f"chunk_mrr@{k}": rr / n}
    return results


def answer_eval(rows: list[dict]) -> dict:
    from scholaragent.agent import run_agent

    correct = faithful = retries = 0
    for row in tqdm(rows, desc="agent + judge"):
        out = run_agent(row["question"])
        retries += int(out["attempts"] > 1)
        verdict = ask_json(JUDGE_PROMPT.format(
            question=row["question"], reference=row["answer"],
            context="\n---\n".join(c[:900] for c in out["contexts"]), answer=out["answer"],
        ))
        correct += int(verdict.get("correct") == 1)
        faithful += int(verdict.get("faithful") == 1)
    n = len(rows)
    return {"n": n, "correctness": correct / n, "faithfulness": faithful / n, "retry_rate": retries / n}


def to_markdown(label: str, retrieval: dict, answers: dict | None) -> str:
    cols = list(next(iter(retrieval.values())))
    lines = [f"### {label}  (embedder: `{cfg.EMBED_MODEL}`)", "",
             "| retriever | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in retrieval.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    if answers:
        lines += ["", f"Agent answers (n={answers['n']}): correctness {answers['correctness']:.2f}, "
                      f"faithfulness {answers['faithfulness']:.2f}, retry rate {answers['retry_rate']:.2f}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answers", type=int, default=0, help="also run the agent + LLM judge on N questions")
    ap.add_argument("--label", default=None, help="name for the results file (default: embedder name)")
    args = ap.parse_args()

    rows = read_jsonl(cfg.EVAL_SET_PATH)
    if not rows:
        raise SystemExit(f"{cfg.EVAL_SET_PATH} is empty. Run: python -m evaluation.make_eval_set")
    label = args.label or slugify(cfg.EMBED_MODEL.rsplit("/", 1)[-1])

    retrieval = retrieval_eval(rows, args.k)
    answers = answer_eval(rows[: args.answers]) if args.answers else None

    md = to_markdown(label, retrieval, answers)
    print("\n" + md)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / f"{label}.md").write_text(md + "\n", encoding="utf-8")
    (cfg.RESULTS_DIR / f"{label}.json").write_text(json.dumps({"retrieval": retrieval, "answers": answers}, indent=2))


if __name__ == "__main__":
    main()
