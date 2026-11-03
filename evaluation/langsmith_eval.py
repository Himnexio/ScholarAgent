"""OPTIONAL: upload the eval set to LangSmith and run the agent against it as a tracked experiment.

    export LANGSMITH_API_KEY=...   # plus GOOGLE_API_KEY etc.
    python -m evaluation.langsmith_eval
"""
from langsmith import Client, evaluate

from scholaragent import config as cfg
from scholaragent.agent import run_agent
from scholaragent.utils import read_jsonl

DATASET = "scholaragent-eval"


def upload() -> None:
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(DATASET, description="Synthetic QA over arXiv chunks (held-out papers)")
    rows = read_jsonl(cfg.EVAL_SET_PATH)
    client.create_examples(
        inputs=[{"question": r["question"]} for r in rows],
        outputs=[{"answer": r["answer"], "chunk_id": r["chunk_id"], "paper_id": r["paper_id"]} for r in rows],
        dataset_id=ds.id,
    )


def target(inputs: dict) -> dict:
    out = run_agent(inputs["question"])
    return {"answer": out["answer"], "cited_paper_ids": [c["paper_id"] for c in out["citations"]],
            "attempts": out["attempts"]}


def cites_gold_paper(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "cites_gold_paper", "score": float(reference_outputs["paper_id"] in outputs["cited_paper_ids"])}


def used_retry(outputs: dict) -> dict:
    return {"key": "used_retry", "score": float(outputs["attempts"] > 1)}


if __name__ == "__main__":
    upload()
    evaluate(target, data=DATASET, evaluators=[cites_gold_paper, used_retry],
             experiment_prefix="scholaragent", max_concurrency=2)
