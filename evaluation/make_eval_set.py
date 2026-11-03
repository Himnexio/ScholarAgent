"""Create the evaluation set from HELD-OUT papers (never used for fine-tuning).

    python -m evaluation.make_eval_set --n 100

Open data/eval_set.jsonl afterwards and delete or fix bad questions - a reviewed set of 60-100 is worth more
than 500 unreviewed ones.
"""
import argparse
import logging
import random

from scholaragent import config as cfg
from scholaragent.synth import generate_questions
from scholaragent.utils import hash_split, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    chunks = [
        c for c in read_jsonl(cfg.CHUNKS_PATH)
        if hash_split(c["paper_id"]) == "test" and c["section"] != "Abstract" and len(c["text"].split()) >= 80
    ]
    if not chunks:
        raise SystemExit("No eligible chunks. Run ingest first (and make sure the corpus has enough papers).")
    random.Random(args.seed).shuffle(chunks)
    added = generate_questions(chunks[: args.n], cfg.EVAL_SET_PATH, sleep=args.sleep)
    print(f"added {added} questions -> {cfg.EVAL_SET_PATH}")


if __name__ == "__main__":
    main()
