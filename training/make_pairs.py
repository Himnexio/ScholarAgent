import argparse
import logging
import random
from scholaragent import config as cfg
from scholaragent.synth import generate_questions
from scholaragent.utils import hash_split, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    chunks = [
        c for c in read_jsonl(cfg.CHUNKS_PATH)
        if hash_split(c["paper_id"]) == "train" and c["section"] != "Abstract" and len(c["text"].split()) >= 80
    ]
    random.Random(args.seed).shuffle(chunks)
    added = generate_questions(chunks[: args.n], cfg.TRAIN_PAIRS_PATH, sleep=args.sleep)
    print(f"added {added} pairs -> {cfg.TRAIN_PAIRS_PATH}")

if __name__ == "__main__":
    main()