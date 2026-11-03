import argparse
import math
import random
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from scholaragent import config as cfg
from scholaragent.utils import read_jsonl

def embed(model: SentenceTransformer, texts: list[str], device: str) -> torch.Tensor:
    feats = {k: v.to(device) for k, v in model.tokenize(texts).items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=cfg.EMBED_MODEL)
    ap.add_argument("--pairs", default=str(cfg.TRAIN_PAIRS_PATH))
    ap.add_argument("--out", default="models/bge-small-scholar")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="inverse temperature of the softmax")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    pairs = read_jsonl(args.pairs)
    if len(pairs) < args.batch * 2:
        raise SystemExit(f"Only {len(pairs)} pairs found - run `python -m training.make_pairs` first.")
    print(f"{len(pairs)} pairs | device={device} | base={args.base}")
    model = SentenceTransformer(args.base, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(pairs) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), args.batch):
            batch = pairs[i : i + args.batch]
            if len(batch) < 2:
                continue
            q = embed(model, [cfg.QUERY_INSTRUCTION + p["question"] for p in batch], device)
            d = embed(model, [p["passage"] for p in batch], device)
            logits = q @ d.T * args.scale
            labels = torch.arange(len(batch), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{steps} loss {loss.item():.4f}")

    model.save(args.out)
    print(f"saved fine-tuned model to {args.out}")

if __name__ == "__main__":
    main()
