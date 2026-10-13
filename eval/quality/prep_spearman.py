"""One-shot prep: turn a cached training split into a corpus the JS Spearman
eval can consume.

Reads `training/distill/cache/<name>.test.pt`, decodes `input_ids` back to text
via bert-base-uncased, and writes `eval/quality/corpus/spearman_<name>.json`:

    [
        {"text": "reset my password",        "teacher_emb": [0.012, -0.034, ...]},
        {"text": "how do I cancel my plan",  "teacher_emb": [...]},
        ...
    ]

Teacher embeddings are L2-normalized 384-d MiniLM-L6 vectors (already computed
during prep). We don't recompute them — the JS eval just consumes them as the
"gold" cosine source.

Usage:
    .venv/bin/python eval/quality/prep_spearman.py            # defaults to smoke_1k
    .venv/bin/python eval/quality/prep_spearman.py msmarco_mix_1M --sample 5000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT       = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/distill"))
from tokenization import load_tokenizer
CACHE_DIR  = ROOT / "training" / "distill" / "cache"
CORPUS_DIR = ROOT / "eval" / "quality" / "corpus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="smoke_1k",
                    help="cache name (matches <name>.test.pt). Default: smoke_1k")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--sample", type=int, default=None,
                    help="If set, take a random subsample of N items (with seed=42).")
    args = ap.parse_args()

    src = args.cache_dir / f"{args.name}.test.pt"
    if not src.exists():
        raise SystemExit(f"missing: {src}")

    print(f"loading {src}")
    samples = torch.load(src, weights_only=False)
    print(f"  {len(samples)} samples")

    if args.sample is not None and args.sample < len(samples):
        random.Random(42).shuffle(samples)
        samples = samples[: args.sample]
        print(f"  subsampled to {len(samples)}")

    manifest = json.loads((args.cache_dir / f"{args.name}.manifest.json").read_text())
    tok = load_tokenizer(manifest.get("tokenizer"))

    out = []
    for s in samples:
        text = s.get("text") or tok.decode(s["input_ids"].tolist(), skip_special_tokens=True)
        out.append({
            "text":        text,
            "teacher_emb": s["teacher_emb"].tolist(),
        })

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dst = CORPUS_DIR / f"spearman_{args.name}.json"
    with open(dst, "w") as f:
        json.dump(out, f)
    size_mb = dst.stat().st_size / 1e6
    print(f"wrote {dst}  ({len(out)} items, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
