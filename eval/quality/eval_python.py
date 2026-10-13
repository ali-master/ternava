"""Python-side Spearman eval — mirror of eval/quality/spearman.js for non-WASM
model variants (the fp32 pre-QAT student checkpoint).

Loads a student checkpoint into a fresh StudentEncoder, embeds the same corpus
the JS script uses (eval/quality/corpus/spearman_<name>.json), generates the
same deterministic pair sample (seed=42, 1000 pairs), and computes Spearman
between student-side cosines and the precomputed teacher cosines.

Use cases:
  - fp32 baseline (pre-QAT): anchors the "no quantization" dot on the chart
  - Any other student ckpt: parity checks, sanity-checking the WASM number

Usage:
  .venv/bin/python eval/quality/eval_python.py \\
      --ckpt training/distill/runs/fp32-baseline-662956d/checkpoint_ep30.pt \\
      --label "student-fp32-ep30"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "distill"))
from model import StudentEncoder
from tokenization import load_tokenizer

CORPUS_DIR = ROOT / "eval" / "quality" / "corpus"


# ── Stats (kept ASCII-equivalent to the JS version) ─────────────────────────

def ranks(arr: list[float]) -> list[float]:
    n = len(arr)
    idx = sorted(range(n), key=lambda i: arr[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[idx[j + 1]] == arr[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx  = sum((x[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy  = sum((y[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


# Same Mulberry32 the JS script uses → same pair sample.
def mulberry32(seed: int):
    s = [seed & 0xFFFFFFFF]
    def rng():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rng


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ── Run ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True, help="Path to student checkpoint .pt")
    ap.add_argument("--corpus", default="spearman_smoke_1k.json")
    ap.add_argument("--pairs",  type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--label",  required=True, help="Short name for stdout/results")
    args = ap.parse_args()

    # Load corpus
    corpus_path = CORPUS_DIR / args.corpus
    items = json.loads(corpus_path.read_text())
    N = len(items)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    if ckpt.get("config", {}).get("enable_qat"):
        raise ValueError("This evaluator is fp32-only; use the packed WASM bilingual evaluator for QAT")
    state = ckpt["model_state"]

    # Infer arch from state_dict (vocab + d_model from embedding weight shape, etc.)
    emb_w = state["embedding.weight"]
    vocab_size, d_model = emb_w.shape
    proj_w = state["projection.weight"]
    output_dim = proj_w.shape[0]
    n_layers = len({k.split(".")[1] for k in state.keys() if k.startswith("layers.")})
    ffn_dim  = state["layers.0.ff.fc1.weight"].shape[0]   # FeedForward.fc1: ffn_dim × d_model

    model = StudentEncoder(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=ckpt.get("config", {}).get("n_heads", 4), ffn_dim=ffn_dim, output_dim=output_dim, dropout=0.0,
        position_embeddings=ckpt.get("config", {}).get("position_embeddings", False),
    )
    model.load_state_dict(state)
    model.eval()

    n_params = model.count_parameters()
    fp32_size_mb = n_params * 4 / 1e6

    # Tokenize + embed (same tokenizer as the prep step, no re-decoding here)
    tok = load_tokenizer(ckpt.get("tokenizer"))
    batch = tok([it["text"] for it in items], padding=True, truncation=True,
                max_length=128, return_tensors="pt")

    with torch.no_grad():
        student_embs = model(batch["input_ids"], batch["attention_mask"])

    teacher_embs = torch.tensor([it["teacher_emb"] for it in items])

    # Deterministic pair sample (matches JS Mulberry32 algorithm with same seed)
    rng = mulberry32(args.seed)
    seen, pairs = set(), []
    max_pairs = N * (N - 1) // 2
    target = min(args.pairs, max_pairs)
    while len(pairs) < target:
        i = int(rng() * N)
        j = int(rng() * N)
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))

    student_cos = [cosine(student_embs[i], student_embs[j]) for i, j in pairs]
    teacher_cos = [cosine(teacher_embs[i], teacher_embs[j]) for i, j in pairs]

    rho = spearman(student_cos, teacher_cos)
    r   = pearson(student_cos, teacher_cos)

    print()
    print(f"  label:        {args.label}")
    ckpt_disp = Path(args.ckpt).resolve()
    try:
        ckpt_disp = ckpt_disp.relative_to(ROOT)
    except ValueError:
        pass
    print(f"  ckpt:         {ckpt_disp}")
    print(f"  corpus:       {args.corpus}  ({N} sentences)")
    print(f"  pairs:        {len(pairs)}  (seed={args.seed})")
    print(f"  params:       {n_params:,}  ({fp32_size_mb:.1f} MB fp32)")
    print()
    print(f"  spearman(student vs teacher):  {rho:.4f}")
    print(f"  pearson (student vs teacher):  {r:.4f}")
    print()


if __name__ == "__main__":
    main()
