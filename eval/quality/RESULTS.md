> **Ternava historical baseline — original upstream English model.** These are not bilingual Ternava release results; see [the current pilot](../../README.md#persian--english-pilot).

# Quality Results — Spearman vs Teacher

**Task.** Given an N-sentence corpus, embed each sentence with both the teacher
(`sentence-transformers/all-MiniLM-L6-v2`) and the candidate model, generate M
deterministic random pairs `(i, j)`, and compute Spearman rank correlation
between teacher-side and candidate-side cosines across the M pair scores.

A Spearman of **1.0** = the candidate ranks pair similarities identically to
the teacher. A drop measures how much of the teacher's pair-ranking signal
was lost to distillation + quantization + packing.

**Setup.** 100 held-out MS MARCO queries (`spearman_smoke_1k.json`), 1000 pairs,
seed=42, commit `a74aa7b`.

---

## Headline numbers

| Variant                       | Bin size | Bits / param | Spearman | Pearson |
| ----------------------------- | -------: | -----------: | -------: | ------: |
| MiniLM-L6 teacher             |  90.9 MB |        32.00 |    1.000 |   1.000 |
| Student fp32 (pre-QAT)        |  38.0 MB |        32.00 |    0.883 |   0.907 |
| upstream English `emb_int8`          |   8.3 MB |         7.37 |    0.841 |   0.872 |
| **upstream English `emb_int4`** ⭐   |   4.6 MB |         4.08 |    0.835 |   0.864 |
| upstream English `emb_ternary`       |   2.9 MB |         2.43 |    0.710 |   0.756 |

**⭐ Ship target.** `emb_int4` was promoted to the default ship build after
this measurement — it sits within **0.006 Spearman of int8 at ~half the bin
size**, while the next step down (ternary) loses 0.13 Spearman. The cliff
between int4 and ternary made int4 the natural sweet spot.

**Story in one line.** From the fp32 teacher down to the shipped int4 build:
a **20×** reduction in bin size costs **0.165** Spearman.

---

## Compression vs fp32 student (same architecture, varying quantization)

This is the apples-to-apples view — same parameter count, same training, only
the storage precision changes.

| Variant                       | Bin size | Compression | Spearman | Retained vs fp32 |
| ----------------------------- | -------: | ----------: | -------: | ---------------: |
| Student fp32 (pre-QAT)        |  38.0 MB |         1×  |    0.883 |              100% |
| upstream English `emb_int8`          |   8.3 MB |        4.5× |    0.841 |               95% |
| **upstream English `emb_int4`** ⭐   |   4.6 MB |        8.2× |    0.835 |               95% |
| upstream English `emb_ternary`       |   2.9 MB |       13.2× |    0.710 |               80% |

**Read.** Quality holds almost flat from fp32 → int8 → int4 (each near 95%
of fp32's pair-ranking signal), then drops sharply at ternary embedding.
The cliff is at ~4 bits-per-element — going below that loses meaningful
distinguishability in the embedding lookup. Linear weights stay ternary in
all three upstream English variants; this curve is purely about embedding precision.

---

## Method notes

- **Where each number comes from.**
  - Teacher: 1.0 by construction (the teacher is what we compute Spearman *against*).
  - Student fp32: Python eval via `eval/quality/eval_python.py`, loads `model_state` into a vanilla `StudentEncoder`.
  - upstream English variants: WASM eval via `eval/quality/spearman.js` against the built `engine/pkg/`. Each variant required rebuilding the engine with the matching feature flag (`emb_int8` / `emb_int4` / `emb_ternary`) and swapping the `.bin`.
- **Why this is a distillation-fidelity metric, not a quality-vs-humans metric.** The reference signal is the teacher's pair-cosine ranking, not human STS labels. STS-B-vs-humans is a separate eval and would go in a different table.
- **Bin size is model-only.** `wasm-bindgen` glue + tokenizer.json are not included on the X-axis. Add them once at the writeup level (≈ +11 MB wasm + 0.7 MB tokenizer) when describing the user-shipped artifact.
- **n_params for upstream English.** Counted from the fp32 student state dict; the packed bins encode the same weights at lower precision. Bits-per-param = `bin_size_bytes × 8 / n_params`.
- **Reproducibility.** All data points are deterministic (seed=42). Re-running any row should produce the exact same Spearman to 4 decimals.

## Raw data

Full numbers (including bytes, params, ckpt paths) live in
[../results/quality.json](../results/quality.json).
