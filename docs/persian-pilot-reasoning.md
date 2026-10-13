# Paired reasoning continuation

This experiment continues the technical/colloquial checkpoint, retaining the
32,000-token bilingual tokenizer, 9,901,952 parameters and int4/ternary export.
It changes training only. Inference remains one encoder and cosine similarity:
no aliases, arithmetic parser, query rewriting or reranking.

## Diagnosis and intervention

The preceding domain checkpoint answered zero of six fresh numeric pairs correctly
in both directions. A 50% individual contrast score therefore did not demonstrate
arithmetic. Reversed temperatures produced almost identical query vectors
(cosine 0.99939). Distinct token IDs rule out an input-token collision for those
cases. Previous domain training also froze digit-only embeddings and gave the
logic margin relatively little weight. These are plausible optimization obstacles,
not proof of a single causal explanation or a capacity guarantee.

The new loss penalizes both individual margins and the worse margin in each pair,
and aligns query differences with answer differences. All existing weights,
including numeric token rows, are trainable. Domain vectors are anchored to the
technical/colloquial checkpoint. English vectors and their similarity geometry
are anchored separately to round three, supplemented by English ranking replay.
This avoids treating unchanged STS as evidence of unchanged retrieval order.

## Fixed recipe

- Seed 719; six epochs of 300 steps on local MPS.
- Embedding LR 0.00012, backbone LR 0.000012; warmup 50 steps and linear decay.
- 32 counterfactual triplets per step, evenly split by language and negation/numbers.
- 20 domain queries against the complete 30-document or 70-emoji corpus.
- 32 English and 32 Persian replay texts, plus 24 English ranking triplets.
- Exact numeric labels generated in preprocessing only; train/validation use
  different wording and nouns. New arithmetic ranges differ where supported;
  inherited temperature/boundary/time generators share their numeric ranges.
- 6,214 training and 640 validation triplets after pairwise deduplication,
  token-collision checks and exact exclusion. A small template validation set is
  not evidence of generalization.

Selection uses validation only: average paired Persian logic success plus one
quarter of the two domain NDCGs. Eligibility additionally requires every historical
English ranking set to recover to round three, domain docs/emoji/unseen-corpus
Top-1 to remain at the source level, historical Persian ranking loss <=0.01,
English validation STS loss <=0.002, Persian <=0.005, and no loss in aggregate
historical Persian contrasts or new validation paired accuracy.

If no epoch is eligible, report the highest-scoring diagnostic checkpoint as a
failed candidate. Do not replace shipped/example models or silently relax gates.

## Frozen evaluation

`eval/quality/corpus/persian-reasoning-hard.json` contains 64 authored queries:
8 technical, 8 colloquial emoji, 12 Persian negation, 12 Persian numeric,
12 English negation and 12 English numeric. Each reasoning category includes
six opposite-query pairs. English logical translations are correlated evidence.
Docs/emoji use known corpora with new questions; logic also uses new documents.

SHA256: `610d16647241e0ef582cdbfcd3ba25fc38d10e1cc877000e3d2c9ca74def2350`.

The prior domain challenge is now explicitly development/regression data. Fresh
scores must not select a checkpoint. Final evaluation uses actual packed WASM,
full-corpus ranking, strict >1e-6 contrast margins, both-directions pair accuracy,
all older retrieval sets and external bilingual STS. Normalized exact and
8-word overlap audits cannot establish absence from teacher pretraining.

```bash
PYTHONPATH=training/distill .venv/bin/python training/distill/prep/reasoning.py
PYTHONPATH=training/distill .venv/bin/python training/distill/reasoning.py --smoke
PYTHONPATH=training/distill .venv/bin/python training/distill/reasoning.py
.venv/bin/python -m unittest discover -s training/tests
```

Preparation and training refuse to overwrite an existing experiment.

## Reproducing the README chart

The chart is produced from the saved WASM evaluation JSON, using an isolated
plotting environment so the training dependencies are unchanged:

```bash
.venv/bin/python eval/quality/reasoning-report.py
uv run --isolated --no-project --with matplotlib --python 3.12 python eval/quality/reasoning-report.py --plot-only
```

The preview screenshot is captured from the running local comparison page. It is
an illustrative query; the chart and per-query report show the evaluation scope.

## Adaptive English retention check (before fresh inference)

The first two epochs improved development paired accuracy but failed English
ranking retention. Before reading any fresh model results, a separate calibration
was specified: restore the complete shared backbone and non-Persian token rows
from round three; interpolate only Persian-script and Persian/Arabic-digit rows
between the domain and trained checkpoints at alpha 0, 0.5 and 1. This changes
no architecture, tokenizer, inference rule or parameter count. English inputs
whose token IDs do not touch those rows are unchanged by construction.

`training/distill/reasoning_calibrate.py` evaluates all three options on the same
development gates. Among eligible artifacts prefer the highest selection score.
If none is eligible, prefer an English-recovered diagnostic artifact, then the
highest selection score; still label it a failed candidate. This adaptive step
is disclosed separately from the original fixed training run. Fresh tests stay
sealed until the final checkpoint and hash are written to `selection.json`.
