# Round-three continuation: native Persian supervision

Local experiment, 2026-10-12. Start from `mini-fa-en-round3/best.pt`, not round four.
Keep the 32,000-token vocabulary, 9,901,952 parameters and the existing engine.

## Hypothesis and fixed recipe

Earlier training relied on translated/synthetic similarity and narrowly authored
contrasts. Add Persian QA retrieval and independently labeled paraphrase/NLI data.
Freeze the backbone and all tokens except tokens consisting of Arabic-script
letters (digits, punctuation, special tokens, Latin and mixed-letter tokens stay
fixed). Train-time int4 straight-through rounding uses the packer's per-row scale.
No adapter, language classifier, alias, calculator or reranker is exported.

One fixed run: seed 531, four epochs, 400 steps each, AdamW learning rate 0.0004,
50-step warmup then linear decay, zero weight decay, zero dropout. Batches:
QA 32, NLI 16, QQP 24, Persian replay 24. Loss weights: 1, .5, .1, .2.
Selection uses native validation NDCG + .1 QQP Spearman + .1 NLI contrast, subject
to Persian validation STS >=97% of baseline and unchanged English STS. Assert
bitwise equality of all frozen weights at every epoch. Test results never select
the checkpoint. Historical rounds 2–4 are explicitly regression sets, no longer
fresh evidence.

Validation-only adjustment recorded after epoch two, before any candidate test
inference: if all four epochs fail the STS gate, interpolate only the changed
Persian rows of the best diagnostic checkpoint with round three using fixed
coefficients .25, .5 and .75. Reapply the same validation score and STS gates.
Save a separate `calibrated.pt`; retain the failed original run. This is an
adaptive validation step, not part of the original preregistered recipe.

Promotion additionally requires native test NDCG to improve by .02, no application
top1 regression and at least 50% top1 in each of the three application sets,
all inherited Persian retrieval/contrast gates to hold, external Persian STS
>=97% of round three, English vectors unchanged, and gzip growth <=0.5%.
Otherwise the artifact remains experimental even if some metrics improve.

## Data and evaluation

- [PersianQA](https://github.com/sajjjadayobi/PersianQA), revision
  `ba4806eb7894b1e78e214e16e7e717840c129450`, repository license GPL-3.0.
- [ParsiNLU](https://github.com/persiannlp/parsinlu), revision
  `a3b08d0b93ca84e0017e2f490a2aa3ab36c79885`, CC-BY-NC-SA-4.0; QQP and
  merged Farstail/NLI contain mixed origins, so do not describe all supervision
  as native authored data.

Raw data and hashes live in the ignored local cache. Download pinned raw files,
never execute remote dataset scripts. Local research only; redistribution and
commercial suitability of resulting weights have not been cleared.

QA: official test held out, validation selected by article-title hash from train.
96-token windows with stride 64 are generated without consulting answer spans;
all windows are retrieval candidates. Labels identify windows containing valid
answer offsets. Unanswerable questions are excluded and counted; abstention is
not measured. Normalized exact endpoints are removed across train/evaluation
tasks. This cannot prove absence from ancestor teacher pretraining.

Fresh application fixture: 36 authored paraphrases across docs, emoji and Odyssey,
full actual corpora, no aliases. English external STS and all inherited hard
retrieval tests are retention checks. Evaluate final packed WASM, measure raw,
WASM and gzip bytes, and verify English vector equality with round three.

## Reproduce locally

```sh
PYTHONPATH=training/distill .venv/bin/python -m prep.native
.venv/bin/python -m unittest discover -s training/tests -p 'test_native.py'
.venv/bin/python training/distill/native.py --smoke
.venv/bin/python training/distill/native.py
# Only if the fixed run has no eligible checkpoint:
.venv/bin/python training/distill/native.py --calibrate
```

Preparation and training refuse to overwrite existing runs. Source data paths and
file hashes are recorded in the generated manifest. Preparation downloads missing
raw sources under `training/distill/cache/persian-native-sources` from pinned revisions.

The completed run uses `calibrated.pt`. Package and measure it with:

```sh
.venv/bin/python training/pack/pack.py \
  --ckpt training/distill/runs/mini-fa-en-round3-native/calibrated.pt \
  --embedding-format int4 --output training/pack/out/pilot-fa-en-v3-native/model.bin
BIN="$PWD/training/pack/out/pilot-fa-en-v3-native/model.bin" \
TOKENIZER="$PWD/training/pack/out/pilot-fa-en-v3-native/model.tokenizer.json" \
PKG=packages/mini TARGET=web WEB_OUT=dist/pilot-fa-en-v3-native bash scripts/build-engine.sh
.venv/bin/python eval/quality/native.py \
  --model round3=dist/pilot-fa-en-v3/d256/tern_engine.js \
  --model candidate=dist/pilot-fa-en-v3-native/d256/tern_engine.js
.venv/bin/python eval/quality/sts-heldout.py \
  --dataset dist/pilot-fa-en/external-sts.json \
  --model candidate=dist/pilot-fa-en-v3-native/d256/tern_engine.js \
  --output dist/pilot-fa-en-v3-native/sts-candidate.json
.venv/bin/python -m http.server 8877 --bind 127.0.0.1
```

Open `/eval/quality/native-preview.html` on that server. Evaluation refuses to
overwrite existing per-model results. See [measured results](persian-pilot-round3-native-results.md)
before considering any deployment.
