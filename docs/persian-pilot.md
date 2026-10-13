# Persian + English mini pilot

This pilot keeps the 2-layer d256/f1024 mini encoder, 384-dimensional normalized
output, ternary BitLinear weights and int4 token embeddings. It trains a **new
32,000-token WordPiece vocabulary** and student from scratch. No old checkpoint
or existing vector index is compatible by assumption. Positional encoding is
unchanged; word-order invariance remains a known limitation of this experiment.

## Setup

Run every command below from the repository root. Python 3.12 is recommended.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r training/distill/requirements.txt
cargo install wasm-pack --locked
rustup target add wasm32-unknown-unknown
```

The teacher is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
its 384-dimensional vectors are cached during preparation. It is never shipped
in WASM. Preparation downloads data and teacher weights; allow disk space and
network time. Training uses CUDA, then MPS, then CPU, in that priority order.
W&B can be disabled as shown below; no W&B account is needed.

## Prepare

```sh
.venv/bin/python training/distill/prep/bilingual.py \
  --config training/distill/configs/pilot-fa-en-32k.yaml --tokenizer-only

.venv/bin/python training/distill/prep/bilingual.py \
  --config training/distill/configs/pilot-fa-en-32k.yaml
```

The first command downloads OPUS-100 `en-fa`, samples 100,000 usable pairs,
deduplicates normalized endpoints, splits by pair, then learns exactly 32,000
tokens from **train only**. An undersized vocabulary is an error, not silently
padded with unused tokens. The second command reuses the saved pairs/tokenizer,
downloads the teacher and encodes both languages independently. The multilingual
teacher supplies the shared space; targets from unrelated teachers are not mixed.

There are approximately 180,000 training sentences, 10,000 validation sentences,
and 10,000 test sentences (balanced English/Persian). A maximum of 1,000 held-out
pairs is exported for the WASM comparison. This is a translation-domain pilot;
it does not cover all Persian domains or informal language.

For local, curated input, pass `--pairs-jsonl path/to/pairs.jsonl` on the first
command. Each line must contain string fields `english` and `non_english`.
Corpus shortfalls fail explicitly. The actual source revision (or local SHA256),
pair-set hash, teacher revision and tokenizer hash are recorded. When resuming
tokenizer-only preparation, the saved plan is used, not newly downloaded rows.
Use a **new output_dir and matching train.cache_dir** for another recipe; a
completed cache will not be overwritten.

Normalization is serialized in `tokenizer.json`, so Python and Rust share it:
NFC, Arabic yeh/kaf → Persian forms, ZWNJ → space, BERT cleaning/lowercasing with
accents retained. Digits and negation are preserved. `[PAD]=0`; the vocabulary
contains the Persian alphabet and continuation pieces. Unknown arbitrary scripts
or emoji are still possible. Both runtimes truncate to 128 including CLS/SEP.

## Train and pack

```sh
WANDB_MODE=disabled .venv/bin/python training/distill/train.py \
  --config training/distill/configs/pilot-fa-en-32k.yaml

CKPT="training/distill/runs/mini-fa-en-32k-pilot-$(git rev-parse --short=7 HEAD)/checkpoint_ep10.pt"
.venv/bin/python training/pack/verify.py --ckpt "$CKPT" \
  --embedding-format int4 --cache-dir training/distill/cache/mini-fa-en-32k \
  --cache-name bilingual
.venv/bin/python training/pack/pack.py --ckpt "$CKPT" \
  --embedding-format int4 --output training/pack/out/pilot-fa-en/model.bin
```

Ten epochs, two warmup epochs, cosine distillation plus relational weight 0.1
are starting hyperparameters, not established optimums. `contrastive_w=0`
disables the old all-pairs repulsion, which would separate translations.
Training snapshots the tokenizer beside checkpoints. New-vocabulary warm-starts
from the English model are rejected, including equal-sized vocabularies with
different IDs. Checkpoints without tokenizer metadata retain the legacy path.

For each changed recipe choose a new `run_name`. Resume only with the same data,
tokenizer, architecture and scheduler horizon. Seed 42 controls sampling and
initialization; exact GPU/MPS reproducibility across hardware is not promised.

The packer writes `model.bin`, `model.tokenizer.json` and `model.bin.json`.
The sidecar links weights to their exact tokenizer and data. Keep these together.
Expected raw model size for this architecture is **5,034,608 bytes**, versus
4,839,512 for the 30,522-token mini: +195,096 bytes. This is not gzip or RAM size.

## Build and compare the actual artifact

```sh
BIN="$PWD/training/pack/out/pilot-fa-en/model.bin" \
TOKENIZER="$PWD/training/pack/out/pilot-fa-en/model.tokenizer.json" \
PKG=packages/mini TARGET=web WEB_OUT=dist/pilot-fa-en \
  bash scripts/build-engine.sh

.venv/bin/python eval/quality/bilingual.py \
  --evaluation training/distill/cache/mini-fa-en-32k/evaluation.json \
  --candidate dist/pilot-fa-en/d256/tern_engine.js \
  --candidate-model training/pack/out/pilot-fa-en/model.bin \
  --retrieval eval/quality/corpus/bilingual-retrieval-smoke.json \
  --output dist/pilot-fa-en/comparison.json
```

The baseline defaults to the committed `examples/docs-search/wasm` mini. Supply
`--baseline /path/to/web/tern_engine.js` to compare another release. Both engines
run the **same texts in Node**, using web-target glue and embedded WASM. They do
not fetch an inference API. Building the pilot does not overwrite the English
tokenizer or published package directories. Build-time checks reject wrong
tokenizer IDs/hashes, altered model checksums and unsupported sequence lengths.

The JSON report contains:

- English/Persian Spearman against the same multilingual teacher (fidelity,
  **not** human-labelled similarity).
- English→Persian and Persian→English bitext Recall@1/10 and NDCG@10.
- Per-language unknown-token fraction, token-length p50/p95 and truncation rate.
- Monolingual retrieval NDCG@10 and Recall@1/10 when qrels are provided.
- Actual WASM/gzip bytes, hashes, warmed latency p50/p95, cold initialization
  time, and allocated WASM linear memory (not process peak RSS).
- Candidate tokenizer parity over every evaluation input, and checks that the
  WASM embeds exactly the packed model and tokenizer being evaluated.

The committed handwritten FAQ set is labelled `scope: smoke`. It exercises the
comparison pipeline; eight queries per language cannot certify quality. For a
real decision, supply held-out FaMTEB/domain retrieval data in the same format:

```json
{
  "scope": "benchmark",
  "corpus": [{"id": "doc-1", "lang": "fa", "text": "..."}],
  "queries": [{"id": "q-1", "lang": "fa", "text": "...", "relevant": {"doc-1": 1}}]
}
```

Include **both** languages, unique document IDs and positive numeric relevance
grades. Keep benchmark documents/queries out of training; this external-data
independence must be checked when assembling the benchmark. Translation
retrieval is not a substitute for monolingual search or human-labelled STS.

Initial comparison checks: gzip growth ≤10%, English retrieval ≥97% of baseline,
and Persian retrieval improvement. These are pilot diagnostics, not automatic
release authorization. `release_ready` stays false. Review the held-out benchmark,
normalization/negation/word-order cases and device performance before release.
Retrain embeddings for any downstream vector index when changing model versions.

## Hard held-out and external STS comparison

`eval/quality/corpus/bilingual-hard-heldout.json` was frozen before the local
training run. It contains 24 queries and 48 documents per language, with idioms,
negation, numerical changes, swapped roles, technical paraphrases and sarcasm.
It is an authored challenge set, not a representative public benchmark. The
optional `contrast` document records a hard negative; per-query ranks, margins
and category results are included in the comparison report. Near-zero margins
(absolute value ≤1e-6) do not count as successful contrast discrimination.

All scoring uses embeddings and cosine only. The emoji demo's aliases and phrase
matching are never imported. The role-order cases intentionally expose the
architecture's lack of positional encoding. Do not select checkpoints or change
the recipe using these test results.

```sh
.venv/bin/python eval/quality/audit-heldout.py \
  --fixture eval/quality/corpus/bilingual-hard-heldout.json \
  --pairs training/distill/cache/mini-fa-en-32k/pairs.json \
  --output dist/pilot-fa-en/heldout-audit.json

.venv/bin/python eval/quality/bilingual.py \
  --evaluation training/distill/cache/mini-fa-en-32k/evaluation.json \
  --candidate dist/pilot-fa-en/d256/tern_engine.js \
  --candidate-model training/pack/out/pilot-fa-en/model.bin \
  --retrieval eval/quality/corpus/bilingual-hard-heldout.json \
  --output dist/pilot-fa-en/comparison-mini.json
```

Repeat the comparison with `--baseline examples/emoji-search/wasm/tern_engine.js`
and a separate output file to include the existing base model. The overlap audit
checks normalized exact matches and contiguous eight-word spans against the
pilot's train and validation splits. It cannot prove semantic independence or
absence from teacher/baseline pretraining.

The external STS evaluator uses pinned dataset revisions and the complete test
splits of FarSick and STS Benchmark, excluding any pair with an endpoint matching
the pilot train/validation texts after normalization. Preserve the generated
dataset: preparation refuses to overwrite it. Omit the preparation command when
reusing an existing frozen dataset.

```sh
.venv/bin/python eval/quality/sts-heldout.py \
  --prepare-from-pairs training/distill/cache/mini-fa-en-32k/pairs.json \
  --dataset dist/pilot-fa-en/external-sts.json

.venv/bin/python eval/quality/sts-heldout.py \
  --dataset dist/pilot-fa-en/external-sts.json \
  --model mini=examples/docs-search/wasm/tern_engine.js \
  --model base=examples/emoji-search/wasm/tern_engine.js \
  --model candidate=dist/pilot-fa-en/d256/tern_engine.js \
  --output dist/pilot-fa-en/sts-comparison.json
```

STS correlations use dataset relevance scores, not teacher predictions. Reports
record dataset and model hashes. The trainer writes `metrics.jsonl` next to its
checkpoints even with W&B disabled; step progress is also printed to stdout.

## Offline tests and cross-runtime smoke

```sh
WANDB_MODE=disabled .venv/bin/python -m unittest discover -s training/tests -v
.venv/bin/python training/tests/make_engine_fixture.py --output dist/pilot-smoke
TERN_MODEL_PATH="$PWD/dist/pilot-smoke/model.bin" \
TERN_TOKENIZER_PATH="$PWD/dist/pilot-smoke/tokenizer.json" \
  cargo test -p tern-engine --features emb_int4 --lib
BIN="$PWD/dist/pilot-smoke/model.bin" \
TOKENIZER="$PWD/dist/pilot-smoke/model.tokenizer.json" \
PKG=packages/mini TARGET=web WEB_OUT=dist/pilot-smoke/web \
  bash scripts/build-engine.sh
node engine/tests/test_bilingual.mjs \
  dist/pilot-smoke/web/d256/tern_engine.js dist/pilot-smoke/parity.json
```

The smoke fixture has **random, untrained weights** and an artificial 32k
vocabulary. It checks packing size, tokenization and Python↔WASM numerical parity,
never Persian quality. Unit tests also exercise a tiny training/cache/checkpoint
round trip with synthetic teacher vectors and no model downloads.

Python pack/unpack uses max absolute error ≤0.0005. Cross-runtime WASM checks use
max absolute error ≤0.001 **and** cosine ≥0.9999, while token IDs must match exactly.
Float32 reduction order can move activations across int8 rounding boundaries:
on the local fixture, PyTorch alone changed by 0.00081 between padded and compact
input shapes. Rust's halfway rounding is also explicitly tested against
PyTorch's ties-to-even rule. These numerical checks do not replace quality tests.

## Local preparation result — 2026-10-12

The 100,000-pair OPUS preparation completed: 90,000 train pairs, 5,000 validation
pairs and 5,000 test pairs. Dataset revision:
`f9fff3fffa048f359508f6aa77ff5bd2226d4016`. The tokenizer trained only on the
180,000 training sentences has exactly 32,000 tokens; its SHA256 is
`dddc3fca56fcd694e8b2268d719b7f73827f8fd73d06aedc2e0304cdfc6f08fe`.

Measured on all 5,000 held-out sentences **per language**, with CLS/SEP included
in sequence lengths and excluded from unknown-token fractions:

| Metric | English baseline → bilingual | Persian baseline → bilingual |
| --- | --- | --- |
| Median tokens | 11 → 11 | 26 → 10 |
| 95th percentile tokens | 32 → 32 | 87 → 31 |
| Sentences exceeding 128 tokens | 0.04% → 0.04% | 2.60% → 0.48% |
| Unknown-token fraction after truncation | 0.0284% → 0.0033% | 0.6429% → 0% |

The local details are in
`training/distill/cache/mini-fa-en-32k/tokenizer-report.json` (ignored generated
data). This is a tokenizer coverage result, **not** a semantic-quality result.
Teacher caching and all 10 student-training epochs completed on October 12,
2026. The trained int4 model and actual WASM were evaluated without aliases;
see [measured pilot results](persian-pilot-results.md). Persian STS improved,
but English STS regressed and difficult semantic contrasts remain unreliable.
The measured gzip WASM is 5,225,124 bytes; random test-weight sizes are not used.

Sources: [multilingual distillation](https://sbert.net/examples/sentence_transformer/training/multilingual/README.html),
[teacher](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
[OPUS pairs](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opus-100),
[FaMTEB](https://arxiv.org/abs/2502.11571).
