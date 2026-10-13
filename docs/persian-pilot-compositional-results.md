# Compositional pilot: diagnosis and measured outcome

Actual WASM evaluation on 2026-10-13. Pure cosine on full corpora; no aliases, calculator, language routing, query rewriting or reranker.
**Release ready: false.** Existing examples and the comparison page remain unchanged; failed candidates are diagnostic only.

## Root cause before training

See the [factorial diagnosis and frozen protocol](persian-pilot-compositional.md). Familiar numeric pair accuracy exaggerated arithmetic ability: current both-direction wins were 47/96, falling to 14/96 with adjacent answers and 12/96 with only the wording changed. Negation fell from 144/144 familiar probes to 57/144 with new wording, while noun substitution stayed 144/144. These repeated template probes are not independent natural-language observations.
No unknown tokens or truncation occurred. fp32 embeddings did not restore generalization. English-safe weight restoration removed some numeric gains, but the raw checkpoint already failed new wording. The evidence identifies lexical/template shortcuts and an underconstrained two-choice objective; it does not prove that general arithmetic is impossible for this architecture.

## Final comparison

| Fresh metric | Current | Binary control | Revised objective |
|---|---:|---:|---:|
| docs full-corpus Top-1 | 7/8 | 7/8 | 8/8 |
| docs NDCG@10 | 0.9539 | 0.9539 | 1.0000 |
| emoji full-corpus Top-1 | 5/8 | 4/8 | 5/8 |
| emoji NDCG@10 | 0.7577 | 0.7116 | 0.7577 |
| fa_negation full-corpus Top-1 | 4/16 | 4/16 | 4/16 |
| fa_negation NDCG@10 | 0.5995 | 0.6182 | 0.6078 |
| fa_negation strict contrast | 7/16 | 8/16 | 8/16 |
| fa_negation both_contrasts_correct | 0/8 | 0/8 | 0/8 |
| fa_negation both_top1_correct | 0/8 | 0/8 | 0/8 |
| fa_numbers full-corpus Top-1 | 7/16 | 5/16 | 8/16 |
| fa_numbers NDCG@10 | 0.7393 | 0.6850 | 0.7705 |
| fa_numbers strict contrast | 9/16 | 7/16 | 9/16 |
| fa_numbers both_contrasts_correct | 2/8 | 0/8 | 1/8 |
| fa_numbers both_top1_correct | 1/8 | 0/8 | 1/8 |
| en_negation full-corpus Top-1 | 2/8 | 2/8 | 2/8 |
| en_negation NDCG@10 | 0.6436 | 0.6436 | 0.6436 |
| en_negation strict contrast | 3/8 | 3/8 | 3/8 |
| en_negation both_contrasts_correct | 0/4 | 0/4 | 0/4 |
| en_negation both_top1_correct | 0/4 | 0/4 | 0/4 |
| en_numbers full-corpus Top-1 | 3/8 | 3/8 | 3/8 |
| en_numbers NDCG@10 | 0.7366 | 0.7366 | 0.7366 |
| en_numbers strict contrast | 4/8 | 4/8 | 4/8 |
| en_numbers both_contrasts_correct | 0/4 | 0/4 | 0/4 |
| en_numbers both_top1_correct | 0/4 | 0/4 | 0/4 |

## Preservation

| Metric | Current | Binary control | Revised objective |
|---|---:|---:|---:|
| External en STS | 0.769946 | 0.769946 | 0.769946 |
| External fa STS | 0.635174 | 0.632080 | 0.631665 |
| docs NDCG@10 | 0.8660 | 0.8424 | 0.8447 |
| emoji NDCG@10 | 0.8067 | 0.8404 | 0.8296 |
| native_qa NDCG@10 | 0.4589 | 0.4559 | 0.4575 |
| unseen_corpus NDCG@10 | 0.7756 | 0.7448 | 0.7887 |
| previous_docs NDCG@10 | 0.7579 | 0.7996 | 0.8150 |
| previous_emoji NDCG@10 | 0.7317 | 0.7548 | 0.7097 |
| previous_odyssey NDCG@10 | 0.6594 | 0.7053 | 0.6901 |
| reasoning_docs NDCG@10 | 0.6516 | 0.5999 | 0.5999 |
| reasoning_emoji NDCG@10 | 0.7571 | 0.7476 | 0.7565 |
| reasoning_fa_negation NDCG@10 | 0.7540 | 0.7847 | 0.7644 |
| reasoning_fa_numbers NDCG@10 | 0.7738 | 0.7680 | 0.7847 |
| round2_fa NDCG@10 | 0.7603 | 0.7538 | 0.7582 |
| round2_en NDCG@10 | 0.8453 | 0.8453 | 0.8453 |
| round3_fa NDCG@10 | 0.7449 | 0.8299 | 0.8102 |
| round3_en NDCG@10 | 0.7825 | 0.7825 | 0.7825 |
| round4_fa NDCG@10 | 0.7673 | 0.7927 | 0.7564 |
| round4_en NDCG@10 | 0.8452 | 0.8452 | 0.8452 |
| wasm_bytes | 8,004,767 | 8,004,767 | 8,004,767 |
| gzip_bytes | 5,347,162 | 5,347,217 | 5,347,212 |

Packed weights: **5,165,680 bytes**; 9,901,952 parameters and unchanged 32k tokenizer.
control: 2,533 English embeddings bit-exact to current: **True**; maximum absolute difference 0.0.
candidate: 2,533 English embeddings bit-exact to current: **True**; maximum absolute difference 0.0.

Known user-query regression results (not fresh evidence):
- `خیلی دوستت دارم` → 😍, 🥰, ❤️; pass: True.
- `بهترینِ تمام دوران` → 🏆, 😌, 👏; pass: True.
- `حیوان شاخ‌دار روی صخره علف می‌خورد` → 🐐, 🌱, 🙄; pass: True.

## Validation and objective ablation

control: selected epoch 3; 4 × 250 steps, 312.6 seconds of epoch runtime. Passing validation gates: 4/6.
- negation: both-view exact 0.406 → 0.444; full validation-corpus Top-1 0.044 → 0.041.
- numbers: both-view exact 0.106 → 0.106; full validation-corpus Top-1 0.006 → 0.019.
candidate: selected epoch 1; 4 × 250 steps, 322.4 seconds of epoch runtime. Passing validation gates: 2/6.
- negation: both-view exact 0.406 → 0.356; full validation-corpus Top-1 0.044 → 0.156.
- numbers: both-view exact 0.106 → 0.131; full validation-corpus Top-1 0.006 → 0.013.

Post-training replay of the same controlled development probes:

| Probe | Binary control | Revised objective |
|---|---:|---:|
| numeric, seen, exact_both | 7/96 | 11/96 |
| numeric, wording, both | 11/96 | 4/96 |
| negation, wording, both | 63/144 | 51/144 |

Both arms use the same corrected data, initialization, frozen weights, seed, replay and step budget. The ablation changes the combined objective: full-batch negatives, adjacent-answer supervision and equivalent-query consistency. It does not separately identify the contribution of each term. Checkpoints are selected independently on validation, never on final scores.

## Gates

- validation: FAIL
- english_exact: PASS
- packed_size: PASS
- wasm_size: PASS
- gzip_size: PASS
- known_user_queries: PASS
- docs: FAIL
- emoji: PASS
- previous_docs: PASS
- previous_emoji: FAIL
- reasoning_docs: FAIL
- reasoning_emoji: FAIL
- fresh_docs: PASS
- fresh_emoji: PASS
- round2_en: PASS
- round2_fa: PASS
- round3_en: PASS
- round3_fa: PASS
- round4_en: PASS
- round4_fa: FAIL
- native_qa: PASS
- previous_odyssey: PASS
- unseen_corpus: PASS
- logic: PASS
- reasoning_fa_negation: PASS
- reasoning_fa_numbers: PASS
- sts_fa: FAIL
- fresh_fa_negation_retention: PASS
- fresh_fa_negation_functional: FAIL
- fresh_fa_numbers_retention: PASS
- fresh_fa_numbers_functional: FAIL
- fresh_en_negation_retention: PASS
- fresh_en_numbers_retention: PASS

## Fresh Persian failures

| Case | Current contrast / full Top-1 | Revised contrast / full Top-1 | Revised top answer |
|---|---:|---:|---|
| q-fa_negation-0-0 | 1/0 | 1/0 | ناهید قرارداد را امضا نکرده است. |
| q-fa_negation-0-1 | 0/0 | 0/0 | ناهید قرارداد را امضا نکرده است. |
| q-fa_negation-1-0 | 0/0 | 0/0 | تمام دریچه‌ها باز هستند. |
| q-fa_negation-2-0 | 0/0 | 0/0 | ناهید قرارداد را امضا نکرده است. |
| q-fa_negation-2-1 | 1/1 | 1/0 | سامانه در گذشته راه افتاده است. |
| q-fa_negation-3-0 | 0/0 | 0/0 | ناهید قرارداد را امضا نکرده است. |
| q-fa_negation-4-0 | 0/0 | 0/0 | ممنوعیت ورود به تونل رفع شده است. |
| q-fa_negation-5-0 | 1/0 | 1/0 | ناهید قرارداد را امضا کرده است. |
| q-fa_negation-5-1 | 0/0 | 0/0 | ناهید قرارداد را امضا کرده است. |
| q-fa_negation-6-0 | 0/0 | 0/0 | خروج مجاز نیست. |
| q-fa_negation-6-1 | 1/0 | 1/0 | قطع‌شدن انتقال داده منتفی نیست. |
| q-fa_negation-7-0 | 0/0 | 0/0 | سامانه تا کنون راه نیفتاده است. |
| q-fa_numbers-0-0 | 1/1 | 0/0 | موجودی برگه‌ها ۴۶ عدد است. |
| q-fa_numbers-1-1 | 1/0 | 0/0 | بهای قابل پرداخت ۲۱۰ تومان است. |
| q-fa_numbers-2-0 | 1/1 | 0/0 | تعداد قطعه‌های موردنظر ۳۶ است. |
| q-fa_numbers-3-1 | 0/0 | 0/0 | مدت حضور ۷۷ دقیقه است. |
| q-fa_numbers-4-0 | 1/0 | 0/0 | مدت حضور ۱۷ دقیقه است. |
| q-fa_numbers-4-1 | 0/0 | 1/0 | مدت حضور ۱۷ دقیقه است. |
| q-fa_numbers-5-1 | 0/0 | 0/0 | نمونه شرط پذیرش را ندارد. |
| q-fa_numbers-7-0 | 0/0 | 0/0 | شمار بسته‌های قابل ارسال ۴۸۵ است. |

## Limits and provenance

The final audit checked 112 query/new-document texts against 580,731 training/development text instances: no normalized full-text collision; 3 shared eight-word spans, disclosed below. Known docs/emoji corpus documents are intentionally supervised. Teacher pretraining exposure is unknown. The 64 authored questions are small, related, and not a representative Persian benchmark.
- Shared span: `started at 16 48 and ended at 17`
- Shared span: `started at 16 48 and ended at 18`
- Shared span: `و حالا منفی چهارده را نشان می دهد`

The preliminary invalid-validation run was interrupted and excluded; it is preserved separately. Tests now reject identical validation views and a one-class boundary set. No fresh-model scores were used to repair that data.
The final fixture is now consumed regression evidence. Future model selection needs a new final test. The frozen-backbone design guarantees retention only for texts using unchanged embedding rows; mixed-language English containing changed Persian tokens may differ.
Source/data/checkpoint hashes, all per-query ranks, model sizes, logs and English comparisons are retained in `dist/pilot-fa-en-compositional/`. Training and evaluation source is versioned; generated checkpoint and WASM files remain local. Default examples and the existing comparison page are unchanged. No model was published.

## Interpretation and checks

The revised objective beats the matched binary control on this final retrieval set, but does not establish logical generalization. It retrieves 8/8 new docs versus 7/8 for current/control and keeps new emoji at 5/8 versus 5/8 current and 4/8 control. Persian negation remains 0/8 complete counterfactual pairs; numeric full-corpus pairs remain 1/8, equal to current. Historical docs/emoji and external Persian STS also have failed preservation gates. Do not promote this checkpoint.
This is not solely a Persian coverage problem: the unchanged English baseline solves zero complete pairs in both new English negation and numeric sets (0/4 each). Preserving English embeddings preserves their existing reasoning weaknesses as well as their retrieval quality. The data/objective fixes remove exploitable supervision patterns, but embedding-row-only adaptation has not taught the missing computation. These runs do not establish an impossibility result for the fixed-size architecture.
A subsequent experiment should target the shared reasoning representation with bilingual compositional supervision while constraining English/domain ranking, rather than another embedding-only run or a post-training weight transplant. Keep the same parameter budget and a new final test; any candidate still has to pass the existing retention gates. This is the next hypothesis, not an already demonstrated solution.
44 Python tests pass, including truth-table labels, exact arithmetic labels, nearest-answer negatives, semantic duplicate masking, distinct validation views and balanced boundary labels. Source/packed int4 parity is exact on 32 samples for each selected arm. Candidate WASM parity passes 136 tokenization/vector cases (maximum absolute error 0.000349149, minimum cosine 0.999998072). Optional wasm-opt was unavailable, as with the baseline. The UI was not changed or re-tested.

## Reproduction in the existing local pilot workspace

The commands intentionally refuse to overwrite prepared data or runs. The earlier checkpoints and caches must already exist, as documented in preceding pilot reports. Results and final tests from this run are now consumed.
```bash
.venv/bin/python training/distill/diagnose_reasoning.py
PYTHONPATH=training/distill .venv/bin/python -m prep.compositional
.venv/bin/python training/distill/compositional.py
.venv/bin/python training/distill/compositional.py --control
# Pack/build each selected.pt with the existing int4 pack and build-engine commands.
.venv/bin/python eval/quality/domain.py --model candidate=dist/pilot-fa-en-compositional/d256/tern_engine.js --output dist/pilot-fa-en-compositional --challenge eval/quality/corpus/persian-compositional-hard.json
```
