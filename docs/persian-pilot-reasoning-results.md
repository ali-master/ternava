# Paired reasoning pilot — measured results

Actual packed WASM on 2026-10-12. Pure cosine, no aliases, calculator, query rewrite or reranker.
Release ready: **false**. Selected epoch 6 using validation only.

| Fresh metric | Round 3 | Domain baseline | Reasoning candidate |
|---|---:|---:|---:|
| fresh_docs Top-1 | 2/8 | 3/8 | 3/8 |
| fresh_docs NDCG@10 | 0.4308 | 0.6404 | 0.6516 |
| fresh_emoji Top-1 | 4/8 | 4/8 | 4/8 |
| fresh_emoji NDCG@10 | 0.6016 | 0.7662 | 0.7571 |
| fresh_fa_negation Top-1 | 6/12 | 7/12 | 4/12 |
| fresh_fa_negation NDCG@10 | 0.7821 | 0.8353 | 0.7540 |
| fresh_fa_negation strict contrast | 7/12 | 9/12 | 6/12 |
| fresh_fa_negation both directions | 2/6 | 4/6 | 2/6 |
| fresh_fa_numbers Top-1 | 6/12 | 4/12 | 5/12 |
| fresh_fa_numbers NDCG@10 | 0.7936 | 0.7321 | 0.7738 |
| fresh_fa_numbers strict contrast | 6/12 | 6/12 | 7/12 |
| fresh_fa_numbers both directions | 0/6 | 0/6 | 1/6 |
| fresh_en_negation Top-1 | 7/12 | 7/12 | 7/12 |
| fresh_en_negation NDCG@10 | 0.7791 | 0.7767 | 0.7791 |
| fresh_en_negation strict contrast | 8/12 | 8/12 | 8/12 |
| fresh_en_negation both directions | 2/6 | 2/6 | 2/6 |
| fresh_en_numbers Top-1 | 6/12 | 6/12 | 6/12 |
| fresh_en_numbers NDCG@10 | 0.7988 | 0.7988 | 0.7988 |
| fresh_en_numbers strict contrast | 6/12 | 6/12 | 6/12 |
| fresh_en_numbers both directions | 0/6 | 0/6 | 0/6 |

## Retention and size

| Metric | Round 3 | Domain baseline | Candidate |
|---|---:|---:|---:|
| External en STS Spearman | 0.769946 | 0.770004 | 0.769946 |
| External fa STS Spearman | 0.638079 | 0.636103 | 0.635174 |
| docs NDCG@10 | 0.5771 | 0.8324 | 0.8660 |
| emoji NDCG@10 | 0.5674 | 0.7872 | 0.8067 |
| unseen_corpus NDCG@10 | 0.8147 | 0.7458 | 0.7756 |
| logic NDCG@10 | 0.6692 | 0.7044 | 0.7210 |
| native_qa NDCG@10 | 0.4569 | 0.4625 | 0.4589 |
| previous_docs NDCG@10 | 0.3807 | 0.7203 | 0.7579 |
| previous_emoji NDCG@10 | 0.4866 | 0.6995 | 0.7317 |
| previous_odyssey NDCG@10 | 0.7126 | 0.7455 | 0.6594 |
| round2_fa NDCG@10 | 0.8565 | 0.7792 | 0.7603 |
| round2_en NDCG@10 | 0.8453 | 0.8248 | 0.8453 |
| round3_fa NDCG@10 | 0.7375 | 0.7636 | 0.7449 |
| round3_en NDCG@10 | 0.7825 | 0.7877 | 0.7825 |
| round4_fa NDCG@10 | 0.7633 | 0.7895 | 0.7673 |
| round4_en NDCG@10 | 0.8452 | 0.8568 | 0.8452 |
| wasm_bytes | 8,004,767 | 8,004,767 | 8,004,767 |
| gzip_bytes | 5,347,369 | 5,346,935 | 5,347,162 |

Packed weights: **5,165,680 bytes**; 9,901,952 parameters; unchanged 32k tokenizer.

## Gates

- validation_eligible: FAIL
- docs_retention: PASS
- fresh_docs: PASS
- emoji_retention: PASS
- fresh_emoji: PASS
- fresh_fa_negation_nonregression: FAIL
- fresh_fa_negation_competence: FAIL
- fresh_fa_numbers_nonregression: PASS
- fresh_fa_numbers_competence: FAIL
- fresh_en_negation_nonregression: PASS
- fresh_en_numbers_nonregression: PASS
- round2_en_recovered: PASS
- round2_fa_retained: FAIL
- round3_en_recovered: PASS
- round3_fa_retained: FAIL
- round4_en_recovered: PASS
- round4_fa_retained: FAIL
- unseen_corpus_retention: PASS
- native_qa_retention: FAIL
- previous_docs_retention: PASS
- previous_emoji_retention: PASS
- previous_odyssey_retention: FAIL
- sts_en: PASS
- sts_fa: PASS
- raw_size_fixed: PASS
- wasm_size_fixed: PASS
- wire_size: PASS

## Evidence and limits

All six epochs completed: 837.4 seconds of epoch runtime. The [fixed protocol](persian-pilot-reasoning.md) records the recipe, source checkpoints and selection rules.
The overlap audit checked 549,799 text instances against 112 fresh query/document texts: no exact or eight-word matches. Teacher pretraining exposure is unknown.
The 64 authored questions are small and correlated. Docs/emoji use known corpora; their questions are new. English logic includes related translations. Numeric success on boundary/direction questions is not evidence of arithmetic; see pair-level results below.
Historical domain and earlier round challenges are now development/regression data. Bootstrap intervals are descriptive and do not cover author/topic bias. Raw per-query ranks, strict margins, hashes and confidence intervals are in `dist/pilot-fa-en-reasoning/`.

## Fresh Persian pair details

| Category/pair | Domain correct contrasts | Candidate correct contrasts | Candidate full-corpus Top-1 |
|---|---:|---:|---:|
| negation/1 | 0/2 | 0/2 | 0/2 |
| negation/2 | 2/2 | 2/2 | 1/2 |
| negation/3 | 2/2 | 1/2 | 1/2 |
| negation/4 | 2/2 | 0/2 | 0/2 |
| negation/5 | 1/2 | 1/2 | 1/2 |
| negation/6 | 2/2 | 2/2 | 1/2 |
| numbers/1 | 1/2 | 1/2 | 1/2 |
| numbers/2 | 1/2 | 2/2 | 1/2 |
| numbers/3 | 1/2 | 1/2 | 1/2 |
| numbers/4 | 1/2 | 1/2 | 1/2 |
| numbers/5 | 1/2 | 1/2 | 0/2 |
| numbers/6 | 1/2 | 1/2 | 1/2 |

## Interpretation and implementation checks

The selected artifact is the English-safe transplant of epoch six with alpha 1: all newly trained Persian-script rows are retained, while shared and non-Persian weights return to round three. All six raw epochs and all three calibration candidates fail at least one eligibility gate. Nothing is promoted to the default examples.
Fresh docs and emoji Top-1 are unchanged from the domain baseline (3/8 and 4/8). Previous-domain regression Top-1 improves from 15/20 to 16/20 and 12/20 to 13/20; these reused cases are not new generalization evidence.
Fresh Persian negation deteriorates from 9/12 to 6/12 strict contrasts and from 4/6 to 2/6 correct pairs. Full-corpus Top-1 falls from 7/12 to 4/12. The model reverses both sides of the ban-revocation and neither/nor counterfactuals. This is a failed reasoning candidate.
Numeric strict contrasts rise from 6/12 to 7/12. The percentage pair is the one newly correct contrast pair, but the discount query retrieves a shelf-count answer ahead of the correct fare. Both-directions full-corpus Top-1 remains 0/6. Stock, fractions, elapsed time and strict bounds retain fixed-answer shortcuts.
The development diagnostic separates memorization from transfer: on 32 seen two-operation Persian pairs, both-direction successes rise from 3 to 18; on 20 validation pairs with different wording and ranges, both checkpoints score 0. This supports a generalization failure; it does not prove a hard architectural impossibility. The same split changes wording and numbers together, so their separate effects are not identified.
Actual WASM English embeddings are bit-identical to round three for 2,533 external sentences (maximum absolute difference 0). The three historical English ranking scores also exactly match round three. Domain-era gains on English rounds three and four are rolled back, rather than preserved; all scores are listed above.
37 Python tests passed. Source versus packed int4 inference matched exactly on 32 samples. Actual WASM parity passed 136 cases: maximum absolute difference 0.000406154, minimum cosine 0.999996419. Browser verification found no console errors. These are implementation checks, not evidence of semantic correctness.
No architecture or tokenizer change; packed weights and WASM are byte-count identical to the prior domain model. Gzip grows by 227 bytes (about 0.0042%). Optional wasm-opt optimization was unavailable, consistently with the compared builds.

README screenshots show a real, known successful Persian query, not a cherry-picked fresh success presented as aggregate evidence. The quality chart is generated directly from the saved evaluation reports. No weights were published.
