# Technical and colloquial Persian pilot — measured results

Local run, 2026-10-12. Actual packed WASM, pure cosine; no aliases, reranker or query rules.
Release ready: **false**. Selected `training/distill/runs/mini-fa-en-domain/diagnostic-best.pt` using validation only.

| Fresh metric | Round3 | Previous native | Domain candidate |
|---|---:|---:|---:|
| docs top1 (20) | 7/20 | 3/20 | 15/20 |
| docs NDCG@10 | 0.5771 | 0.4412 | 0.8324 |
| emoji top1 (20) | 6/20 | 6/20 | 12/20 |
| emoji NDCG@10 | 0.5674 | 0.5410 | 0.7872 |
| unseen_corpus top1 (12) | 8/12 | 5/12 | 6/12 |
| unseen_corpus NDCG@10 | 0.8147 | 0.6532 | 0.7458 |
| logic top1 (24) | 9/24 | 11/24 | 11/24 |
| logic NDCG@10 | 0.6692 | 0.7301 | 0.7044 |
| negation strict contrast | 7/12 | 8/12 | 6/12 |
| negation both directions correct | 1/6 | 2/6 | 1/6 |
| numbers strict contrast | 6/12 | 6/12 | 6/12 |
| numbers both directions correct | 0/6 | 0/6 | 0/6 |

## Retention

| Metric | Round3 | Previous native | Candidate |
|---|---:|---:|---:|
| External en STS Spearman | 0.769946 | 0.769946 | 0.770004 |
| External fa STS Spearman | 0.638079 | 0.630935 | 0.636103 |
| native_qa NDCG@10 | 0.4569 | 0.5475 | 0.4625 |
| previous_docs NDCG@10 | 0.3807 | 0.4120 | 0.7203 |
| previous_emoji NDCG@10 | 0.4866 | 0.5958 | 0.6995 |
| previous_odyssey NDCG@10 | 0.7126 | 0.6746 | 0.7455 |
| round2_fa NDCG@10 | 0.8565 | 0.7764 | 0.7792 |
| round2_en NDCG@10 | 0.8453 | 0.8453 | 0.8248 |
| round3_fa NDCG@10 | 0.7375 | 0.6887 | 0.7636 |
| round3_en NDCG@10 | 0.7825 | 0.7825 | 0.7877 |
| round4_fa NDCG@10 | 0.7633 | 0.7581 | 0.7895 |
| round4_en NDCG@10 | 0.8452 | 0.8452 | 0.8568 |

## Gates and size

- validation_eligible: FAIL
- docs: PASS
- emoji: PASS
- fresh_negation: FAIL
- fresh_numbers: PASS
- unseen_corpus: FAIL
- sts_en: PASS
- sts_fa: PASS
- size: PASS
- round2_en_ranking: FAIL
- round2_fa_ranking: FAIL
- round3_en_ranking: PASS
- round3_fa_ranking: PASS
- round4_en_ranking: PASS
- round4_fa_ranking: PASS

Packed weights: 5,165,680 bytes; 9,901,952 parameters; unchanged 32k tokenizer.

| Bundle | Round3 | Previous native | Candidate |
|---|---:|---:|---:|
| wasm_bytes | 8,004,767 | 8,004,767 | 8,004,767 |
| gzip_bytes | 5,347,369 | 5,346,874 | 5,346,935 |

## Scope and evidence

Small authored fresh set; query bootstrap does not measure author/topic bias. Known-corpus supervision and unseen-corpus evaluation are separate. Historical fixtures are development regressions, not fresh evidence.

300 training queries and 100 validation queries supervise 30 known docs and 70 emoji. The 76 new queries were frozen before training; 12 additionally use a new document corpus. The 538,417-text audit found no normalized exact or eight-word overlap for the 112 held-out query/new-document texts.

The original six-epoch run and its three calibration candidates failed validation. A separately recorded retention repair increased Persian replay and historical ranking preservation while lowering learning rates; see the [protocol](persian-pilot-domain.md). All logs and rejected checkpoints are preserved.

Raw per-query ranks, contrast margins, paired-direction results, source hashes and bootstrap intervals are in `dist/pilot-fa-en-domain/`. Preview: `/eval/quality/native-preview.html`.

## What improved and what remains broken

- Fresh docs: queued state updates now retrieve the functional setter document; lifting state, cleanup, refs, context and reducers also improve. One dependency-array query still incorrectly retrieves cleanup.
- Fresh colloquial emoji: the champion paraphrase retrieves 🏆; fever retrieves 🤒; debugging retrieves 🐛. Known user regressions also pass without aliases: محبت → 😍 and «بهترینِ تمام دوران» → 🏆; a literal goat remains 🐐.
- Numeric contrast at 6/12 is not successful reasoning: both-direction accuracy is 0/6. The model returns 32 remaining jobs for both 37−9+4 and 37−9−4, although the second answer is 24. The non-regression gate passing does not imply numeric competence.
- Complex negation declines to 6/12; only 1/6 pairs is correct in both directions. Having a necessary approval is confused with missing approval even when other conditions are explicitly unknown.
- English STS is stable, but historical round2 English ranking drops from .8453 to .8248 (top1 11/18 to 10/18). New-corpus retrieval improves over the native pilot but remains below round3; native QA also loses the previous native pilot gain.

The selected artifact is the original run epoch2 diagnostic checkpoint. All nine training epochs completed (~15.5 minutes of epoch runtime); none of the six validation interpolation candidates passed every gate. No fresh candidate score was used for checkpoint selection.

Validation: 31 Python tests passed; int4 source/packed inference matched exactly for 32 samples; actual WASM parity passed 148 cases (max absolute difference .000298389, minimum cosine .999998351). Browser docs and emoji checks use the same pure-cosine artifact. These checks establish implementation parity, not general language competence.

No model was published or installed as the default example model. Existing source-data licensing and pretraining exposure have not been re-audited in this experiment.
