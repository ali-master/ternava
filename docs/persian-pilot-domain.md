# Technical and colloquial Persian pilot

Local experiment, 2026-10-12. Continue round-three `best.pt`; keep the 32k
tokenizer, 9,901,952 parameters and existing packed runtime. The previous native
continuation improved QA but failed application and logic retention gates.

## Fixed experiment, recorded before training

300 authored training queries and 100 separate validation queries cover all
30 docs and 70 emoji entries. Known corpus documents are supervised training
inputs. Love and celebration queries allow multiple appropriate emoji. No alias
fields, runtime query rules, calculators, extra model or reranker are used.

Seed 613, six epochs of 250 steps, AdamW zero decay, 50-step warmup and linear
decay. Persian embedding rows learn at 0.00012 with int4 STE; the backbone learns
at 0.000003. Other embedding rows stay fixed, but English output is not assumed
unchanged. Domain listwise loss includes the entire task corpus. Balanced paired
logic, historical triplets, English/Persian replay and STS retain prior skills.
Exact batch sizes and loss weights are saved in `recipe.json` with each run.

Select only on the 100 domain validation queries and historical development
fixtures: mean docs/emoji NDCG + .15 worst negation/numbers contrast. Eligibility:
English validation STS loss <=.002; Persian <=.01; each old round2–4 FA/EN NDCG
loss <=.02 absolute; pooled FA negation/numeric accuracy and previous demo top1
must not fall. If no eligible checkpoint exists, try validation-only weight
interpolation of diagnostic-best with round three at .25/.5/.75 under exactly
the same gates. Preserve the failed run; never select on fresh tests.

## Frozen final comparison

`persian-domain-hard.json` contains 76 fresh queries: docs20, emoji20, a new
12-document corpus with 12 queries, and 24 balanced logic queries (12 negation,
12 numeric). The last are 12 counterfactual pairs; report both per-query and
both-directions-correct pair accuracy. The new corpus tests new documents, not
necessarily new concepts. Hashes are frozen in the preparation manifest.
Audit normalized exact/8-word overlap against accessible training/validation
texts. This cannot establish absence from ancestor teacher pretraining.

Evaluate packed WASM by pure cosine. Compare round3, previous native and new
candidate on identical inputs. Also report every old FA/EN retrieval fixture,
external STS (FA8566/EN1365), previous native QA and application regression sets.

Promotion requires eligibility above; fresh docs and emoji top1 must each
exceed both baselines and reach >=50%; fresh negation and numbers contrast
must not regress versus either baseline; new-corpus top1 must not regress;
external English STS loss <=.002 and Persian <=.01 versus round3;
gzip growth <=0.5%. Report pair failures even when individual accuracy passes.
Any failure keeps the artifact experimental and existing example defaults intact.

## Validation-only repair (after first run, before candidate test inference)

All six epochs and all three interpolations failed historical retrieval gates.
English STS was stable; weak Persian replay (weight 2) was a plausible contributor
to distorted inherited ranking, not an established sole cause. Run a separate fixed retention repair from
round3: seed613, three epochs x200 steps, embedding LR .00006, backbone .000001,
Persian replay weight20 and historical triplet weight1. Everything else, including
all gates, stays fixed. The same three validation-only interpolation coefficients
are available if required. This is an adaptive validation experiment, not part of
the original preregistration; fresh candidate results remain unopened throughout.
Compare eligible candidates by the same validation score. If neither run yields
one, export the highest validation-scoring diagnostic candidate explicitly as
failed/experimental; no default model replacement.

## Reproduce

```sh
PYTHONPATH=training/distill .venv/bin/python -m prep.domain
.venv/bin/python -m unittest discover -s training/tests -p 'test_domain.py'
PYTHONPATH=training/distill .venv/bin/python -m domain --smoke
PYTHONPATH=training/distill .venv/bin/python -m domain
# Validation-only fallback and separately recorded retention repair:
PYTHONPATH=training/distill .venv/bin/python -m domain --calibrate
PYTHONPATH=training/distill .venv/bin/python -m domain --retention-repair --smoke
PYTHONPATH=training/distill .venv/bin/python -m domain --retention-repair
PYTHONPATH=training/distill .venv/bin/python -m domain --retention-repair --calibrate
```

Preparation/training/evaluation refuse to replace existing artifacts. No new
dependencies or downloads are required. Existing source-data provenance and
research-only distribution limits remain applicable.

Completed result: no candidate passed every gate. The original run epoch2
`diagnostic-best.pt` is exported as an experimental artifact, selected by validation
score before any fresh candidate inference. See [measured results](persian-pilot-domain-results.md).

```sh
.venv/bin/python training/pack/pack.py --ckpt training/distill/runs/mini-fa-en-domain/diagnostic-best.pt --embedding-format int4 --output training/pack/out/pilot-fa-en-domain/model.bin
BIN="$PWD/training/pack/out/pilot-fa-en-domain/model.bin" TOKENIZER="$PWD/training/pack/out/pilot-fa-en-domain/model.tokenizer.json" PKG=packages/mini TARGET=web WEB_OUT=dist/pilot-fa-en-domain bash scripts/build-engine.sh
.venv/bin/python eval/quality/domain.py --model round3=dist/pilot-fa-en-v3/d256/tern_engine.js --model native=dist/pilot-fa-en-v3-native/d256/tern_engine.js --model candidate=dist/pilot-fa-en-domain/d256/tern_engine.js
.venv/bin/python eval/quality/sts-heldout.py --dataset dist/pilot-fa-en/external-sts.json --model candidate=dist/pilot-fa-en-domain/d256/tern_engine.js --output dist/pilot-fa-en-domain/sts-candidate.json
.venv/bin/python eval/quality/domain-report.py
```
