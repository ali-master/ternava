# Compositional diagnostic and pilot

This experiment continues the local 32k mini; it does not promote the experimental model to the published package. Its final test is `eval/quality/corpus/persian-compositional-hard.json`. Earlier challenges are development/regression data.

## Diagnosis before training

`training/distill/diagnose_reasoning.py` reconstructs 96 exact numeric training pairs and changes one factor at a time. It also checks 144 negation pairs, with repeated wording probes sharing a smaller set of unique examples; these are controlled probes, not 144 independent natural-language tests. The same source checkpoint is tested with int4 embeddings, fp32 embeddings, and zeroed positional embeddings. The previous raw trained checkpoint and the post-training English-safe transplant are measured separately.

| Current model intervention | Numeric both-direction wins / 96 | With adjacent-answer negatives / 96 | Negation both-direction wins / 144 |
|---|---:|---:|---:|
| Familiar examples | 47 | 14 | 144 |
| New wording only | 12 | 3 | 57 |
| New number range only | 17 | 5 | — |
| New noun only | 41 | 12 | 144 |
| Uniform digit representation | 12 | 2 | — |
| Mask query numeral identities | 38 | 11 | — |
| Use fp32 embeddings | 46 | 15 | 144 |

No unknown tokens or truncation occurred (longest sequence: 25 tokens). The arithmetic two-choice objective admits an operator/magnitude shortcut; 38 pairs still pass after replacing query numeral identities. This intervention leaves numeral-word count and conjunction cues, so it is evidence of shortcut use, not a proof that every numeric feature is ignored. Adjacent-answer discrimination is substantially harder. Negation is strongly dependent on familiar wording, not just domain nouns. Int4 is not the main cause. Removing positions does not explain the wording failure. Post-training weight restoration lowers familiar numeric wins from 63 to 47, but the raw model already fails new wording (5/96); restoration is not the sole cause.

## Revised method and controlled comparison

- 4,800 training and 320 validation records, with two equivalent queries and four candidate answers per record.
- Numeric positives come from exact training-only arithmetic. Negatives include neighboring answers and a wrong operation. Numeric representation is sampled independently of the answer label.
- Negation composes explicit truth/falsehood operators and a distinct unknown-evidence state; proposition truth is checked separately from certainty. Three-valued labels prevent treating lack of evidence as falsity.
- New objective: full-batch multi-positive retrieval, nearest-answer margin, equivalent-query consistency. Semantically duplicate answers are positives, not false negatives.
- Control: same data, initialization, row mask, seed, replay and 4 × 250 optimizer steps, but the previous binary margin sees only the first query, positive and wrong-operation negative. This isolates the **combined objective change**, not its three individual components.
- Both runs freeze the existing backbone and all non-Persian embedding rows from the start; no post-training transplant. Current-model domain/replay geometry and supervised docs/emoji retrieval are retained. No inference-time branch, rule, calculator, aliases, added parameters or tokenizer changes.
- Fixed English weights constrain learning capacity. This is a bounded experiment, not a promise that a frozen backbone can acquire general arithmetic.

The initial preparation accidentally gave 293/300 training boundary questions a positive label and identical negation validation views. The first run was interrupted during epoch two, preserved under `*-invalid-validation`, and excluded from selection. The corrected generator balances below/equal/above cases and asserts different query views; tests cover these failures. Both final arms start again from the same original checkpoint.

## Selection and evidence

Select on disjoint validation wording/nouns: eligibility first, then the mean of exact-answer agreement across both query views and full-corpus top1 for negation and numbers. Validation includes all historical retrieval sets and previously consumed challenges. Docs/emoji top1 and NDCG cannot decline on their domain, native and previous reasoning fixtures; English must remain exact; Persian STS tolerance is .002, historical Persian NDCG tolerance .01. Logic must not regress. Every failed gate stays visible.

Freeze source, data and final-test hashes before training. Freeze each selected checkpoint before final inference. Report actual WASM cosine retrieval, paired contrast and paired full-corpus top1, external STS, English vector equality and actual model/WASM/gzip sizes. Never use final-test labels for checkpoint selection. A failed candidate remains diagnostic; the existing model remains the deployment choice.

To claim a functional fix on the small final Persian logic sets, require at least 14/16 strict contrasts and 6/8 pairs with **both full-corpus top1 answers correct** in each category, alongside all preservation gates. Smaller gains are reported as partial. The audit discloses shared eight-word spans; absence of complete-text matches does not establish independence from teacher pretraining.
