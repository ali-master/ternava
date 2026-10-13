# eval — quality + performance scorecard

> For contributors running quality and performance evaluations against the shipped engine. If you're *using* ternava in an app, see [`packages/mini/`](../packages/mini). For the latest published numbers, see [`quality/RESULTS.md`](quality/RESULTS.md).

Cross-cutting evaluation of the *shipped* engine (engine + bundled model + JS API). Distinct from:

- [`engine/tests/`](../engine/) — element-level parity tests (does the Rust math match Python?)
- [`packages/mini/tests/`](../packages/mini/) — JS API integration tests

## Layout

```
eval/
├── quality/         Quality scorecard — Spearman, retrieval, charts
├── regression/      Engine vs baselines on real eval tasks
├── benchmarks/      Latency (cold + warm), throughput, memory
├── compatibility/   Target-runtime matrix (Node, Browser, CF Workers, Deno, Bun)
├── results/         Committed JSON outputs per release version
└── REPORT.md        Human-readable scorecard, regenerated per release
```

## Run all evals

```bash
bash scripts/run-eval.sh        # populates results/v<X.Y.Z>.json + regenerates REPORT.md
```

## The release scorecard

Six dimensions per release. See [`../docs/eval-methodology.md`](../docs/eval-methodology.md) for what each metric measures and why.

| Dimension | What gets measured |
|---|---|
| Quality | Teacher alignment, STS-B AUC + Spearman, Recall@K |
| Quantization gap | fp32 baseline vs ternary, per component (embedding / BitLinear / activations) |
| Runtime performance | Latency (cold + warm), throughput, memory peak |
| Size | `.wasm` bytes, `.bin` bytes, gzipped over the wire |
| Compatibility | Required Wasm features, min Node/browser, per-target pass/fail |
| Honest comparison | Side-by-side with `transformers.js` + MiniLM, ONNX, embedding APIs |

## Operating principles

1. **Always publish the gap, not just the headline.** Numbers without context mislead.
2. **Version-anchor everything.** `results/v<X.Y.Z>.json` is committed at release time.
3. **Same harness, every release.** `scripts/run-eval.sh` regenerates everything; manual scorecard updates are a smell.
4. **Distinguish engine from model.** Both engine bugs and model weaknesses show up as metric drops — separate them by reporting parity vs absolute quality side-by-side.
