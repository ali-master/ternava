# Documentation

In-depth concepts and methodology behind ternava. For end-user usage, see
[`packages/mini/`](../packages/mini). For build and contribution
notes specific to one component, see the README in that directory.

## What's here

| Doc | What it covers |
|---|---|
| [overview.md](overview.md) | Project overview — what ternava is, how the three technical choices stack, when to use it |
| [architecture.md](architecture.md) | System design — the `.bin` format, runtime model, packaging pipeline |
| [inference-engine.md](inference-engine.md) | Runtime internals — engine layout, BitLinear forward pass, tokenization path |
| [model-internals.md](model-internals.md) | Canonical math reference — forward pass, backprop, distillation dynamics |
| [eval-methodology.md](eval-methodology.md) | Quality scorecard methodology — each metric, how to reproduce it |

## Reading paths

- **New visitor curious about the project**
  - [overview.md](overview.md) — end-to-end framing in a single read

- **Contributor touching the inference engine**
  - [architecture.md](architecture.md) — system design and packaging
  - [inference-engine.md](inference-engine.md) — runtime internals
  - [model-internals.md](model-internals.md) — canonical math reference for runtime/training parity

- **Researcher reproducing results**
  - [eval-methodology.md](eval-methodology.md) — scorecard methodology
  - [`eval/quality/RESULTS.md`](../eval/quality/RESULTS.md) — the published numbers
  - [model-internals.md](model-internals.md) — training-side math
