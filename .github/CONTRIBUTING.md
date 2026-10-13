# Contributing to Ternava

> **Status:** v0.1, pre-alpha. Not yet accepting PRs from outside contributors — this file is here so the structure exists when we open up.

## Where to start

The repo is organized so contributors only need the toolchain for the layer they're touching:

| If you're touching | You need | Read first |
|---|---|---|
| `packages/` (JS API) | Node.js, pnpm | [packages/mini/README.md](../packages/mini/README.md) |
| `engine/` (Wasm engine) | Rust, wasm-pack | [engine/README.md](../engine/README.md) and [docs/training/model-internals.md](../docs/training/model-internals.md) |
| `training/` (model training) | Python, PyTorch, GPU | [training/README.md](../training/README.md) and [docs/training/](../docs/training/) |
| `eval/` (quality + perf) | Node.js + Python | [eval/README.md](../eval/README.md) |
| `docs/` | Markdown editor | [docs/](../docs/) |

## Before you start work

- Open an issue describing what you intend to change. Avoids duplicate effort and surfaces design concerns early.
- For changes that touch the engine or training math: read [docs/training/postmortem-bitlinear-asymmetry.md](../docs/training/postmortem-bitlinear-asymmetry.md). It's the cautionary tale about how easy it is to ship a divergent forward pass and what tests catch it.

## Quality bar

- Engine changes that affect output: must pass `node engine/tests/test_embed.js` AND `bash scripts/run-eval.sh` with no regression vs the prior release's `eval/results/v<X.Y.Z>.json`.
- Training changes: must produce a checkpoint that `training/distill/evaluate.py` reports as PASS on all three Phase 1 tasks.
- JS API changes: must include tests in `packages/<pkg>/tests/`.
- Doc changes: no formal bar, but follow the existing voice (terse, technical, no marketing language).

## Style

- Code: follow existing conventions per language (rustfmt for Rust, prettier defaults for JS, black for Python).
- Commits: imperative tense ("add internal LayerNorm to bitlinear_forward"), reference issue numbers.
- PRs: describe the *why*, not just the *what*. Include eval/REPORT.md before/after if quality-affecting.

## Reporting issues

For bugs, include:
- Reproduction (input string + expected output + actual output)
- Engine version + npm package version
- Runtime + OS + Node version

For quality regressions, run `bash scripts/run-eval.sh` and paste the resulting `eval/results/v<X.Y.Z>.json` into the issue.

## License

By contributing, you agree your contributions will be licensed under the MIT License (see [../LICENSE](../LICENSE)).