# engine — Rust → Wasm inference engine

> For contributors building or modifying the WASM engine. If you're *using* ternava in an app, see [`packages/mini/`](../packages/mini) instead.

The math half of ternava. Reads the trained model from `assets/model.bin`, tokenizes input via the embedded BERT tokenizer, runs the full BitLinear-faithful forward pass, and returns a 384-dim L2-normalized embedding.

## Build

```bash
rustup target add wasm32-unknown-unknown

wasm-pack build --target nodejs --features emb_int4     # primary ship
wasm-pack build --target nodejs --features emb_int8     # higher-quality variant
wasm-pack build --target nodejs --features emb_ternary  # size-extreme variant
wasm-pack build --target nodejs --features emb_fp32     # parity reference (not for users)
```

Exactly one feature per build, enforced at compile time. Output lands in `pkg/`. `scripts/build-engine.sh` at the repo root runs the full sweep, applies `wasm-opt -Oz`, and copies the artifacts into `packages/mini/pkg/`.

## Layout

```
engine/
├── Cargo.toml         build features (mutually exclusive)
├── src/
│   ├── lib.rs         WASM API (#[wasm_bindgen])
│   ├── format.rs      .bin header parse + verify
│   ├── tokenizer.rs   BERT tokenizer (vocab embedded)
│   ├── model.rs       weight layout precompute
│   ├── kernels.rs     BitLinear + embedding kernels
│   └── inference.rs   forward pass
├── assets/
│   ├── tokenizer.json BERT vocab (committed)
│   └── model.bin      weights (release artifact, not committed)
└── tests/             parity tests
```

## Math reference

The canonical forward-pass math lives in [`../docs/model-internals.md`](../docs/model-internals.md). The `bitlinear_forward` function in `kernels.rs` is an independent Rust reimplementation that mirrors [`bitlinear==2.4.6`](https://github.com/schneiderkamplab/bitlinear)'s `BitLinear.forward()` byte-for-byte — keeping training-time and runtime arithmetic in lockstep is required for quality to transfer.

## Verification — parity tests

```bash
node tests/test_embed.js
```

For every build target, a parity test asserts that `embed(text)` output matches the Python reference (`UnpackedModel.from_bin(...).forward(...)` from `training/pack/unpack.py`) within a per-format tolerance, after L2-normalization.

| Build | Per-dim tolerance |
|---|---|
| `emb_fp32` | 1e-5 (effectively bit-exact) |
| `emb_int4` | 5e-4 |
| `emb_int8` | 1e-4 |
| `emb_ternary` | 5e-3 |

**The contract is inference-level, not byte-level.** Byte equality is necessary but not sufficient — a buggy forward pass could pass byte tests while shipping a different model. Parity tests run the full forward pass on a fixed input set and compare against the Python reference.

Failure modes the parity test must catch:

- Endianness drift in header or weight reads
- Sign-table mismatch in ternary unpack
- Nibble-order mismatch in int4 unpack
- Off-by-one in per-row scale indexing
- Wrong `activation_scale × weight_scale` combination order in BitLinear
- LayerNorm epsilon mismatch
- Output projection accidentally ternarized (silent quality loss)

Reference dumps and test harness live in `tests/`.

## Smoke test

After every build, before measuring perf:

```bash
node ../eval/benchmarks/smoke.js
```

Runs ~10 sentence pairs through the engine and prints cosine scores. Catches "model loaded but produces semantic garbage" failure modes that aggregate quality metrics would mask.

## Status

v0.1, pre-alpha. Engine ships `emb_int4` by default — ~2 ms per call (p50) on M4 Max, ~450 emb/sec sustained throughput. Full benchmark history in [`../eval/benchmarks/results/`](../eval/benchmarks/results/).
