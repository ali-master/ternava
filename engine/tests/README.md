# engine/tests — parity tests

Element-level tests that the Rust engine's output matches the Python reference at
[`training/pack/unpack.py`](../../training/pack/unpack.py) (`UnpackedModel.forward()`).

```bash
# Build a feature first
wasm-pack build --target nodejs --features emb_int8
# Then run the tests
node tests/test_embed.js
```

## What goes here

- `test_embed.js` — full forward-pass parity test (engine vs `unpack.UnpackedModel.forward()`)
- `test_tokenizer.js` — spot-check `tokenize()` against the Python `tokenizers` library
- `test_embedding.js` — per-format embedding lookup parity
- `test_bitlinear.js` — single-BitLinear forward parity
- `reference_dumps/` — captured Python reference outputs (gitignored; regenerated via script)

## What does NOT go here

- Engine quality eval against held-out tasks → that lives in [`../../eval/regression/`](../../eval/regression/)
- Performance benchmarks → [`../../eval/benchmarks/`](../../eval/benchmarks/)
- Package integration tests → `packages/*/tests/`

## The reference must be the real model

The single source of truth for engine forward-pass correctness is `training/pack/unpack.py`.
That module uses the same quant math as the pinned `bitlinear==2.4.6` library — so matching
`unpack.UnpackedModel.forward()` IS matching the training-time model.

DO NOT hand-write Python reference math separately. The prior tern-core POC hit exactly that
trap: parity tests passed at 1e-7 because both the engine and the hand-written reference were
buggy in the same way. See
[`../../docs/training/postmortem-bitlinear-asymmetry.md`](../../docs/training/postmortem-bitlinear-asymmetry.md).

Per-format tolerances are defined in
[`../../docs/tern-inference-engine.md`](../../docs/tern-inference-engine.md#verification--parity-contract).

## Status

`test_bilingual.mjs` checks Python/WASM token IDs and int4 forward parity, including
Persian normalization, empty input and truncation. Generate its untrained fixture
and build it using the commands in [the pilot runbook](../../docs/persian-pilot.md#offline-tests-and-cross-runtime-smoke).

## Portable loader

`test_runtime.mjs` exercises the shared runtime against a built web artifact:
offline file loading, HTTP loading, bytes and compiled modules, concurrent init,
failed-load retry, validation, helper results, and both `/runtime` export maps.
The supplied artifact is reused for both export-map tests; this does not evaluate
the base model's quality. Every tested vector must equal the raw WASM API output.

```sh
# From the repository root, after building the pilot:
node engine/tests/test_runtime.mjs dist/pilot-fa-en-reasoning/d256 dist/pilot-fa-en-reasoning/d256/runtime-parity.json
bun engine/tests/test_runtime.mjs dist/pilot-fa-en-reasoning/d256
python3 -m http.server 8877 --bind 127.0.0.1
# Open http://127.0.0.1:8877/engine/tests/runtime.html
```

The browser page compares its vectors with the Node-generated fixture and checks
the WASM hash. Use `?artifact=/dist/your-build/d256/` to test another artifact;
generate its `runtime-parity.json` first. These are compatibility checks, not
evidence of semantic quality. CI runs Node/Bun against the untrained pilot fixture;
the browser check is manual.
