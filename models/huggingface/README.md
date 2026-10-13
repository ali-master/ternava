---
license: mit
language:
  - fa
  - en
pipeline_tag: sentence-similarity
tags:
  - sentence-embeddings
  - semantic-search
  - on-device
  - wasm
  - bitnet
  - ternary
  - quantization
  - distillation
---

# Ternava — ترنوا

An **experimental Persian/English mini sentence encoder** for local inference
in Node.js, Bun and browsers. The npm package is **`@usex/ternava`**.

Built on the [upstream project credited in the README](../../README.md#acknowledgments)
by Wenshu Tang. Thank you for the model, Rust/WASM engine and training foundation.
Original MIT attribution remains in [LICENSE](../../LICENSE).

## Model

| Property | Current bilingual mini |
|---|---|
| Tokenizer | 32,000-token bilingual WordPiece |
| Parameters | 9,901,952 |
| Transformer | 2 layers, hidden size 256, 4 attention heads |
| Output | 384-dimensional L2-normalized embedding |
| Input limit | 128 tokens; longer input is truncated |
| Quantization | int4 token embeddings, ternary linear weights |
| Packed model | 5,165,680 bytes |
| Complete web WASM | 8,004,767 bytes |
| Selected local reference | `pilot-fa-en-reasoning` |

## Use

After npm publication:

```bash
npm install @usex/ternava
```

```js
import { embed, cosineSim } from '@usex/ternava';
console.log(cosineSim(embed('خیلی دوستت دارم'), embed('عاشقتم')));
```

For explicit loading in Node.js, Bun or browsers:

```js
import init, { embed } from '@usex/ternava/runtime';
await init();
console.log(embed('سلام دنیا'));
```

Before publication, build and install the local tarball using the
[release instructions](../../README.md#build-verify-and-publish).
This card does not imply a live Hugging Face or npm release. No separate model
files are fetched at inference time; the model and tokenizer are embedded in WASM.

## Quality and limitations

Persian negation, numerical reasoning and generalization remain unreliable.
Use the [measured bilingual pilot results](../../README.md#persian--english-pilot)
and [latest compositional comparison](../../docs/persian-pilot-compositional-results.md).
The latest candidate failed the preservation gates and does not replace the
selected reference. These small authored test sets are not a general benchmark.

The [historical English baseline](../../eval/quality/RESULTS.md) was measured on
the upstream model before this fork; those numbers do not describe bilingual
Ternava quality. Aliases in a demo are not evidence of semantic understanding.

## Training and attribution

See the [pilot protocol](../../docs/persian-pilot.md) for data sources,
multilingual supervision and tokenizer preparation, and the
[reasoning continuation](../../docs/persian-pilot-reasoning.md) for the selected
model. Training and evaluation reports document experiment-specific limitations.

- [Source](https://github.com/ali-master/ternava)
- [Package API](../../packages/mini/README.md)
- [Acknowledgments](../../README.md#acknowledgments)
