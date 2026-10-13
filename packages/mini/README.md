# @usex/ternava — Ternava / ترنوا

Experimental Persian/English semantic embeddings for Node.js, Bun and browsers.
Built on [the upstream project](https://github.com/ali-master/ternava#acknowledgments) by Wenshu Tang — thank
you for the open-source model, Rust/WASM engine and training foundation.
Original MIT copyright is preserved in LICENSE.

The bilingual mini has 32,000 WordPiece tokens, 9,901,952 parameters, a 128-token
input limit, and 384-dimensional normalized embeddings. Weights use int4
embeddings and ternary linear layers. The default build uses the existing
`pilot-fa-en-reasoning` reference, not the failed compositional candidate.
**Persian negation and numeric reasoning remain unreliable.** Translated demos
and a successful import are not general language-quality evidence.

## Installation

After publication: `npm install @usex/ternava` or `bun add @usex/ternava`.
Before publication, install the tarball produced by `npm run verify:release`
from the repository root. Nothing here implies that a release is already live.

```js
import { embed, cosineSim, similar, engineInfo } from '@usex/ternava';
const vector = embed('خیلی دوستت دارم');
console.log(vector.length); // 384
console.log(cosineSim(vector, embed('عاشقتم')));
console.log(similar('forgot my password', [
  'Reset a forgotten password', 'Track a delivery',
], { topK: 1 }));
console.log(engineInfo());
```

Node.js and Bun can also use `const { embed } = require('@usex/ternava')`.
The main entry initializes synchronously from local WASM. It makes no API calls.

## Portable loader

```js
import init, { embed, similar } from '@usex/ternava/runtime';
await init();
console.log(embed('سلام دنیا'));
```

Node/Bun read adjacent WASM from disk; browsers fetch it relative to the module
URL. In a plain browser, serve `pkg-web/` and import `./pkg-web/index.mjs`, or use
an import map for the package name. Keep its generated files together. `init`
accepts a URL, WASM bytes, or WebAssembly.Module; concurrent calls share a
promise and failed loads can be retried. Inference is synchronous after init.

`@usex/ternava/web` exposes the raw engine (`embed`, `tokenize`,
`config_summary`) and default init, without the JS helpers. The main browser
entry imports WASM as an ESM asset and requires a compatible bundler; webpack 5
uses `experiments: { asyncWebAssembly: true }`. For custom loaders, prefer
`/runtime` and explicit initialization.

## API

| Export | Behavior |
|---|---|
| `embed(text)` | Float32Array(384), normalized; truncates at 128 tokens |
| `cosineSim(a, b)` | Dot product of normalized vectors |
| `similar(query, corpus, { topK })` | Sorted `{ text, sim }` results; default topK 5 |
| `engineInfo()` | Embedded model and tokenizer configuration |
| `TernError` | Stable `INVALID_INPUT`, `DIM_MISMATCH` codes; runtime adds `NOT_INITIALIZED` |

Cache corpus embeddings for repeated searches. Rebuild indexes whenever model
weights or tokenizer change; vectors from different models are not compatible.

[Source and measured pilot reports](https://github.com/ali-master/ternava).
`tern_engine` asset names and
`TernError` remain stable engine/API identifiers.

## License

MIT. See LICENSE for the original attribution.
