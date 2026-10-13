# @usex/ternava-base — internal research workspace

This workspace is **private** and is not published to npm. The supported package
is [`@usex/ternava`](../mini/README.md), the bilingual mini model.

The base architecture is retained for research and upstream comparisons. Build
it explicitly with a compatible d384 packed model and matching tokenizer:

```bash
BIN=/absolute/path/to/base.bin TOKENIZER=/absolute/path/to/tokenizer.json \
PKG=packages/base bash scripts/build-engine.sh
```

It shares the main, `/web` and `/runtime` APIs with mini, but requires its own
model artifacts. This is not a trained bilingual base release.

Based on [the upstream project](../../README.md#acknowledgments) by Wenshu Tang.
MIT; original copyright preserved in LICENSE.
