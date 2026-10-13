# Ternava model artifacts

Ternava's public npm package is `@usex/ternava`. The default build embeds the
experimental bilingual mini from `training/pack/out/pilot-fa-en-reasoning/`:
`model.bin`, `model.bin.json` (provenance/checksum), and `model.tokenizer.json`.
These generated files are not committed to git. Reproduce them with the
[pilot protocol](../docs/persian-pilot.md) and
[reasoning continuation](../docs/persian-pilot-reasoning.md).

The model has 32,000 tokens and 5,165,680 bytes of packed weights. It remains
experimental; see the [current results](../README.md#persian--english-pilot).
Build with `bun run build`; verify/install the package with
`bun run verify:release`. No upload or publication occurs automatically.

## Upstream provenance

The [original English model artifacts](../README.md#acknowledgments) belong to
Wenshu Tang's upstream project. They are **not Ternava's bilingual model**.
The [Ternava model card](huggingface/README.md) describes the current bilingual
pilot and links its measured limitations. Do not upload Ternava artifacts to
the upstream author's model repository.
