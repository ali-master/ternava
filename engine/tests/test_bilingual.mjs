import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const expected = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const result = spawnSync(process.execPath, [fileURLToPath(new URL('../../eval/quality/collect-wasm.mjs', import.meta.url)), process.argv[2]], {
  input: JSON.stringify(expected.texts), encoding: 'utf8', maxBuffer: 10 * 1024 * 1024,
});
assert.equal(result.status, 0, result.stderr);
const actual = JSON.parse(result.stdout);
assert.equal(actual.model_sha256, expected.model_sha256);
assert.equal(actual.tokenizer_sha256, expected.tokenizer_sha256);
assert.deepEqual(actual.tokens, expected.input_ids);
let maxDiff = 0, minCosine = 1;
for (let i = 0; i < expected.embeddings.length; i++) {
  for (let j = 0; j < 384; j++) maxDiff = Math.max(maxDiff, Math.abs(actual.embeddings[i][j] - expected.embeddings[i][j]));
  const a = actual.embeddings[i], b = expected.embeddings[i];
  const dot = a.reduce((sum, v, j) => sum + v * b[j], 0);
  const norm = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0) * b.reduce((sum, v) => sum + v * v, 0));
  minCosine = Math.min(minCosine, dot / norm);
}
// fp32 reductions can cross int8 rounding boundaries; even PyTorch padded vs compact inputs differ.
assert(maxDiff <= 1e-3 && minCosine >= 0.9999, `Python/WASM difference ${maxDiff}, cosine ${minCosine}`);
console.log(`PASS: ${expected.texts.length} tokenizer cases, max difference ${maxDiff}, minimum cosine ${minCosine}`);
