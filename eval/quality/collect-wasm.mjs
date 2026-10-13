// Web-target glue accepts bytes, so this evaluates the browser artifact in Node.
import { readFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';

const glue = resolve(process.argv[2]);
const engine = await import(pathToFileURL(glue).href);
const bytes = readFileSync(join(dirname(glue), 'tern_engine_bg.wasm'));
const start = performance.now();
const runtime = await engine.default({ module_or_path: bytes });
const config = engine.config_summary();
const coldStartMs = performance.now() - start;
const texts = JSON.parse(readFileSync(0, 'utf8'));
if (!Array.isArray(texts) || !texts.every(t => typeof t === 'string')) throw new Error('Expected text array');
for (let i = 0; i < 5; i++) engine.embed('سلام hello');
const embeddings = [], tokens = [], latencyMs = [];
for (const text of texts) {
  const t = performance.now();
  const vector = Array.from(engine.embed(text));
  latencyMs.push(performance.now() - t);
  if (vector.length !== 384 || !vector.every(Number.isFinite)) throw new Error('Invalid embedding');
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (Math.abs(norm - 1) > 0.001) throw new Error('Embedding is not normalized');
  embeddings.push(vector);
  tokens.push(Array.from(engine.tokenize(text)));
}
console.log(JSON.stringify({ config, embeddings, tokens, latency_ms: latencyMs,
  runtime: { node: process.version, platform: process.platform, arch: process.arch },
  cold_start_ms: coldStartMs, wasm_bytes: bytes.length,
  gzip_bytes: gzipSync(bytes, { level: 9 }).length,
  wasm_sha256: createHash('sha256').update(bytes).digest('hex'),
  model_sha256: engine.model_sha256?.() ?? null,
  tokenizer_sha256: engine.tokenizer_sha256?.() ?? null,
  wasm_linear_memory_bytes: runtime.memory.buffer.byteLength,
}));
