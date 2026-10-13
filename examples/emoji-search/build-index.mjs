// Rebuild Persian text vectors with the exact WASM shipped by this demo.
import init, { embed, engineInfo } from './wasm/0.2.0/index.mjs';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { searchPhrases } from './search.js';

const source = readFileSync(new URL('./corpus.fa.json', import.meta.url));
const items = JSON.parse(source);
assert(items.length > 0 && new Set(items.map(e => e.emoji)).size === items.length, 'دادهٔ ایموجی خالی یا تکراری است');
const wasm = readFileSync(new URL('./wasm/0.2.0/tern_engine_bg.wasm', import.meta.url));
await init(wasm);
const vectors = items.map(({ name, text }) => {
  assert(typeof name === 'string' && typeof text === 'string' && /[\u0600-\u06ff]/u.test(name + text), 'متن فارسی لازم است');
  const vector = Array.from(embed(`${name}، ${text}`));
  assert(vector.length === 384 && vector.every(Number.isFinite), 'بردار نامعتبر است');
  return [vector.map(x => Math.round(x * 1e5) / 1e5)];
});
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const index = { model: engineInfo(), language: 'fa', wasm_sha256: hash(wasm), corpus_sha256: hash(source),
  dim: 384, count: items.length, emojis: items.map(e => e.emoji), names: items.map(e => e.name),
  phrases: items.map(searchPhrases), vecs: vectors };
writeFileSync(new URL('./emoji-index.json', import.meta.url), JSON.stringify(index));
console.log(`فهرست فارسی با ${items.length.toLocaleString('fa-IR')} ایموجی ساخته شد.`);
