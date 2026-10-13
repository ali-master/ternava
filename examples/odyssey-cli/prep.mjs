// Persian retelling paragraphs are the editable source; no remote book download.
import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const text = readFileSync(new URL('./odyssey.fa.txt', import.meta.url), 'utf8');
const chunks = text.trim().split(/\n\s*\n/u).map(p => p.replace(/\s+/gu, ' ').trim());
assert(chunks.length > 0 && chunks.every(p => /[\u0600-\u06ff]/u.test(p)), 'متن فارسی خالی یا نامعتبر است');
writeFileSync(new URL('./odyssey.json', import.meta.url), JSON.stringify(chunks, null, 2) + '\n');
console.log(`${chunks.length.toLocaleString('fa-IR')} گزیدهٔ فارسی آماده شد.`);
