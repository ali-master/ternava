// index.mjs — embed the entire Odyssey, live, on your CPU.  `node index.mjs`
import { embed } from '@usex/ternava';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const source = readFileSync(new URL('./odyssey.json', import.meta.url));
const chunks = JSON.parse(source);
console.log(`\n  ساخت نمایهٔ ادیسه — ${chunks.length.toLocaleString('fa-IR')} گزیدهٔ فارسی\n`);

const t0 = performance.now();
const vecs = chunks.map((text, i) => {
  const v = embed(text); // string -> 384-dim vector
  if (i % 100 === 0) bar(i + 1, chunks.length, t0);
  return Array.from(v);
});
bar(chunks.length, chunks.length, t0);

const secs = (performance.now() - t0) / 1000;
writeFileSync(new URL('./index.json', import.meta.url), JSON.stringify({ corpus_sha256: createHash('sha256').update(source).digest('hex'), chunks, vecs }));
console.log(`\n\n  آماده شد — ${chunks.length.toLocaleString('fa-IR')} بردار در ${secs.toLocaleString('fa-IR', { maximumFractionDigits: 1 })} ثانیه`);
console.log(`  فقط پردازندهٔ دستگاه؛ بدون GPU و ارسال متن به سرور.\n`);

function bar(done, total, t0) {
  const frac = done / total;
  const fill = '█'.repeat(Math.round(frac * 32)).padEnd(32, '░');
  const eps = Math.round(done / ((performance.now() - t0) / 1000));
  process.stdout.write(`\r  [${fill}] ${Math.round(frac * 100).toLocaleString('fa-IR').padStart(3)}%   ${eps.toLocaleString('fa-IR').padStart(4)} بردار/ثانیه`);
}
