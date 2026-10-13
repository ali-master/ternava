// probe.mjs — fire a battery of candidate demo queries, show top hit for each.
import { embed, cosineSim } from '@usex/ternava';
import { loadIndex } from './load-index.mjs';
const { chunks, vecs } = loadIndex();

const QUERIES = [
  "همسری که سال‌ها چشم‌به‌راه بازگشت شوهرش است",
  "غول یک‌چشم مردان را در غار زندانی می‌کند",
  "آواز فریبنده دریانوردان را به سوی صخره‌ها می‌کشاند",
  "بازگشت به خانه در لباس گدایی ناشناس",
  "جادوگر همراهان را به خوک تبدیل می‌کند",
  "کشیدن زه کمانی که هیچ‌کس نمی‌تواند آماده‌اش کند",
  "مرا به دکل ببندید تا آواز را بشنوم و زنده بمانم",
  "الهه‌ای که قهرمان را در جزیره نگه داشته است"
];

console.log(`\n  مجموعه: ${chunks.length.toLocaleString('fa-IR')} گزیدهٔ فارسی\n`);
for (const query of QUERIES) {
  const q = embed(query);
  let best = { score: -1, text: '' };
  for (let i = 0; i < vecs.length; i++) {
    const s = cosineSim(q, vecs[i]);
    if (s > best.score) best = { score: s, text: chunks[i] };
  }
  console.log(`▸ ${query}`);
  console.log(`  ${best.score.toLocaleString('fa-IR', { minimumFractionDigits: 3, maximumFractionDigits: 3 })}  ${best.text.slice(0, 150)}${best.text.length > 150 ? '…' : ''}\n`);
}
