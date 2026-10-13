// search.mjs — semantic search over the indexed Odyssey.  `node search.mjs`
import { embed, cosineSim } from '@usex/ternava';
import { loadIndex } from './load-index.mjs';

const { chunks, vecs } = loadIndex();
const query = process.argv.slice(2).join(' ').trim() || 'دلتنگی برای خانه و خانواده';
console.log('ترنوا — پایلوت دوزبانهٔ آزمایشی؛ کیفیت عمومی فارسی هنوز تأیید نشده است.');

const q = embed(query);
const hits = chunks
  .map((text, i) => ({ text, score: cosineSim(q, vecs[i]) }))   // cosine — no vector DB
  .sort((a, b) => b.score - a.score)
  .slice(0, 3);

console.log(`\n  "${query}"\n`);
for (const { text, score } of hits) console.log(`  ${score.toLocaleString('fa-IR', { minimumFractionDigits: 3, maximumFractionDigits: 3 })}  ${text}\n`);
