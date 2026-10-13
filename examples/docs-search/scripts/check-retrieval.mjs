// Report actual cosine ranks; low retrieval quality must not be hidden by structural tests.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import init, { embed } from '../wasm/0.2.0/index.mjs';

const source = await readFile(new URL('../chunks.json', import.meta.url));
const corpus = JSON.parse(source);
const wasm = await readFile(new URL('../wasm/0.2.0/tern_engine_bg.wasm', import.meta.url));
await init(wasm);
const vectors = corpus.map(row => embed(row.text));
for (const vector of vectors) {
    assert.equal(vector.length, 384);
    assert(vector.every(Number.isFinite));
    assert(Math.abs(Math.hypot(...vector) - 1) < 1e-5);
}
const checks = [
    ['راه شاهی برای چی ساخته شد؟', ['راه شاهی؛ ارتباط شوش و سارد', 'چاپارها چطور خبر می‌رسوندند؟']],
    ['خیام غیر از شعر چه کارهایی می‌کرد؟', ['خیام فقط شاعر نبود', 'خیام و تقویم جلالی']],
    ['دور میدان نقش‌جهان چه بناهایی ساخته شد؟', ['دور میدان نقش‌جهان چه بناهایی هست؟', 'نقش‌جهان؛ یک میدان و چند کارکرد']],
    ['امیرکبیر دارالفنون رو برای چی راه انداخت؟', ['دارالفنون برای چه ساخته شد؟', 'دارالفنون و آموزش تخصصی']],
    ['راه‌آهن سراسری شمال رو چطور به جنوب وصل کرد؟', ['راه‌آهن سراسری شمال و جنوب']],
    ['شاهنامه رو کی نوشته؟', ['فردوسی؛ شاعر توس', 'فردوسی از چه میراثی استفاده کرد؟']],
    ['کدوم پادشاه ساسانی امپراتور روم رو اسیر کرد؟', ['شاپور یکم و اسارت والرین']],
    ['مجلس چطور قرار بود قدرت پادشاه رو محدود کنه؟', ['مشروطه یعنی محدود شدن قدرت شاه', 'مجلس و کشمکش مشروطه']],
];
const results = checks.map(([query, expected]) => {
    assert(expected.every(title => corpus.some(row => row.title === title)));
    const q = embed(query);
    const ranked = vectors.map((vector, i) => ({ title: corpus[i].title,
        score: vector.reduce((sum, value, j) => sum + value * q[j], 0) }))
        .sort((a, b) => b.score - a.score);
    const rank = ranked.findIndex(row => expected.includes(row.title)) + 1;
    return { query, expected, rank, top1: ranked[0].title, pass_at_8: rank <= 8 };
});
console.log(JSON.stringify({ model: '@usex/ternava@0.2.0; bilingual mini', aliases: false,
    corpus_sha256: createHash('sha256').update(source).digest('hex'),
    wasm_sha256: createHash('sha256').update(wasm).digest('hex'),
    corpus_count: corpus.length, top1: results.filter(row => row.rank === 1).length,
    hit_at_8: results.filter(row => row.pass_at_8).length, total: results.length, results }, null, 2));
