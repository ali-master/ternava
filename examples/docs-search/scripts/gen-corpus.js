// Validate the complete historical corpus before replacing the public demo data.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const source = process.argv[2];
assert(source, 'روش اجرا: node examples/docs-search/scripts/gen-corpus.js /path/to/persian-corpus.json');
const chunks = JSON.parse(fs.readFileSync(source, 'utf8'));
const provenance = JSON.parse(fs.readFileSync(path.join(__dirname, '../sources.wikipedia.json'), 'utf8'));
const sources = new Map(provenance.sources.map(s => [s.id, s]));
assert(Array.isArray(chunks) && chunks.length > 0, 'مجموعهٔ متن خالی است');
assert(new Set(chunks.map(c => c.id)).size === chunks.length, 'شناسه‌ها باید یکتا باشند');
for (const c of chunks) {
  assert(typeof c.id === 'string' && c.id, 'شناسه لازم است');
  for (const key of ['title', 'breadcrumb', 'text']) assert(typeof c[key] === 'string' && /[\u0600-\u06ff]/u.test(c[key]), `متن فارسی لازم است: ${key}`);
  assert(c.lang === 'fa' && new URL(c.url).protocol === 'https:', 'زبان یا نشانی منبع نامعتبر است');
  assert(typeof c.period === 'string' && c.period.length > 0, 'دورهٔ تاریخی لازم است');
  assert(Number.isInteger(c.year_start) && Number.isInteger(c.year_end)
    && c.year_start >= -550 && c.year_start <= c.year_end && c.year_end <= 1977,
  'بازهٔ تاریخی باید از هخامنشیان تا پیش از سال ۱۳۵۷ باشد');
  const origin = sources.get(c.source_id);
  assert(origin && c.url === origin.revision_url && c.license === 'CC-BY-SA-4.0', 'ارجاع به نسخهٔ منبع و مجوز لازم است');
}
fs.writeFileSync(path.join(__dirname, '../chunks.json'), JSON.stringify(chunks, null, 2) + '\n');
console.log(`${chunks.length.toLocaleString('fa-IR')} متن فارسی آماده شد.`);
