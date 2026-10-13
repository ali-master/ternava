import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import init, { embed } from '../emoji-search/wasm/0.2.0/index.mjs';
import { normalizeSearchText, searchPhrases, rankEmojis } from '../emoji-search/search.js';
import { CHALLENGES } from '../emoji-search/queries.js';

const root = new URL('../', import.meta.url);
const read = path => readFileSync(new URL(path, root));
const json = path => JSON.parse(read(path));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const persian = text => typeof text === 'string' && /[\u0600-\u06ff]/u.test(text);

test('both browser demos ship the unmodified runtime from npm 0.2.0', () => {
  assert.equal(json('node_modules/@usex/ternava/package.json').version, '0.2.0');
  for (const demo of ['docs-search', 'emoji-search']) {
    for (const file of readdirSync(new URL('node_modules/@usex/ternava/pkg-web/', root))) {
      assert.deepEqual(read(`${demo}/wasm/0.2.0/${file}`), read(`node_modules/@usex/ternava/pkg-web/${file}`), `${demo}: ${file}`);
    }
  }
});

test('Persian corpora have usable unique records and no leftover English prose', () => {
  const docs = json('docs-search/chunks.json');
  assert(docs.length >= 120);
  assert.equal(new Set(docs.map(x => x.id)).size, docs.length);
  for (const item of docs) {
    assert.equal(item.lang, 'fa');
    assert(persian(item.title) && persian(item.text) && persian(item.breadcrumb));
    assert.equal(new URL(item.url).origin, 'https://en.wikipedia.org');
  }
  const emoji = json('emoji-search/corpus.fa.json');
  assert(emoji.length >= 50);
  assert.equal(new Set(emoji.map(x => x.emoji)).size, emoji.length);
  assert(emoji.every(x => persian(x.name) && persian(x.text)));
  const odyssey = json('odyssey-cli/odyssey.json');
  assert(odyssey.length >= 20 && odyssey.every(persian));
  assert.deepEqual(odyssey, read('odyssey-cli/odyssey.fa.txt').toString().trim().split(/\n\s*\n/u).map(p => p.replace(/\s+/gu, ' ').trim()));
});

test('emoji vectors belong to the exact Persian source and bundled WASM', async () => {
  const items = json('emoji-search/corpus.fa.json');
  const index = json('emoji-search/emoji-index.json');
  const wasm = read('emoji-search/wasm/0.2.0/tern_engine_bg.wasm');
  assert.equal(index.corpus_sha256, hash(read('emoji-search/corpus.fa.json')));
  assert.equal(index.wasm_sha256, hash(wasm));
  assert.equal(index.language, 'fa');
  assert.equal(index.count, items.length);
  assert.deepEqual(index.names, items.map(x => x.name));
  assert.deepEqual(index.emojis, items.map(x => x.emoji));
  assert.deepEqual(index.phrases, items.map(searchPhrases));
  await init(wasm);
  for (let i = 0; i < items.length; i++) {
    const actual = Array.from(embed(`${items[i].name}، ${items[i].text}`));
    const expected = index.vecs[i][0];
    assert.equal(expected.length, 384);
    assert(actual.every((v, j) => Number.isFinite(v) && Math.abs(v - expected[j]) < 0.000006));
    assert(Math.abs(Math.hypot(...expected) - 1) < 0.001);
  }
});

test('Persian emoji ranking handles affection variants and the other demo queries with real WASM', async () => {
  await init(read('emoji-search/wasm/0.2.0/tern_engine_bg.wasm'));
  const index = json('emoji-search/emoji-index.json');
  const cases = [
    ['خیلی دوستت دارم', '🥰'],
    ['دوستت دارم', '🥰'],
    ['من واقعاً دوستت دارم عزیزم', '🥰'],
    ['خِيلي دوستت دارم!', '🥰'],
    ['دوستت\u200cدارم', '🥰'],
    ['دوستت   دارم ❤️', '🥰'],
    ['خیلی خسته‌ام، دلم می‌خواهد بخوابم', '😴'],
    ['از خنده اشکم درآمد', '😂'],
    ['بهترینِ تمام دوران', '🏆'],
    ['بالاخره کدم کار کرد', '✅'],
    ['بدون قهوه نمی‌توانم بیدار بمانم', '☕'],
    ['مغزم سوت کشید', '🤯'],
    ['برنده شدیم، وقت جشن است', '🥳'],
    ['یک قلب قرمز برای تو', '❤️'],
  ];
  for (const [query, expected] of cases) {
    const ranked = rankEmojis(query, embed(query), index.vecs, index.phrases);
    assert.equal(index.emojis[ranked[0].i], expected, query);
  }
});

test('curated challenging queries and contrasting meanings rank correctly with the shipped WASM', async () => {
  const cases = json('tests/fixtures/emoji-challenges.fa.json');
  assert.equal(new Set(cases.map(x => x.query)).size, cases.length);
  for (const query of CHALLENGES) assert(cases.some(x => x.query === query), `Missing chip coverage: ${query}`);
  await init(read('emoji-search/wasm/0.2.0/tern_engine_bg.wasm'));
  const index = json('emoji-search/emoji-index.json');
  for (const {query, expected, category} of cases) {
    const ranked = rankEmojis(query, embed(query), index.vecs, index.phrases);
    assert.equal(index.emojis[ranked[0].i], expected, `${category}: ${query}`);
  }
});

test('aliases retain whole sentences and reject malformed data', () => {
  const item = { name: 'خواب', text: 'استراحت', aliases: ['نه چای، فقط خواب', 'خواب'] };
  assert.deepEqual(searchPhrases(item), ['خواب', 'استراحت', 'نه چای فقط خواب']);
  for (const aliases of ['عبارت', null, [42], [''], ['؟!']]) {
    assert.throws(() => searchPhrases({...item, aliases}), /عبارت‌های جایگزین/);
  }
});

test('phrase matches preserve word boundaries and negation, and cosine remains the fallback', () => {
  assert.equal(normalizeSearchText('كِتابـ‌هاي من!'), 'کتاب های من');
  const phrases = [searchPhrases({name: 'محبت', text: 'دوستت دارم'}), ['نام دیگر']];
  const vectors = [[[0, 1]], [[1, 0]]];
  assert.equal(rankEmojis('خیلی دوستت دارم', [1, 0], vectors, phrases)[0].i, 0);
  for (const query of ['دوستت ندارم', 'دوستت دارمندان', 'خیلی', 'دارم', 'unseen English query']) {
    const ranked = rankEmojis(query, [1, 0], vectors, phrases);
    assert(ranked.every(x => x.phraseScore === 0), query);
    assert.deepEqual(ranked.map(x => x.i), [1, 0], query);
  }
  assert.deepEqual(rankEmojis('  ؟! ', [1, 0], vectors, phrases), []);
});

test('corpus importer rejects an English corpus before overwriting Persian data', () => {
  const temp = mkdtempSync(join(tmpdir(), 'ternava-corpus-'));
  const before = hash(read('docs-search/chunks.json'));
  try {
    const input = join(temp, 'english.json');
    writeFileSync(input, JSON.stringify([{id: 'bad', title: 'English', text: 'English', breadcrumb: 'English', url: 'https://react.dev', lang: 'en'}]));
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('docs-search/scripts/gen-corpus.js', root)), input]);
    assert.notEqual(result.status, 0);
    assert.equal(hash(read('docs-search/chunks.json')), before);
  } finally { rmSync(temp, { recursive: true, force: true }); }
});

test('Odyssey rejects missing and stale indexes but accepts its current corpus', async () => {
  const temp = mkdtempSync(join(tmpdir(), 'ternava-odyssey-'));
  try {
    copyFileSync(new URL('odyssey-cli/load-index.mjs', root), join(temp, 'load-index.mjs'));
    const { loadIndex } = await import(pathToFileURL(join(temp, 'load-index.mjs')));
    assert.throws(loadIndex, /نمایه آماده نیست/);
    const source = JSON.stringify(['روایت فارسی']);
    writeFileSync(join(temp, 'odyssey.json'), source);
    writeFileSync(join(temp, 'index.json'), JSON.stringify({ chunks: ['Old English corpus'], vecs: [] }));
    assert.throws(loadIndex, /متن‌ها تغییر کرده‌اند/);
    const index = { corpus_sha256: hash(source), chunks: ['روایت فارسی'], vecs: [[1]] };
    writeFileSync(join(temp, 'index.json'), JSON.stringify(index));
    assert.deepEqual(loadIndex(), index);
  } finally { rmSync(temp, { recursive: true, force: true }); }
});
