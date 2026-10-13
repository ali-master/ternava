import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = new URL('../docs-search/', import.meta.url);
const corpus = JSON.parse(readFileSync(new URL('chunks.json', root)));
const provenance = JSON.parse(readFileSync(new URL('sources.wikipedia.json', root)));

test('history spans the requested eras with pinned, attributed Wikipedia sources', () => {
    assert(corpus.length >= 120);
    assert(provenance.sources.length >= 60);
    const sources = new Map(provenance.sources.map(source => [source.id, source]));
    assert.equal(sources.size, provenance.sources.length);
    assert.equal(new Set(corpus.map(row => row.id)).size, corpus.length);
    assert.equal(new Set(corpus.map(row => row.text)).size, corpus.length);
    for (const row of corpus) {
        const source = sources.get(row.source_id);
        assert(source, row.id);
        assert.equal(row.lang, 'fa');
        assert.equal(row.license, 'CC-BY-SA-4.0');
        assert.equal(row.url, source.revision_url);
        assert.equal(new URL(row.url).origin, 'https://en.wikipedia.org');
        assert.equal(new URL(row.url).searchParams.get('oldid'), String(source.revision_id));
        assert.equal(new URL(source.history_url).searchParams.get('action'), 'history');
        assert(source.attribution && source.retrieved_on);
        assert(Number.isInteger(row.year_start) && Number.isInteger(row.year_end));
        assert(row.year_start >= -550 && row.year_end <= 1977 && row.year_start <= row.year_end, row.id);
        assert(row.text.startsWith(`${row.breadcrumb}. `));
        assert(!/react|useEffect|کامپوننت|انقلاب اسلامی|جمهوری اسلامی|۱۹۷۹|۱۳۵۸/iu.test(row.text), row.id);
        assert(!('aliases' in row));
    }
    for (const era of ['هخامنشیان', 'سلوکیان', 'اشکانیان', 'ساسانیان', 'طاهریان', 'صفاریان', 'سامانیان', 'آل‌بویه', 'غزنویان', 'سلجوقیان', 'خوارزمشاهیان', 'ایلخانان', 'تیموریان', 'صفویان', 'افشاریان', 'زندیان', 'قاجار', 'پهلوی']) {
        assert(corpus.some(row => row.period.includes(era)), era);
    }
    const html = readFileSync(new URL('index.html', root), 'utf8');
    assert(!/React|react\.dev|کامپوننت/u.test(html));
    assert(html.includes('sources.wikipedia.json') && html.includes('creativecommons.org/licenses/by-sa/4.0/'));
});

test('historical importer rejects invalid dates and attribution before writing', () => {
    const temp = mkdtempSync(join(tmpdir(), 'tern-history-'));
    try {
        mkdirSync(join(temp, 'scripts'));
        copyFileSync(new URL('scripts/gen-corpus.js', root), join(temp, 'scripts/gen-corpus.js'));
        copyFileSync(new URL('sources.wikipedia.json', root), join(temp, 'sources.wikipedia.json'));
        const output = join(temp, 'chunks.json');
        const input = join(temp, 'input.json');
        writeFileSync(output, 'unchanged');
        for (const patch of [{ year_end: 1978 }, { year_start: -551 }, { year_end: -999 },
            { year_start: '1900' }, { source_id: 'missing' }, { url: 'https://example.com' }, { license: 'MIT' }]) {
            writeFileSync(input, JSON.stringify([{ ...corpus[0], ...patch }]));
            const result = spawnSync(process.execPath, [join(temp, 'scripts/gen-corpus.js'), input]);
            assert.notEqual(result.status, 0, JSON.stringify(patch));
            assert.equal(readFileSync(output, 'utf8'), 'unchanged');
        }
        writeFileSync(input, JSON.stringify(corpus));
        const result = spawnSync(process.execPath, [join(temp, 'scripts/gen-corpus.js'), input]);
        assert.equal(result.status, 0, result.stderr.toString());
        assert.deepEqual(JSON.parse(readFileSync(output)), corpus);
    } finally { rmSync(temp, { recursive: true, force: true }); }
});

test('the original React supervision stays byte-identical outside the demo', () => {
    const previous = readFileSync(new URL('../../eval/quality/corpus/react-docs.fa.json', import.meta.url));
    assert.equal(createHash('sha256').update(previous).digest('hex'), '8bf0b8e78a4e5b446a3536bc60bc92fcd54914236a6dcf022b07bc2aebc20a22');
    assert.equal(JSON.parse(previous).length, 30);
});
