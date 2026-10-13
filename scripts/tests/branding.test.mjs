import test from 'node:test';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const json = path => JSON.parse(readFileSync(resolve(root, path), 'utf8'));
const legacy = 'ternlight';
const persianLegacy = 'ترن‌لایت';
const obsolete = new RegExp(`${legacy}|ترن[‌ ]?لایت`, 'i');
const credit = `**Ternava is built on [${legacy[0].toUpperCase() + legacy.slice(1)}](https://github.com/soycaporal/${legacy}) by`;
const allowed = {
    'README.md': [
        credit,
        `[surfacing the Vite setup](https://github.com/soycaporal/${legacy}/issues/3)`,
        `[English model artifacts](https://huggingface.co/wenshutang/${legacy})`,
    ],
    'scripts/tests/branding.test.mjs': [`const legacy = '${legacy}';`, `const persianLegacy = '${persianLegacy}';`],
};

function hasObsoleteBrand(file, source) {
    // Each exact reference is allowed once, in its designated file only.
    for (const reference of allowed[file] ?? []) source = source.replace(reference, '');
    return obsolete.test(file) || obsolete.test(source);
}

test('one public package with consistent names, repository metadata and upstream credit', () => {
    const mini = json('packages/mini/package.json');
    const base = json('packages/base/package.json');
    const workspace = json('package.json');
    assert.equal(mini.name, '@usex/ternava');
    assert.equal(mini.publishConfig.access, 'public');
    assert.notEqual(mini.private, true);
    assert.equal(base.name, '@usex/ternava-base');
    assert.equal(base.private, true);
    assert.equal(workspace.private, true);
    for (const manifest of [mini, base, workspace]) {
        assert.equal(manifest.repository.url, 'https://github.com/ali-master/ternava.git');
    }
    for (const manifest of [mini, base]) {
        assert.equal(manifest.homepage, 'https://github.com/ali-master/ternava');
        assert.equal(manifest.bugs.url, 'https://github.com/ali-master/ternava/issues');
    }
    assert.equal(json('examples/odyssey-cli/package.json').dependencies[mini.name], 'file:../../packages/mini');
    const readme = readFileSync(resolve(root, 'README.md'), 'utf8');
    assert.ok(readme.slice(0, 700).includes(credit));
    assert.match(readme.slice(0, 700), /Thank you/);
    assert.match(readFileSync(resolve(root, 'packages/mini/LICENSE'), 'utf8'), /Wenshu Tang/);
});

test('owned sources, including historical results and the model card, contain no obsolete branding', () => {
    const files = new Set(execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { cwd: root, encoding: 'utf8' }).split('\0'));
    const problems = [];
    for (const file of files) {
        if (!file || !existsSync(resolve(root, file))) continue;
        const bytes = readFileSync(resolve(root, file));
        if (bytes.includes(0)) {
            if (obsolete.test(file)) problems.push(file);
            continue;
        }
        if (hasObsoleteBrand(file, bytes.toString('utf8'))) problems.push(file);
    }
    assert.deepEqual(problems, []);
});

test('historical files, comments, casing and attribution wording cannot bypass the guard', () => {
    for (const file of ['models/huggingface/README.md', 'eval/quality/RESULTS.md', 'eval/results/quality.json', 'src/example.js']) {
        for (const source of [legacy, legacy.toUpperCase(), `@${legacy}/mini`, `upstream ${legacy}`,
            `built on ${legacy}`, `Wenshu Tang ${legacy}`, `https://github.com/ali-master/${legacy}`, persianLegacy]) {
            assert.equal(hasObsoleteBrand(file, source), true, `${file}: ${source}`);
        }
    }
    assert.equal(hasObsoleteBrand(`examples/${legacy}.svg`, '<svg/>'), true);
});

test('credit exceptions are exact, single-use, and restricted to README', () => {
    for (const reference of allowed['README.md']) {
        assert.equal(hasObsoleteBrand('README.md', reference), false);
        assert.equal(hasObsoleteBrand('README.md', `${reference}\n${reference}`), true);
        assert.equal(hasObsoleteBrand('README.md', `${reference}\n${legacy}`), true);
        assert.equal(hasObsoleteBrand('models/huggingface/README.md', reference), true);
        assert.equal(hasObsoleteBrand('README.md', reference.replace(legacy, legacy.toUpperCase())), true);
    }
});

// PDF text, raster labels and QR destination were visually checked; recheck before changing these snapshots.
test('reviewed PDF, chart and QR assets cannot silently revert to old branding', () => {
    const reviewed = {
        'ternava-overview.pdf': '844928791e16ea488f3e54c1537d69b97f9902a42c6c3502404575f24585eca7',
        'docs/assets/english-size-quality.png': '9672ec6c60a27486c032a4fe8ef77e9eae4a77b6773d664487206ae9fd020875',
        'examples/emoji-search/qr.svg': '3cc345d596bf27f6c01ca64aead1039fff031a88a0265ec19be4c6f2edd94a10',
    };
    for (const [file, hash] of Object.entries(reviewed)) {
        assert.equal(createHash('sha256').update(readFileSync(resolve(root, file))).digest('hex'), hash,
            `Brand asset changed; inspect its text, rendered labels or QR destination before updating: ${file}`);
    }
});
