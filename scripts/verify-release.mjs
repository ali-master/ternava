// Install and exercise the exact tarball, retaining the consumer for browser QA.
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, writeFileSync, readFileSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(root, 'dist/release');
mkdirSync(output, { recursive: true });
const run = (command, args, cwd = root) => execFileSync(command, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
console.log(run('node', ['--test', 'scripts/tests/branding.test.mjs']).trim());
const manifest = JSON.parse(readFileSync(join(root, 'packages/mini/package.json')));
const filename = `${manifest.name.replace('@', '').replace('/', '-')}-${manifest.version}.tgz`;
const tarball = join(output, filename);
run('bun', ['pm', 'pack', '--filename', tarball, '--quiet'], join(root, 'packages/mini'));
const archivePaths = run('tar', ['-tzf', tarball]).trim().split('\n').filter(path => !path.endsWith('/'));
assert.ok(archivePaths.every(path => path.startsWith('package/') && !path.split('/').includes('..')));
const packed = { name: manifest.name, version: manifest.version, filename,
    files: archivePaths.map(path => ({ path: path.slice('package/'.length) })),
    size: statSync(tarball).size,
    integrity: `sha512-${createHash('sha512').update(readFileSync(tarball)).digest('base64')}` };
assert.equal(new Set(archivePaths).size, archivePaths.length, 'Duplicate package entries');
assert.equal(packed.name, '@usex/ternava');
assert.deepEqual(packed.files.map(({ path }) => path).sort(), [...manifest.files, 'package.json'].sort(),
    'Archive must contain exactly the publish allowlist');
assert.ok(packed.size <= 17_000_000, `Package exceeds 17 MB compressed budget: ${packed.size} bytes`);
const consumer = mkdtempSync(join(output, 'consumer-'));
writeFileSync(join(consumer, 'package.json'), JSON.stringify({ name: 'ternava-release-consumer', private: true, type: 'module' }));
run('bun', ['add', '--ignore-scripts', '--no-cache', tarball], consumer);
packed.unpackedSize = 0;
for (const { path } of packed.files) {
    packed.unpackedSize += statSync(join(consumer, 'node_modules/@usex/ternava', path)).size;
    assert.deepEqual(readFileSync(join(consumer, 'node_modules/@usex/ternava', path)),
        readFileSync(join(root, 'packages/mini', path)), `Installed package differs: ${path}`);
}
assert.ok(packed.unpackedSize <= 25_000_000, `Package exceeds 25 MB unpacked budget: ${packed.unpackedSize} bytes`);
writeFileSync(join(consumer, 'smoke.cjs'), `const assert = require('node:assert/strict');
const api = require('@usex/ternava');
assert.equal(api.embed('سلام دنیا').length, 384);
assert.match(api.engineInfo(), /vocab=32000 d_model=256/);
assert.throws(() => api.embed(null), { code: 'INVALID_INPUT' });
console.log('CommonJS passed');
`);
writeFileSync(join(consumer, 'smoke.mjs'), `import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { embed, similar, engineInfo } from '@usex/ternava';
import init, * as runtime from '@usex/ternava/runtime';
import rawInit, * as raw from '@usex/ternava/web';
await init();
await rawInit({ module_or_path: readFileSync(new URL('./node_modules/@usex/ternava/pkg-web/tern_engine_bg.wasm', import.meta.url)) });
assert.equal(runtime.engineInfo(), engineInfo());
for (const text of ['خیلی دوستت دارم', 'بهترینِ تمام دوران', 'هیچ سفارشی بدون پرداخت ارسال نمی‌شود.', 'از ۱۲ بسته، ۵ بسته باقی مانده است.', 'How do I reset my password?', 'The weather is sunny today', '']) {
  const v = embed(text);
  assert.equal(v.length, 384);
  assert.ok(v.every(Number.isFinite));
  assert.ok(Math.abs(v.reduce((sum, x) => sum + x*x, 0) - 1) < 1e-5);
  assert.deepEqual(runtime.embed(text), v);
  assert.deepEqual(raw.embed(text), v);
}
assert.equal(similar('سلام', ['سلام', 'خداحافظ'], { topK: 1 })[0].text, 'سلام');
console.log('ESM, runtime and raw web passed');
`);
writeFileSync(join(consumer, 'smoke.bundler.mjs'), `import assert from 'node:assert/strict';
import * as browser from './node_modules/@usex/ternava/src/index.browser.mjs';
import * as node from '@usex/ternava';
assert.equal(browser.engineInfo(), node.engineInfo());
for (const text of ['خیلی دوستت دارم', '۱۲ بسته و ۵ بسته', 'Reset my password']) {
  assert.deepEqual(browser.embed(text), node.embed(text));
}
console.log('Browser entry / bundler WASM graph passed in Node ESM integration');
`);
for (const executable of ['node', 'bun']) {
    for (const smoke of ['smoke.cjs', 'smoke.mjs']) console.log(executable, run(executable, [smoke], consumer).trim());
}
console.log(run('node', ['--experimental-wasm-modules', 'smoke.bundler.mjs'], consumer).trim());
const web = join(consumer, 'node_modules/@usex/ternava/pkg-web');
console.log(run('node', ['engine/tests/test_runtime.mjs', web, join(web, 'runtime-parity.json')]).trim());
const report = { name: packed.name, version: packed.version, tarball: relative(root, join(output, packed.filename)),
    compressedBytes: packed.size, unpackedBytes: packed.unpackedSize, integrity: packed.integrity,
    browserURL: `http://127.0.0.1:8877/engine/tests/runtime.html?artifact=/${relative(root, web)}/`,
    browserStatus: 'manual check required',
    parity: JSON.parse(readFileSync(join(web, 'runtime-parity.json'))).wasmSHA256 };
writeFileSync(join(output, 'verification.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
