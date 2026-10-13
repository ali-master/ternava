import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';

const root = new URL('../', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('package.json', root)));
assert.equal(manifest.name, '@usex/ternava');
assert.equal(manifest.repository.url, 'https://github.com/ali-master/ternava.git');
assert.equal(manifest.homepage, 'https://github.com/ali-master/ternava');
assert.equal(manifest.bugs.url, 'https://github.com/ali-master/ternava/issues');
assert.equal(manifest.publishConfig.access, 'public');
assert.notEqual(manifest.private, true);
function checkExports(value) {
    if (typeof value === 'string') {
        assert.ok(statSync(new URL(value, root)).isFile(), `Missing export: ${value}`);
    } else {
        Object.values(value).forEach(checkExports);
    }
}
checkExports(manifest.exports);
for (const target of ['node', 'bundler', 'web']) {
    const bytes = readFileSync(new URL(`pkg-${target}/tern_engine_bg.wasm`, root));
    assert.ok(WebAssembly.validate(bytes), `Invalid ${target} WASM`);
}
const node = createRequire(import.meta.url)('../src/index.js');
assert.match(node.engineInfo(), /embedding_format=int4/);
assert.match(node.engineInfo(), /vocab=32000 d_model=256 n_layers=2 n_heads=4 ffn_dim=1024 output_dim=384 max_seq_len=128/);
const web = await import(new URL('pkg-web/index.mjs', root));
await web.init();
const browser = await import(new URL('src/index.browser.mjs', root));
assert.equal(browser.engineInfo(), node.engineInfo());
assert.equal(web.engineInfo(), node.engineInfo());
for (const text of ['خیلی دوستت دارم', 'هیچ سفارشی بدون پرداخت ارسال نمی‌شود.', '۱۲ بسته و ۵ بسته', 'Reset my password']) {
    const vector = node.embed(text);
    assert.equal(vector.length, 384);
    assert.ok(vector.every(Number.isFinite));
    assert.deepEqual(browser.embed(text), vector, 'Bundler model/tokenizer differs');
    assert.deepEqual(web.embed(text), vector, 'Node and web must embed the same model/tokenizer');
}
