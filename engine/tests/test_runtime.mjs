// Run with either node or bun; uses the actual built WASM, without aliases.
import assert from 'node:assert/strict';
import { mkdtemp, copyFile, readFile, writeFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';

const artifact = resolve(process.argv[2] ?? 'dist/pilot-fa-en-reasoning/d256');
const bytes = await readFile(join(artifact, 'tern_engine_bg.wasm'));
const temporary = await mkdtemp(join(tmpdir(), 'tern-runtime-'));
const originalFetch = globalThis.fetch;
const texts = ['خیلی دوستت دارم', 'بهترینِ تمام دوران', 'هیچ سفارشی بدون پرداخت ارسال نمی‌شود.',
    'از ۱۲ بسته، ۵ بسته باقی مانده است.', 'How do I reset my password?', 'The weather is sunny today', ''];
let server;

async function fixture(name) {
    const dir = join(temporary, name);
    await mkdir(dir, { recursive: true });
    for (const file of ['index.mjs', 'index.d.mts', 'tern_engine.js', 'package.json']) {
        await copyFile(join(artifact, file), join(dir, file));
    }
    return { dir, runtime: await import(pathToFileURL(join(dir, 'index.mjs'))) };
}

try {
    // Install before importing consumers: Bun caches directory resolution during a run.
    for (const tier of ['mini', 'base']) {
        const manifest = JSON.parse(await readFile(new URL(`../../packages/${tier}/package.json`, import.meta.url)));
        const packageDir = join(temporary, 'node_modules', manifest.name);
        await mkdir(join(packageDir, 'pkg-web'), { recursive: true });
        await copyFile(new URL(`../../packages/${tier}/package.json`, import.meta.url), join(packageDir, 'package.json'));
        for (const file of ['index.mjs', 'index.d.mts', 'tern_engine.js', 'tern_engine_bg.wasm', 'package.json']) {
            await copyFile(join(artifact, file), join(packageDir, 'pkg-web', file));
        }
        await writeFile(join(temporary, `consumer-${tier}.mjs`), `export * from '${manifest.name}/runtime';`);
    }
    const { dir, runtime } = await fixture('default');
    const { init, embed, cosineSim, similar, engineInfo, TernError } = runtime;
    assert.equal(runtime.default, init);
    assert.throws(() => embed('سلام'), { code: 'NOT_INITIALIZED' });
    assert.throws(() => engineInfo(), { code: 'NOT_INITIALIZED' });
    assert.throws(() => embed(42), error => error instanceof TernError && error.code === 'INVALID_INPUT');
    globalThis.fetch = () => { throw new Error('Local loading must not fetch'); };
    await assert.rejects(init(), { code: 'ENOENT' });
    await copyFile(join(artifact, 'tern_engine_bg.wasm'), join(dir, 'tern_engine_bg.wasm'));
    const pending = init();
    assert.equal(init(), pending, 'concurrent init must share one promise');
    await pending;
    assert.equal(init(), pending, 'successful init must remain idempotent');
    const vectors = texts.map(text => Array.from(embed(text)));
    for (let i = 0; i < texts.length; i++) {
        const vector = embed(texts[i]);
        assert.ok(vector instanceof Float32Array);
        assert.equal(vector.length, 384);
        assert.ok(vector.every(Number.isFinite));
        assert.ok(Math.abs(cosineSim(vector, vector) - 1) < 1e-5);
        assert.deepEqual(Array.from(vector), vectors[i]);
    }
    assert.throws(() => cosineSim([1], [1, 2]), { code: 'DIM_MISMATCH' });
    assert.deepEqual(similar(texts[0], []), []);
    assert.equal(similar(texts[0], texts).length, 5);
    assert.equal(similar(texts[0], texts, { topK: 1 })[0].text, texts[0]);
    assert.deepEqual(similar(texts[0], texts, { topK: 0 }), []);
    const ranked = similar(texts[0], texts);
    assert.ok(ranked.every((item, i) => i === 0 || ranked[i - 1].sim >= item.sim));

    // Compare every float against the pre-existing raw engine API.
    const raw = await import(pathToFileURL(join(artifact, 'tern_engine.js')));
    await raw.default({ module_or_path: bytes });
    assert.equal(engineInfo(), raw.config_summary());
    texts.forEach((text, i) => assert.deepEqual(Array.from(raw.embed(text)), vectors[i]));

    // Respect byte offsets when callers provide a view, and permit retry after invalid bytes.
    const fromBytes = (await fixture('bytes')).runtime;
    await assert.rejects(fromBytes.init(new Uint8Array([0, 1, 2])));
    const padded = new Uint8Array(bytes.length + 16);
    padded.set(bytes, 8);
    await fromBytes.init(padded.subarray(8, 8 + bytes.length));
    assert.deepEqual(Array.from(fromBytes.embed(texts[0])), vectors[0]);
    const fromModule = (await fixture('compiled')).runtime;
    await fromModule.init(await WebAssembly.compile(bytes));
    assert.deepEqual(Array.from(fromModule.embed(texts[1])), vectors[1]);
    const fromURL = (await fixture('file-url')).runtime;
    await fromURL.init(pathToFileURL(join(artifact, 'tern_engine_bg.wasm')));
    assert.deepEqual(Array.from(fromURL.embed(texts[2])), vectors[2]);

    globalThis.fetch = originalFetch;
    server = createServer((request, response) => {
        response.writeHead(request.url === '/model.wasm' ? 200 : 404, { 'Content-Type': 'application/wasm' });
        response.end(request.url === '/model.wasm' ? bytes : 'missing');
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    const fromHTTP = (await fixture('http')).runtime;
    await assert.rejects(fromHTTP.init(new URL('/missing', origin)));
    await fromHTTP.init(new URL('/model.wasm', origin));
    assert.deepEqual(Array.from(fromHTTP.embed(texts[3])), vectors[3]);

    // Exercise both package export maps from an isolated consumer installation.
    for (const tier of ['mini', 'base']) {
        const consumer = join(temporary, `consumer-${tier}.mjs`);
        const packaged = await import(pathToFileURL(consumer));
        await packaged.init();
        assert.deepEqual(Array.from(packaged.embed(texts[0])), vectors[0]);
    }
    const parity = { texts, vectors, wasmSHA256: createHash('sha256').update(bytes).digest('hex'), config: engineInfo() };
    // Optional output is also consumed by the real-browser smoke page.
    if (process.argv[3]) await writeFile(resolve(process.argv[3]), JSON.stringify(parity));
    console.log(JSON.stringify({ status: 'passed', runtime: process.versions.bun ? `Bun ${process.versions.bun}` : `Node ${process.version}`,
        texts: texts.length, dimensions: 384, wasmBytes: bytes.length, wasmSHA256: parity.wasmSHA256 }));
} finally {
    globalThis.fetch = originalFetch;
    if (server) await new Promise(resolve => server.close(resolve));
    await rm(temporary, { recursive: true, force: true });
}
