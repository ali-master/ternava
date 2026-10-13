import test from 'node:test';
import assert from 'node:assert/strict';
import { appendFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

for (const [name, mutate, expected] of [
    ['outdated repository URL', dir => {
        const path = join(dir, 'package.json');
        const manifest = JSON.parse(readFileSync(path));
        manifest.repository.url = 'https://example.com/outdated';
        writeFileSync(path, JSON.stringify(manifest));
    }, /AssertionError/],
    ['missing export', dir => rmSync(join(dir, 'pkg-web/index.d.mts')), /ENOENT/],
    ['corrupt WASM', dir => writeFileSync(join(dir, 'pkg-bundler/tern_engine_bg.wasm'), 'broken'), /Invalid bundler WASM/],
    ['oversized package', dir => appendFileSync(join(dir, 'pkg-node/tern_engine_bg.wasm'), Buffer.alloc(1_000_000)), /exceeds 25 MB unpacked budget/],
]) {
    test(`prepack rejects ${name}`, () => {
        const temporary = mkdtempSync(join(tmpdir(), 'ternava-package-test-'));
        try {
            cpSync(new URL('../../packages/mini/', import.meta.url), temporary, { recursive: true });
            mutate(temporary);
            const tarball = join(temporary, 'rejected.tgz');
            const result = spawnSync('bun', ['pm', 'pack', '--filename', tarball], { cwd: temporary, encoding: 'utf8' });
            assert.notEqual(result.status, 0);
            assert.match(result.stderr, expected);
            assert.equal(existsSync(tarball), false, 'Invalid package was packed');
        } finally {
            rmSync(temporary, { recursive: true, force: true });
        }
    });
}

test('Bun and npm pack only runtime files despite development artifacts', () => {
    const temporary = mkdtempSync(join(tmpdir(), 'ternava-pack-content-'));
    try {
        const cwd = join(temporary, 'mini');
        cpSync(new URL('../../packages/mini/', import.meta.url), cwd, { recursive: true });
        for (const path of ['.env.local', 'tests/leak.js', 'training/model.bin', 'examples/data.json',
            'src/.env', 'src/debug.js', 'src/debug.test.js', 'src/index.js.map', 'pkg-web/debug.log',
            'pkg-web/model-backup.wasm', 'pkg-web/runtime-parity.json', 'old-release.tgz']) {
            const file = join(cwd, path);
            mkdirSync(dirname(file), { recursive: true });
            writeFileSync(file, 'fixture: must not ship');
        }
        const { files } = JSON.parse(readFileSync(join(cwd, 'package.json')));
        const expected = [...files, 'package.json'].sort();
        assert.ok(!expected.includes('pkg-node/tern_engine_bg.js'));
        assert.ok(!expected.includes('pkg-web/tern_engine_bg.js'));
        assert.ok(expected.includes('pkg-bundler/tern_engine_bg.js'));
        execFileSync('bun', ['pm', 'pack', '--filename', join(temporary, 'bun.tgz'), '--quiet'], { cwd, stdio: 'pipe' });
        const bunFiles = execFileSync('tar', ['-tzf', join(temporary, 'bun.tgz')], { encoding: 'utf8' })
            .trim().split('\n').filter(path => !path.endsWith('/')).map(path => path.replace(/^package\//, '')).sort();
        assert.deepEqual(bunFiles, expected, 'Bun archive contents');
        const [npmPack] = JSON.parse(execFileSync('npm', ['pack', '--json', '--pack-destination', temporary], { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }));
        assert.deepEqual(npmPack.files.map(({ path }) => path).sort(), expected, 'npm archive contents');
    } finally {
        rmSync(temporary, { recursive: true, force: true });
    }
});
