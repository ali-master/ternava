import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

for (const [name, mutate, expected] of [
    ['outdated repository URL', dir => {
        const path = join(dir, 'package.json');
        const manifest = JSON.parse(readFileSync(path));
        manifest.repository.url = 'https://example.com/outdated';
        writeFileSync(path, JSON.stringify(manifest));
    }, /AssertionError/],
    ['missing export', dir => rmSync(join(dir, 'pkg-web/index.d.mts')), /ENOENT/],
    ['corrupt WASM', dir => writeFileSync(join(dir, 'pkg-bundler/tern_engine_bg.wasm'), 'broken'), /Invalid bundler WASM/],
]) {
    test(`prepack rejects ${name}`, () => {
        const temporary = mkdtempSync(join(tmpdir(), 'ternava-package-test-'));
        try {
            cpSync(new URL('../../packages/mini/', import.meta.url), temporary, { recursive: true });
            mutate(temporary);
            const result = spawnSync(process.execPath, ['--experimental-wasm-modules', 'scripts/check-package.mjs'], { cwd: temporary, encoding: 'utf8' });
            assert.notEqual(result.status, 0);
            assert.match(result.stderr, expected);
        } finally {
            rmSync(temporary, { recursive: true, force: true });
        }
    });
}
