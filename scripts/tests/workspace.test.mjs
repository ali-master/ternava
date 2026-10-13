import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('../../', import.meta.url));

test('examples install the pinned npm release independently of local workspaces', () => {
    const directory = mkdtempSync(join(tmpdir(), 'ternava-workspace-'));
    try {
        for (const file of ['package.json', 'bun.lock', 'packages/mini/package.json',
            'packages/base/package.json', 'examples/package.json', 'examples/bun.lock', 'examples/emoji-search/package.json',
            'examples/odyssey-cli/package.json']) {
            cpSync(join(root, file), join(directory, file), { recursive: true });
        }
        const examples = join(directory, 'examples');
        const lock = readFileSync(join(examples, 'bun.lock'));
        const install = () => spawnSync('bun', ['install', '--frozen-lockfile', '--offline', '--ignore-scripts'],
            { cwd: examples, encoding: 'utf8' });
        const result = install();
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.deepEqual(readFileSync(join(examples, 'bun.lock')), lock);
        const consumer = join(directory, 'examples/odyssey-cli');
        const resolved = spawnSync('node', ['-p', "require.resolve('@usex/ternava/package.json')"],
            { cwd: consumer, encoding: 'utf8' });
        assert.equal(resolved.status, 0, resolved.stderr);
        assert.equal(realpathSync(resolved.stdout.trim()), realpathSync(join(examples, 'node_modules/@usex/ternava/package.json')));
        assert.equal(JSON.parse(readFileSync(resolved.stdout.trim())).version, '0.2.0');
        assert.notEqual(realpathSync(resolved.stdout.trim()), realpathSync(join(directory, 'packages/mini/package.json')));
        const file = join(examples, 'package.json');
        const manifest = JSON.parse(readFileSync(file));
        manifest.dependencies['@usex/ternava'] = '0.1.0';
        writeFileSync(file, JSON.stringify(manifest));
        rmSync(join(examples, 'node_modules'), { recursive: true, force: true });
        const drift = install();
        assert.notEqual(drift.status, 0, 'Install accepted a version that differs from the frozen lockfile');
        assert.deepEqual(readFileSync(join(examples, 'bun.lock')), lock);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});
