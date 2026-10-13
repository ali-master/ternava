import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('../../', import.meta.url));

test('Bun frozen install links local workspaces and rejects incompatible versions', () => {
    const directory = mkdtempSync(join(tmpdir(), 'ternava-workspace-'));
    try {
        for (const file of ['package.json', 'bun.lock', 'packages/mini/package.json',
            'packages/base/package.json', 'examples/emoji-search/package.json',
            'examples/odyssey-cli/package.json']) {
            cpSync(join(root, file), join(directory, file), { recursive: true });
        }
        const lock = readFileSync(join(directory, 'bun.lock'));
        const install = () => spawnSync('bun', ['install', '--frozen-lockfile', '--offline', '--ignore-scripts'],
            { cwd: directory, encoding: 'utf8' });
        const result = install();
        assert.equal(result.status, 0, result.stdout + result.stderr);
        assert.deepEqual(readFileSync(join(directory, 'bun.lock')), lock);
        const consumer = join(directory, 'examples/odyssey-cli');
        const resolved = spawnSync('node', ['-p', "require.resolve('@usex/ternava/package.json')"],
            { cwd: consumer, encoding: 'utf8' });
        assert.equal(resolved.status, 0, resolved.stderr);
        assert.equal(realpathSync(resolved.stdout.trim()), realpathSync(join(directory, 'packages/mini/package.json')));
        const file = join(consumer, 'package.json');
        const manifest = JSON.parse(readFileSync(file));
        manifest.dependencies['@usex/ternava'] = 'workspace:9.9.9';
        writeFileSync(file, JSON.stringify(manifest));
        rmSync(join(directory, 'node_modules'), { recursive: true, force: true });
        const drift = install();
        assert.notEqual(drift.status, 0, 'Install accepted an incompatible local workspace version');
        assert.match(drift.stdout + drift.stderr, /(?:workspace|version).*9\.9\.9/i);
        assert.deepEqual(readFileSync(join(directory, 'bun.lock')), lock);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});
