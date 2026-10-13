import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));

test('release preview generates notes for the public package without modifying release files', () => {
    const files = ['package.json', 'bun.lock', 'packages/base/package.json',
        'packages/mini/package.json', 'packages/mini/CHANGELOG.md'];
    const snapshot = () => files.map(file => readFileSync(resolve(root, file), 'utf8'));
    const before = snapshot();
    // Disable Git/auth checks only for this read-only preview, including on CI branches.
    const result = spawnSync('bun', ['run', 'release', '--dry-run', '--ci', '--no-git', '--npm.skipChecks'],
        { cwd: root, encoding: 'utf8', timeout: 30000 });
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0, output);
    assert.match(output, /Let's release @usex\/ternava/);
    assert.match(output, /Changelog:/);
    assert.match(output, /bun publish --tag latest --dry-run/);
    assert.deepEqual(snapshot(), before);
});
