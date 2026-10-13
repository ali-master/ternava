import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const releaseBin = resolve(root, 'node_modules/release-it/bin/release-it.js');

function releaseFixture(t) {
    const cwd = mkdtempSync(resolve(tmpdir(), 'ternava-release-'));
    t.after(() => rmSync(cwd, { recursive: true, force: true }));
    const git = (...args) => {
        const result = spawnSync('git', args, { cwd, encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
        return result.stdout.trim();
    };
    const config = JSON.parse(readFileSync(resolve(root, 'packages/mini/.release-it.json'), 'utf8'));
    config.git.push = false;
    config.git.requireUpstream = false;
    // Keep the real version/tag guard; never run builds, publication or release commits in this fixture.
    config.hooks = { 'before:bump': config.hooks['before:bump'], 'after:bump': 'exit 93' };
    const manifest = JSON.stringify({ name: '@usex/ternava', version: '0.0.0' });
    writeFileSync(resolve(cwd, 'package.json'), manifest);
    writeFileSync(resolve(cwd, '.release-it.json'), JSON.stringify(config));
    writeFileSync(resolve(cwd, 'CHANGELOG.md'), '# Changelog\n');
    git('init', '--initial-branch=main');
    // Reuse the repository author so CI does not need a global Git identity.
    const author = spawnSync('git', ['log', '-1', '--format=%an%n%ae'], { cwd: root, encoding: 'utf8' });
    assert.equal(author.status, 0, author.stderr);
    const [name, email] = author.stdout.trim().split('\n');
    git('config', 'user.name', name);
    git('config', 'user.email', email);
    git('add', '.');
    // Do not invoke signing or hooks for temporary test commits.
    const commit = message => git('-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
        'commit', '--allow-empty', '-m', message);
    commit('chore: initialize release fixture');
    git('-c', 'tag.gpgsign=false', 'tag', 'v0.1.0');
    commit('fix: previous release');
    git('-c', 'tag.gpgsign=false', 'tag', 'v0.1.1');
    commit('feat: next release');
    const run = (...args) => spawnSync('bun', [releaseBin, '--ci', ...args],
        { cwd, encoding: 'utf8', timeout: 30000 });
    return { cwd, git, manifest, run };
}

test('release uses Git tags when the package version was reset to zero', t => {
    const { cwd, git, manifest, run } = releaseFixture(t);
    const before = git('rev-parse', 'HEAD');
    const result = run('--dry-run', '--npm.skipChecks');
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0, output);
    assert.match(output, /Let's release @usex\/ternava \(0\.1\.1\.\.\.0\.2\.0\)/);
    assert.match(output, /bun publish --tag latest --dry-run/);
    assert.equal(readFileSync(resolve(cwd, 'package.json'), 'utf8'), manifest);
    assert.equal(git('rev-parse', 'HEAD'), before);
    assert.equal(git('status', '--porcelain'), '');
});

test('an existing release tag fails before bump, build or publish', t => {
    const { cwd, git, manifest, run } = releaseFixture(t);
    const before = git('rev-parse', 'HEAD');
    // This is a real hook run, not a dry run. Publication and Git writes are disabled independently.
    const result = run('0.1.1', '--no-npm.publish', '--no-git.commit', '--no-git.tag');
    const output = result.stdout + result.stderr;
    assert.notEqual(result.status, 0, output);
    assert.match(output, /Release tag v0\.1\.1 already exists; choose a new version\./);
    assert.doesNotMatch(output, /npm version|exit 93|bun publish/);
    assert.equal(readFileSync(resolve(cwd, 'package.json'), 'utf8'), manifest);
    assert.equal(git('rev-parse', 'HEAD'), before);
    assert.equal(git('status', '--porcelain'), '');
});

test('release preview generates notes for the public package without modifying release files', () => {
    const files = ['package.json', 'bun.lock', 'packages/base/package.json',
        'packages/mini/package.json', 'packages/mini/CHANGELOG.md'];
    const snapshot = () => files.map(file => readFileSync(resolve(root, file), 'utf8'));
    const before = snapshot();
    // Test root command wiring independently of Git history; fixtures above test the real Git baseline.
    const result = spawnSync('bun', ['run', 'release', 'patch', '--dry-run', '--ci', '--no-git', '--no-npm.ignoreVersion', '--npm.skipChecks'],
        { cwd: root, encoding: 'utf8', timeout: 30000 });
    const output = result.stdout + result.stderr;
    assert.equal(result.status, 0, output);
    assert.match(output, /Let's release @usex\/ternava/);
    assert.match(output, /Changelog:/);
    assert.match(output, /bun publish --tag latest --dry-run/);
    assert.deepEqual(snapshot(), before);
});
