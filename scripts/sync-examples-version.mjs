// Point the examples at a published @usex/ternava version.
//
//   node scripts/sync-examples-version.mjs 0.3.0
//
// The examples install the package from npm, so this runs only after that version
// is published (release-it calls it between `npm publish` and the release commit).
// It rewrites the dependency, the versioned wasm paths and the install snippets,
// then refreshes examples/bun.lock. Measured reports keep the version they were
// measured with.
import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const name = '@usex/ternava';
const next = process.argv[2];
if (!/^\d+\.\d+\.\d+$/.test(next ?? '')) {
    console.error('usage: node scripts/sync-examples-version.mjs <x.y.z>');
    process.exit(1);
}

const manifestPath = `${root}examples/package.json`;
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const previous = manifest.dependencies[name];
const measured = new Set(['examples/docs-search/retrieval-report.json']);
const swaps = [
    [`${name}@${previous}`, `${name}@${next}`],
    [`wasm/${previous}/`, `wasm/${next}/`],
    [`published \`${previous}\``, `published \`${next}\``],
];

if (previous !== next) {
    const tracked = execFileSync('git', ['ls-files', '-z', 'README.md', 'examples'], { cwd: root, encoding: 'utf8' });
    for (const file of tracked.split('\0')) {
        if (!file || measured.has(file) || file.endsWith('bun.lock') || !statSync(root + file).isFile()) continue;
        const bytes = readFileSync(root + file);
        if (bytes.includes(0)) continue; // binary
        const source = bytes.toString('utf8');
        const updated = swaps.reduce((text, [from, to]) => text.replaceAll(from, to), source);
        if (updated !== source) {
            writeFileSync(root + file, updated);
            console.log(`updated ${file}`);
        }
    }
    manifest.dependencies[name] = next;
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
    console.log(`examples: ${name} ${previous} -> ${next}`);
}

// npm can take several minutes to process a fresh publish: the version shows up in the
// metadata first and its tarball later. Wait for the tarball itself (bun also caches
// package manifests, so retrying `bun install` alone keeps seeing the old one).
const published = `https://registry.npmjs.org/${name}/-/${name.split('/')[1]}-${next}.tgz`;
for (let attempt = 1; ; attempt++) {
    const status = await fetch(published, { method: 'HEAD' }).then(response => response.status, () => 0);
    if (status === 200) break;
    if (attempt === 60) {
        console.error(`${name}@${next} is still not on npm after 15 minutes. Once it is, rerun the release with --no-npm.publish.`);
        process.exit(1);
    }
    console.log(`waiting for ${name}@${next} on npm (${attempt}/60)`);
    await new Promise(resolve => setTimeout(resolve, 15_000));
}
const install = spawnSync('bun', ['install', '--no-cache', '--cwd', `${root}examples`], { stdio: 'inherit' });
process.exit(install.status ?? 1);
