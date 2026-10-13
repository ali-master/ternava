import assert from 'node:assert/strict';
import { cpSync, readFileSync, realpathSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const manifestPath = realpathSync(require.resolve('@usex/ternava/package.json'));
const { version } = JSON.parse(readFileSync(manifestPath, 'utf8'));
assert.equal(version, '0.2.0');
assert.equal(manifestPath, join(realpathSync(new URL('./', import.meta.url)), 'node_modules/@usex/ternava/package.json'),
    'Examples must use the npm package; remove stale workspace links and reinstall in examples.');
for (const demo of ['docs-search', 'emoji-search']) {
    cpSync(join(dirname(manifestPath), 'pkg-web'), new URL(`./${demo}/wasm/${version}/`, import.meta.url),
        { recursive: true });
}
await import('./emoji-search/build-index.mjs');
await import('./odyssey-cli/index.mjs');
console.log(`Examples built with @usex/ternava@${version} from npm.`);
