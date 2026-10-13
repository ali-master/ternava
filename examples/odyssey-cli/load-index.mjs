import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export function loadIndex() {
  let index;
  try {
    index = JSON.parse(readFileSync(new URL('./index.json', import.meta.url), 'utf8'));
  } catch {
    throw new Error('نمایه آماده نیست؛ ابتدا node examples/odyssey-cli/index.mjs را اجرا کنید.');
  }
  const source = readFileSync(new URL('./odyssey.json', import.meta.url));
  const hash = createHash('sha256').update(source).digest('hex');
  if (index.corpus_sha256 !== hash) {
    throw new Error('متن‌ها تغییر کرده‌اند؛ نمایه را با اجرای index.mjs دوباره بسازید.');
  }
  return index;
}
