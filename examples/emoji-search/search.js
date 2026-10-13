export function normalizeSearchText(text) {
  return text.normalize('NFKC').toLowerCase()
    .replace(/[يى]/gu, 'ی').replace(/ك/gu, 'ک')
    .replace(/[\u0640\u064b-\u065f\u0670]/gu, '')
    .replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

export function searchPhrases({ name, text, aliases = [] }) {
  if (!Array.isArray(aliases) || !aliases.every(p => typeof p === 'string' && normalizeSearchText(p))) {
    throw new TypeError('عبارت‌های جایگزین باید آرایه‌ای از متن‌های غیرخالی باشند.');
  }
  return [...new Set([name, ...text.split(/[،,؛;]/u), ...aliases]
    .map(normalizeSearchText).filter(Boolean))];
}

export function rankEmojis(query, queryVector, vectors, phrases) {
  const normalized = normalizeSearchText(query);
  if (!normalized) return [];
  const padded = ` ${normalized} `;
  // shortcut: phrase coverage is limited to the corpus; train the bilingual model for unseen paraphrases.
  return vectors.map((list, i) => {
    let phraseScore = 0;
    for (const phrase of phrases[i]) {
      if (padded.includes(` ${phrase} `)) {
        phraseScore = Math.max(phraseScore, phrase.split(' ').length);
      }
    }
    let best = -2;
    for (const vector of list) {
      let dot = 0;
      for (let j = 0; j < queryVector.length; j++) dot += queryVector[j] * vector[j];
      best = Math.max(best, dot);
    }
    return { i, best, phraseScore };
  }).sort((a, b) => b.phraseScore - a.phraseScore || b.best - a.best);
}
