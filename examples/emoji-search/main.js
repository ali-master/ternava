// Persian phrase matching followed by cosine ranking, entirely in the browser.

import init, { embed } from './wasm/tern_engine.js';
import { rankEmojis } from './search.js';
import { CHIPS, CHALLENGES } from './queries.js';

const $ = (id) => document.getElementById(id);
const grid = $('grid'), input = $('q'), latEl = $('lat'), chipsEl = $('chips');

const TOP_K = 8;

let EMOJIS = [], NAMES = [], VECS = [], PHRASES = [];

// ── Boot ────────────────────────────────────────────────────────────────
(async function boot() {
  input.disabled = true;
  input.placeholder = 'در حال بارگیری مدل…';

  const response = await fetch('./emoji-index.json');
  if (!response.ok) throw new Error('دریافت داده‌های ایموجی ناموفق بود.');
  const idx = await response.json();
  const wasmResponse = await fetch('./wasm/tern_engine_bg.wasm');
  if (!wasmResponse.ok) throw new Error('دریافت مدل ناموفق بود.');
  const wasmBytes = await wasmResponse.arrayBuffer();
  await init({ module_or_path: wasmBytes });
  const digest = await crypto.subtle.digest('SHA-256', wasmBytes);
  const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  if (idx.wasm_sha256 !== hash) throw new Error('مدل و فهرست ایموجی هماهنگ نیستند؛ فهرست را دوباره بسازید.');
  if (!Array.isArray(idx.phrases) || idx.phrases.length !== idx.emojis.length ||
      !idx.phrases.every(list => Array.isArray(list) && list.length && list.every(p => typeof p === 'string' && p.trim()))) {
    throw new Error('عبارت‌های جست‌وجو آماده نیستند؛ فهرست را دوباره بسازید.');
  }
  EMOJIS = idx.emojis; NAMES = idx.names; PHRASES = idx.phrases;
  VECS = idx.vecs.map((list) => list.map((v) => Float32Array.from(v)));

  buildChips(chipsEl, CHIPS);
  buildChips($('challengeChips'), CHALLENGES);
  input.disabled = false;
  input.placeholder = 'این جلسه می‌توانست یک پیام باشد…';
  input.focus();
})().catch((err) => {
  console.error(err);
  $('status').textContent = `بارگیری انجام نشد: ${err.message}`;
  input.placeholder = 'بارگیری نشد؛ صفحه را با سرور محلی باز کنید';
});

// ── Search ─────────────────────────────────────────────────────────────
let debounce;
input.addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(runSearch, 80);
});

function runSearch() {
  const q = input.value.trim();
  if (!q) { grid.innerHTML = ''; setLatency(null); return; }

  const t0 = performance.now();
  const qv = embed(q);                              // <-- the only per-keystroke embed
  const embedMs = performance.now() - t0;

  const scored = rankEmojis(q, qv, VECS, PHRASES);

  setLatency(embedMs);
  render(scored.slice(0, TOP_K));
}

function render(top) {
  grid.innerHTML = top.map(({ i }, rank) => `
    <div class="cell ${rank === 0 ? 'top' : ''}">
      <span class="emoji">${EMOJIS[i]}</span>
      <span class="name" title="${escapeHtml(NAMES[i])}">${escapeHtml(NAMES[i])}</span>
    </div>`).join('');
}

function setLatency(ms) {
  const num = ms == null ? '&mdash;' : ms.toLocaleString('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  latEl.innerHTML = `<span class="lat-num">${num}</span><span class="lat-unit">میلی‌ثانیه</span>`;
  if (ms == null) return;
  latEl.classList.add('flash');
  setTimeout(() => latEl.classList.remove('flash'), 160);
}

// ── Chips ──────────────────────────────────────────────────────────────
function buildChips(container, queries) {
  container.innerHTML = queries.map((c) => `<button class="chip">${escapeHtml(c)}</button>`).join('');
  for (const btn of container.querySelectorAll('.chip')) {
    btn.addEventListener('click', () => {
      input.value = btn.textContent;
      input.focus();
      runSearch();
    });
  }
}

// ── QR overlay: click the QR or press "f" for a full-screen finale ──────
const overlay = $('qrOverlay');
const toggleQR = () => {
  overlay.hidden = !overlay.hidden;
  document.querySelector('.page').inert = !overlay.hidden;
  (overlay.hidden ? $('qrBtn') : $('closeQR')).focus();
};
$('closeQR').addEventListener('click', e => { e.stopPropagation(); toggleQR(); });
$('qrBtn').addEventListener('click', toggleQR);
overlay.addEventListener('click', toggleQR);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !overlay.hidden) toggleQR();
  else if (e.key === 'Tab' && !overlay.hidden) { e.preventDefault(); $('closeQR').focus(); }
  else if ((e.key === 'f' || e.key === 'F') && document.activeElement !== input) toggleQR();
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
