// main.js — ternava Iran history search (browser, ESM, no bundler)
//
// Two phases:
//   1. INDEXING — load wasm, fetch chunks, embed each one, live counters
//   2. SEARCH   — debounced input, embed query, cosine over corpus, top-K
//
// All on-device after first load. The info pane reports live latency, model
// architecture, and corpus stats.

import { inject } from '@vercel/analytics';
import init, { embed, config_summary } from './wasm/tern_engine.js';

// Initialize Vercel Web Analytics
inject();

// ── Element refs ────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

// Indexing phase
const $indexingPhase = $('indexing-phase');
const $progressFill  = $('progress-fill');
const $statCount     = $('stat-count');
const $statTput      = $('stat-tput');
const $statElapsed   = $('stat-elapsed');
const $indexAction   = $('indexing-action');

// Search phase
const $searchPhase  = $('search-phase');
const $query        = $('query');
const $results      = $('results');
const $latencyBadge = $('latency-badge');

// Info pane
const $latencyBig      = $('latency-big');
const $copyBtn         = $('copy-vector');
const $infoEmb         = $('info-emb');
const $infoLayers      = $('info-layers');
const $infoHeads       = $('info-heads');
const $infoDmodel      = $('info-dmodel');
const $infoOutput      = $('info-output');
const $infoMaxseq      = $('info-maxseq');
const $infoBundle      = $('info-bundle');
const $infoInit        = $('info-init');
const $infoTput        = $('info-tput');

// ── Cosine similarity ─────────────────────────────────────────────────────

function cosine(a, b) {
    let dot = 0;
    const len = a.length;
    for (let i = 0; i < len; i++) dot += a[i] * b[i];
    return dot;   // ternava outputs are L2-normalized → dot == cosine
}

// ── Parse engine config_summary string ────────────────────────────────────
//
// Input: "tern-engine v1 | embedding_format=int4 | vocab=30522 d_model=256 n_layers=2 n_heads=4 ffn_dim=1024 output_dim=384 max_seq_len=128"
// Output: { embedding_format: "int4", vocab: 30522, d_model: 256, ... }

function parseConfigSummary(s) {
    const out = {};
    for (const m of s.matchAll(/([a-z_]+)=([a-zA-Z0-9_]+)/g)) {
        const key = m[1];
        const val = m[2];
        out[key] = /^\d+$/.test(val) ? parseInt(val, 10) : val;
    }
    return out;
}

// ── State ─────────────────────────────────────────────────────────────────

let CORPUS = [];          // chunk metadata
let CORPUS_VECS = [];     // Float32Array(384) per chunk
let CONFIG = {};          // parsed engine config
let BUILD_TIME_SEC = 0;
let FINAL_TPUT = 0;
let WASM_BYTES = 0;
let WASM_INIT_MS = 0;     // how long the wasm took to compile
let LAST_QUERY_VEC = null; // most recent query embedding (for copy-button)

// ── Boot ──────────────────────────────────────────────────────────────────

(async function boot() {
    $indexAction.textContent = 'در حال بارگیری موتور…';
    const wasmStart = performance.now();
    await init();
    WASM_INIT_MS = Math.round(performance.now() - wasmStart);

    CONFIG = parseConfigSummary(config_summary());

    // (info-output-dim was removed from the HTML; nothing to update here now)

    $indexAction.textContent = 'در حال دریافت متن‌ها…';
    const corpusResp = await fetch('./chunks.json', { cache: 'no-store' });
    if (!corpusResp.ok) throw new Error('دریافت متن‌های نمونه ناموفق بود.');
    CORPUS = await corpusResp.json();
    $('corpus-count').textContent = CORPUS.length.toLocaleString('fa-IR');
    $statCount.textContent = `0 / ${CORPUS.length.toLocaleString('fa-IR')}`;
    $indexAction.textContent = 'در حال ساخت بردارها…';

    // HEAD request for wasm bytes — we surface the actual size in PACKAGE.
    fetch('./wasm/tern_engine_bg.wasm', { method: 'HEAD' })
        .then(r => {
            const bytes = parseInt(r.headers.get('content-length') || '0', 10);
            if (bytes > 0) WASM_BYTES = bytes;
        })
        .catch(() => { /* non-critical */ });

    await indexCorpus();
    finishIndexing();
})().catch(err => {
    console.error(err);
    $indexingPhase.innerHTML = `
        <p class="indexing-status" style="color:var(--accent)">بارگیری نمونه انجام نشد</p>
        <p style="text-align:center; color:var(--ink); font-weight:600; font-size:20px; margin-bottom:18px">
            ${escapeHtml(err.message)}
        </p>
        <p style="text-align:center; color:var(--ink-mute); font-size:13px">
            صفحه را با یک سرور محلی باز کنید؛ بازکردن مستقیم فایل کافی نیست.
            دستور اجرا در راهنما آمده است.
        </p>`;
});

// ── Index ─────────────────────────────────────────────────────────────────

async function indexCorpus() {
    const N = CORPUS.length;
    const BATCH = 25;
    const start = performance.now();

    CORPUS_VECS = new Array(N);
    let lastFrame = performance.now();

    for (let i = 0; i < N; i += BATCH) {
        const end = Math.min(i + BATCH, N);
        for (let j = i; j < end; j++) {
            CORPUS_VECS[j] = embed(CORPUS[j].text);
        }

        const done    = end;
        const elapsed = (performance.now() - start) / 1000;
        const tput    = Math.round(done / Math.max(elapsed, 0.001));

        $statCount.textContent = `${done.toLocaleString('fa-IR')} / ${N.toLocaleString('fa-IR')}`;
        $statTput.innerHTML    = `${tput.toLocaleString('fa-IR')} <span class="stat-unit">بردار/ثانیه</span>`;
        $statElapsed.innerHTML = `${elapsed.toLocaleString('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} <span class="stat-unit">ثانیه</span>`;
        $progressFill.style.width = `${(done / N) * 100}%`;

        if (performance.now() - lastFrame > 16) {
            await new Promise(r => requestAnimationFrame(r));
            lastFrame = performance.now();
        }
    }

    const totalMs = performance.now() - start;
    BUILD_TIME_SEC = totalMs / 1000;
    FINAL_TPUT = Math.round(N / (totalMs / 1000));
}

function finishIndexing() {
    const N = CORPUS.length;

    // Populate info pane
    $infoEmb.innerHTML      = `${CONFIG.embedding_format} <span class="muted">(کوانتیزه‌سازی سطری)</span>`;
    $infoLayers.textContent = CONFIG.n_layers.toLocaleString('fa-IR');
    $infoHeads.textContent  = CONFIG.n_heads.toLocaleString('fa-IR');
    $infoDmodel.textContent = CONFIG.d_model.toLocaleString('fa-IR');
    $infoOutput.textContent = `${CONFIG.output_dim.toLocaleString('fa-IR')} بُعد`;
    $infoMaxseq.textContent = `${CONFIG.max_seq_len.toLocaleString('fa-IR')} توکن`;

    // PACKAGE block
    if (WASM_BYTES > 0) {
        const mb = (WASM_BYTES / 1024 / 1024).toLocaleString('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
        $infoBundle.textContent = `${mb} مگابایت`;
    } else {
        $infoBundle.textContent = '— مگابایت';
    }
    $infoInit.textContent = `${WASM_INIT_MS.toLocaleString('fa-IR')} میلی‌ثانیه`;
    $infoTput.textContent = `~${FINAL_TPUT.toLocaleString('fa-IR')} بردار/ثانیه`;

    // Swap the indexing phase for the search phase
    $indexingPhase.classList.add('hidden');
    $searchPhase.classList.remove('hidden');
    $query.focus();

    // Initial empty state
    renderEmpty();
}

// ── Search ────────────────────────────────────────────────────────────────

const TOP_K = 8;
let searchDebounce;

$query.addEventListener('input', () => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(runSearch, 120);
});

for (const btn of document.querySelectorAll('.try-q')) {
    btn.addEventListener('click', () => {
        $query.value = btn.dataset.q;
        runSearch();
        $query.focus();
    });
}

function runSearch() {
    const q = $query.value.trim();
    if (!q) {
        renderEmpty();
        $latencyBadge.textContent = '—';
        $latencyBig.textContent = '—';
        LAST_QUERY_VEC = null;
        $copyBtn.disabled = true;
        return;
    }

    // Measure just the embed() call — that's "embed latency" in the strict
    // sense, and what the info pane reports. The cosine + sort below is
    // dominated by the embed call anyway (<0.2 ms vs ~2 ms at this corpus
    // size), but we keep them separate so the displayed number is accurate.
    const tEmbedStart = performance.now();
    const qv = embed(q);
    const embedMs = performance.now() - tEmbedStart;

    const scored = new Array(CORPUS.length);
    for (let i = 0; i < CORPUS.length; i++) {
        scored[i] = { idx: i, score: cosine(qv, CORPUS_VECS[i]) };
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, TOP_K);

    const embedStr = embedMs.toLocaleString('fa-IR', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    $latencyBadge.textContent = `${embedStr} میلی‌ثانیه`;
    $latencyBig.textContent   = embedStr;

    // Cache the query vector so the Copy button can grab it
    LAST_QUERY_VEC = qv;
    $copyBtn.disabled = false;

    renderResults(top);
}

// ── Copy embedding button ────────────────────────────────────────────────

$copyBtn.addEventListener('click', async () => {
    if (!LAST_QUERY_VEC) return;
    // Format as JSON array of floats, 6 decimal places — readable + paste-able
    // into Python / JS / numpy as a literal.
    const formatted = '[' +
        Array.from(LAST_QUERY_VEC).map(v => v.toFixed(6)).join(', ') +
        ']';
    try {
        await navigator.clipboard.writeText(formatted);
        $copyBtn.classList.add('copied');
        $copyBtn.querySelector('.copy-btn-label').textContent =
            `${LAST_QUERY_VEC.length.toLocaleString('fa-IR')} عدد کپی شد`;
        setTimeout(() => {
            $copyBtn.classList.remove('copied');
            $copyBtn.querySelector('.copy-btn-label').textContent =
                'کپی بردار پرسش';
        }, 1600);
    } catch (err) {
        console.warn('Clipboard write failed:', err);
        $copyBtn.querySelector('.copy-btn-label').textContent = 'کپی انجام نشد؛ دوباره تلاش کنید';
    }
});

function renderEmpty() {
    $results.innerHTML = '<p class="empty-state">پرسشتان را بنویسید یا یکی از پیشنهادها را انتخاب کنید.</p>';
}

function renderResults(top) {
    if (top.length === 0) {
        $results.innerHTML = '<div class="empty-state">نتیجه‌ای پیدا نشد.</div>';
        return;
    }

    const html = top.map(({ idx, score }) => {
        const item = CORPUS[idx];
        const snippet = stripBreadcrumbPrefix(item.text, item.breadcrumb);
        return `
            <a class="result-card" href="${escapeAttr(item.url)}" target="_blank" rel="noopener noreferrer">
                <div class="card-top">
                    <div class="card-title">${escapeHtml(item.title)}</div>
                    <div class="card-score">شباهت <span class="card-score-value">${score.toLocaleString('fa-IR', { minimumFractionDigits: 3, maximumFractionDigits: 3 })}</span></div>
                </div>
                <div class="card-source">${escapeHtml(item.period)} · منبع: ویکی‌پدیا ↗</div>
                <div class="card-snippet">${escapeHtml(snippet)}</div>
            </a>`;
    }).join('');

    $results.innerHTML = html;
}

function stripBreadcrumbPrefix(text, breadcrumb) {
    const prefix = `${breadcrumb}. `;
    return text.startsWith(prefix) ? text.slice(prefix.length) : text;
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
const escapeAttr = escapeHtml;
