import initWasm, { embed as embedWasm, config_summary } from './tern_engine.js';

let initialization;
let ready = false;

export class TernError extends Error {
    constructor(message, code) {
        super(message);
        this.name = 'TernError';
        this.code = code;
    }
}

/** Load once, then run synchronous inference in Node.js, Bun, or a browser. */
export function init(source = new URL('./tern_engine_bg.wasm', import.meta.url)) {
    if (!initialization) {
        initialization = (async () => {
            if (source instanceof URL && source.protocol === 'file:') {
                const { readFile } = await import('node:fs/promises');
                source = await readFile(source);
            }
            await initWasm({ module_or_path: source });
            ready = true;
        })().catch(error => {
            initialization = undefined;
            throw error;
        });
    }
    return initialization;
}

export default init;

function requireReady() {
    if (!ready) throw new TernError('Call await init() before inference', 'NOT_INITIALIZED');
}

export function embed(text) {
    if (typeof text !== 'string') {
        throw new TernError('embed(text): text must be a string', 'INVALID_INPUT');
    }
    requireReady();
    return embedWasm(text);
}

/** Dot product of L2-normalized embeddings. */
export function cosineSim(a, b) {
    if (a.length !== b.length) {
        throw new TernError(`vector length mismatch: ${a.length} vs ${b.length}`, 'DIM_MISMATCH');
    }
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
}

export function similar(query, corpus, opts = {}) {
    const q = embed(query);
    return corpus
        .map(text => ({ text, sim: cosineSim(q, embed(text)) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, opts.topK ?? 5);
}

export function engineInfo() {
    requireReady();
    return config_summary();
}
