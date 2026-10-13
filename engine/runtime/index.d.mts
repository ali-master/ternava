/** Initialize once. Defaults to the adjacent WASM file (disk on Node/Bun, HTTP in browsers).
 * A URL, bytes, or compiled module may be supplied instead. Failed loads can be retried.
 * After the first successful initialization this instance keeps the same model.
 */
export declare function init(source?: URL | BufferSource | WebAssembly.Module): Promise<void>;
export default init;
export declare class TernError extends Error {
    code: string;
    constructor(message: string, code: string);
}
/** Synchronous inference after await init(); returns a normalized 384-dimensional vector. */
export declare function embed(text: string): Float32Array;
/** Dot product; inputs must be L2-normalized and have equal dimensions. */
export declare function cosineSim(a: ArrayLike<number>, b: ArrayLike<number>): number;
export declare function similar(query: string, corpus: string[], opts?: { topK?: number }): { text: string; sim: number }[];
export declare function engineInfo(): string;
