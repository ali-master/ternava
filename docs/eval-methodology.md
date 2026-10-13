# Eval Methodology

How we measure ternava before each release. Six dimensions, each answering a different question:

| Dimension | What it answers |
|---|---|
| **Model quality** | Does the model produce embeddings that match the teacher and human judgements? |
| **Quantization gap** | How much quality did ternary quantization cost vs full fp32? |
| **Performance** | How fast does it run, on what hardware? |
| **Size** | What does the user actually download? |
| **Compatibility** | Where does it run cleanly? |
| **Honest comparison** | How does it stack up against existing alternatives? |

Each section below names the metrics for one dimension and explains what they measure.

## 1. Model quality

### Teacher alignment

Mean cosine similarity between student and teacher (`all-MiniLM-L6-v2`) embeddings on a 2,000-query held-out slice of MS MARCO. The most direct test of whether distillation transferred the teacher's structure.

### STS-B AUC

On the STS Benchmark test split (1,379 sentence pairs), the probability that a human-rated "similar" pair scores higher than a "dissimilar" pair under the model's cosine similarity. Robust to scale and offset; aligns with binary-classification use cases like FAQ matching and intent routing.

### STS-B Spearman

Rank correlation between model similarity scores and human scores on the same 1,379-pair STS-B split. Sensitive to the full distribution rather than a single threshold.

### Recall@K

Fraction of queries whose correct match appears in the top-K nearest neighbors. Two hand-curated test sets — 20 general queries and 20 tech queries, each against their respective corpora.

---

## 2. Quantization gap

### Embedding quantization

Quality delta from the ternary post-training embedding table (AbsMean scaling) vs an fp32 reference. Surfaces how much the embedding lookup is bottlenecking quality.

### BitLinear weight quantization

Quality delta from ternary BitLinear weights (AbsMedian round-clamp) vs an fp32 equivalent. Tests whether QAT made the ternary constraint cheap to live with.

### Activation int8 quantization

Quality delta from int8 activations vs fp32. Expected to be the smallest since int8 is far more precise than ternary — mainly a check for surprises.

---

## 3. Runtime Performance

### Cold start

**What it measures:** time from `import` to first `embed()` call returning. Includes Wasm instantiation, tokenizer init, model.bin parse + weight unpacking.

### Warm latency

**What it measures:** steady-state per-call time after the engine is warm. The number users see in production.

### Throughput

**What it measures:** strings/sec under sustained sequential load. Useful for offline indexing scenarios.

### Memory peak

**What it measures:** peak RSS during 100 sequential `embed()` calls. Surfaces memory leaks and per-call allocation pressure.

**Methodology:** TBD. Per-target via `eval/benchmarks/`.

---

## 4. Size

| Artifact | Why it matters |
|---|---|
| `engine.wasm` bytes | What the JS bundler ships |
| `engine.wasm` gzipped | What the user actually downloads |
| `model.bin` bytes | Largest single asset |
| `model.bin` gzipped | model.bin is mostly random ternary bytes — compresses poorly |
| Total npm install | What `du -sh node_modules/@tern/` shows |

---

## Schema for `eval/results/v<X.Y.Z>.json`

See [../../eval/results/README.md](../../eval/results/README.md).