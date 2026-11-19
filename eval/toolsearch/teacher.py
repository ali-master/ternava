# /// script
# dependencies = ["sentence-transformers"]
# ///
"""Ceiling check: run diag.ts's benchmark on teacher candidates.

    MIKROTIK_MCP=... bun diag.ts --dump /tmp/toolsearch.json
    uv run teacher.py /tmp/toolsearch.json

MODEL_DIR=<dir> loads <dir>/<model name> instead of the Hub (offline / sandboxed runs).
"""
import json
import os
import statistics
import sys

from sentence_transformers import SentenceTransformer

# (model, query prefix, document prefix)
MODELS = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),  # current teacher
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
]
data = json.load(open(sys.argv[1]))
names, cases = data["names"], data["cases"]

for model_id, qp, dp in MODELS:
    local = os.environ.get("MODEL_DIR")
    model = SentenceTransformer(os.path.join(local, model_id.split("/")[1]) if local else model_id)
    enc = lambda texts, p: model.encode([p + t for t in texts], normalize_embeddings=True, batch_size=64)
    docs = enc(data["texts"], dp)
    # same sampling of unrelated pairs as diag.ts
    pairs = [(i, j) for i in range(0, len(docs), 7) for j in range(i + 3, len(docs), 97)]
    unrelated = sum(float(docs[i] @ docs[j]) for i, j in pairs) / len(pairs)
    print(f"\n== {model_id} == mean cosine of unrelated tool texts: {unrelated:.2f}")
    for k, lang in enumerate(["EN", "FA informal", "FA formal"]):
        scores = enc([c[k] for c in cases], qp) @ docs.T
        ranks, top, s50 = [], [], []
        for row, c in zip(scores, cases):
            order = (-row).argsort()
            ranks.append([names[i] for i in order].index(c[3]) + 1)
            top.append(row[order[0]])
            s50.append(row[order[49]])
        print(f"{lang}: median rank {sorted(ranks)[len(ranks) // 2]} | in top5 {sum(r <= 5 for r in ranks)}/{len(ranks)}"
              f" | mean top score {statistics.mean(map(float, top)):.2f}, mean 50th score {statistics.mean(map(float, s50)):.2f}"
              f" | ranks {ranks}")
    sim = lambda a, b: float(enc([a], qp)[0] @ enc([b], qp)[0])
    print(f"details: digits {sim('۵ مگابیت', '5 مگابیت'):.2f} | word order {sim('روتر را خاموش کن', 'خاموش کن را روتر'):.2f}"
          f" | خاموش~روشن {sim('روتر را خاموش کن', 'روتر را روشن کن'):.2f}"
          f" | enable~disable {sim('enable interface', 'disable interface'):.2f}"
          f" | روتر~router {sim('روتر', 'router'):.2f} | بکاپ~backup {sim('بکاپ', 'backup'):.2f} | وایرگارد~wireguard {sim('وایرگارد', 'wireguard'):.2f}")
