# /// script
# dependencies = ["torch", "numpy", "scipy", "transformers", "tokenizers", "bitlinear==2.4.6"]
# ///
"""Run the repo's frozen ranking corpora (and optionally STS-B) on checkpoints, as shipped (ternary + int4).

    uv run gates.py [--stsb stsb-test.jsonl] run/a.pt run/b.pt

Needs no training cache. Not the release protocol: no overlap audit, no WASM. The
shipped model was tuned and selected on several of these corpora; a fresh model was not.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from checkpoints import ROOT, RUNS, embed, load

CORPUS = ROOT / "eval/quality/corpus"


def suites():
    for name in ["bilingual-hard-heldout", "bilingual-hard-round2", "bilingual-hard-round3", "bilingual-hard-round4"]:
        d = json.loads((CORPUS / f"{name}.json").read_text())
        for lang in ("en", "fa"):  # full corpus, queries of one language
            yield f"{name.removeprefix('bilingual-hard-')}/{lang}", d["corpus"], [q for q in d["queries"] if q["lang"] == lang]
    for name in ["persian-domain-hard", "persian-reasoning-hard", "persian-compositional-hard"]:
        for k, s in json.loads((CORPUS / f"{name}.json").read_text())["suites"].items():
            yield f"{name.removeprefix('persian-').removesuffix('-hard')}/{k}", s["corpus"], s["queries"]
    d = json.loads((CORPUS / "persian-native-demos.json").read_text())
    for k, corpus in d["corpora"].items():
        yield f"native-demos/{k}", corpus, d["queries"][k]


def rank_suite(model, corpus, queries):
    ids = [c["id"] for c in corpus]
    scores = (embed(model, [q["text"] for q in queries]) @ embed(model, [c["text"] for c in corpus]).T).numpy()
    top1 = ndcg = contrast = pairs = 0
    for row, q in zip(scores, queries):
        rel = q["relevant"]
        order = np.argsort(-row, kind="stable")[:10]
        top1 += ids[order[0]] in rel
        ideal = sum((2 ** g - 1) / np.log2(r + 2) for r, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
        ndcg += sum((2 ** rel.get(ids[i], 0) - 1) / np.log2(r + 2) for r, i in enumerate(order)) / ideal
        if q.get("contrast") in ids:  # right answer must beat its counterfactual twin
            pairs += 1
            contrast += max(row[ids.index(i)] for i in rel) > row[ids.index(q["contrast"])]
    return top1, ndcg / len(queries), contrast, pairs


if __name__ == "__main__":
    args = sys.argv[1:]
    stsb = [json.loads(line) for line in open(args[1])] if args[0] == "--stsb" else None
    paths = args[2:] if stsb else args
    models = [load(RUNS / p, True) for p in paths]
    print("suite".ljust(26) + "".join(p.split("/")[0][-22:].rjust(34) for p in paths))
    print(" " * 26 + "    top1   ndcg@10   contrast     " * len(paths))
    for name, corpus, queries in suites():
        cells = []
        for m in models:
            t, n, c, p = rank_suite(m, corpus, queries)
            cells.append(f"{t:>4}/{len(queries):<3} {n:>8.3f}   {f'{c}/{p}' if p else '-':>8}     ")
        print(name.ljust(26) + "".join(c.rjust(34) for c in cells))
    if stsb:
        gold = [r["score"] for r in stsb]
        for p, m in zip(paths, models):
            a, b = embed(m, [r["sentence1"] for r in stsb]), embed(m, [r["sentence2"] for r in stsb])
            print(f"STS-B test spearman  {p}: {spearmanr((a * b).sum(-1).numpy(), gold).statistic:.4f}")
