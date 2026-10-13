"""Compare baseline and candidate WASM on exactly the same held-out en/fa texts."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/distill"))
from tokenization import load_tokenizer


def retrieval_metrics(scores, relevant):
    """Graded NDCG and recall; ties retain corpus order for reproducibility."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(scores) == 0 or len(relevant) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid retrieval scores")
    metrics = {"ndcg@10": [], "recall@1": [], "recall@10": []}
    for row, rel in zip(scores, relevant):
        if not rel or any(i < 0 or i >= len(row) or not math.isfinite(g) or g <= 0 for i, g in rel.items()):
            raise ValueError("Each query needs valid positive relevance judgments")
        ranked = np.argsort(-row, kind="stable")[:10]
        dcg = sum((2 ** rel.get(int(i), 0) - 1) / math.log2(rank + 2) for rank, i in enumerate(ranked))
        ideal = sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
        metrics["ndcg@10"].append(dcg / ideal)
        for k in (1, 10):
            metrics[f"recall@{k}"].append(sum(int(i) in rel for i in ranked[:k]) / len(rel))
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def token_stats(texts, tokenizer, actual_ids):
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths = [len(e.ids) for e in tokenizer.encode_batch(texts)]
    unk = tokenizer.token_to_id("[UNK]")
    special = {tokenizer.token_to_id(t) for t in ("[PAD]", "[CLS]", "[SEP]")}
    ids = [i for row in actual_ids for i in row if i not in special]
    return {"tokens_p50": float(np.median(lengths)), "tokens_p95": float(np.percentile(lengths, 95)),
            "truncated_fraction": float(np.mean(np.array(lengths) > 128)),
            "unk_fraction": ids.count(unk) / max(len(ids), 1)}


def retrieval_details(scores, queries, corpus):
    """Expose failures and hard-negative margins, including ambiguous ties."""
    scores = np.asarray(scores)
    mapping = {d["id"]: i for i, d in enumerate(corpus)}
    details = []
    for row, query in zip(scores, queries):
        ranked = np.argsort(-row, kind="stable")
        positives = {mapping[key]: grade for key, grade in query["relevant"].items()}
        item = {"id": query["id"], "text": query["text"], "category": query.get("category", "unspecified"),
                "top1": corpus[int(ranked[0])]["id"], "top1_correct": int(ranked[0]) in positives,
                "first_relevant_rank": next(rank + 1 for rank, i in enumerate(ranked) if int(i) in positives),
                "top3": [{"id": corpus[int(i)]["id"], "cosine": float(row[i])} for i in ranked[:3]]}
        if "contrast" in query:
            margin = max(float(row[i]) for i in positives) - float(row[mapping[query["contrast"]]])
            item["contrast_margin"] = margin
            item["contrast_tied"] = abs(margin) <= 1e-6
            item["contrast_correct"] = margin > 1e-6
        details.append(item)
    categories = {}
    for category in sorted({q["category"] for q in details}):
        subset = [q for q in details if q["category"] == category]
        categories[category] = {"count": len(subset),
                                "top1_accuracy": sum(q["top1_correct"] for q in subset) / len(subset)}
        contrasted = [q for q in subset if "contrast_correct" in q]
        if contrasted:
            categories[category]["contrast_accuracy"] = sum(q["contrast_correct"] for q in contrasted) / len(contrasted)
            categories[category]["contrast_ties"] = sum(q["contrast_tied"] for q in contrasted)
    return {"categories": categories, "queries": details}


def score(result, evaluation, retrieval, tokenizer_path):
    pairs = evaluation["pairs"]
    n = len(pairs)
    vectors = np.asarray(result["embeddings"], dtype=np.float32)
    output = {k: v for k, v in result.items() if k not in ("embeddings", "tokens", "latency_ms")}
    output["latency_p50_ms"] = float(np.median(result["latency_ms"]))
    output["latency_p95_ms"] = float(np.percentile(result["latency_ms"], 95))
    output["languages"] = {}
    for offset, lang in enumerate(("en", "fa")):
        student = vectors[offset:2*n:2]
        teacher = np.asarray([p[f"teacher_{lang}"] for p in pairs])
        idx = np.triu_indices(n, k=1)
        rho = float(spearmanr((student @ student.T)[idx], (teacher @ teacher.T)[idx]).statistic)
        output["languages"][lang] = {
            "teacher_spearman": rho if math.isfinite(rho) else None,
            **token_stats([p[lang] for p in pairs], Tokenizer.from_file(str(tokenizer_path)), result["tokens"][offset:2*n:2]),
        }
    cross = vectors[:2*n:2] @ vectors[1:2*n:2].T
    positives = [{i: 1} for i in range(n)]
    output["en_to_fa"] = retrieval_metrics(cross, positives)
    output["fa_to_en"] = retrieval_metrics(cross.T, positives)
    if retrieval:
        corpus = retrieval["corpus"]
        docs = vectors[2*n:2*n+len(corpus)]
        queries = vectors[2*n+len(corpus):]
        for lang in ("en", "fa"):
            indices = [i for i, d in enumerate(corpus) if d["lang"] == lang]
            mapping = {corpus[i]["id"]: j for j, i in enumerate(indices)}
            selected = [(i, q) for i, q in enumerate(retrieval["queries"]) if q["lang"] == lang]
            if not indices or not selected:
                raise ValueError(f"Retrieval dataset needs documents and queries for {lang}")
            rels = [{mapping[key]: grade for key, grade in q["relevant"].items()} for _, q in selected]
            scores = queries[[i for i, _ in selected]] @ docs[indices].T
            output["languages"][lang]["retrieval"] = retrieval_metrics(scores, rels)
            output["languages"][lang]["retrieval_details"] = retrieval_details(
                scores, [q for _, q in selected], [corpus[i] for i in indices])
    return output


def compare(evaluation_path, candidate, candidate_model, baseline, retrieval_path, output_path):
    evaluation = json.loads(evaluation_path.read_text())
    pairs = evaluation["pairs"]
    if len(pairs) < 3 or len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Need at least 3 unique held-out pairs")
    for lang in ("en", "fa"):
        targets = np.asarray([p[f"teacher_{lang}"] for p in pairs])
        if targets.shape != (len(pairs), 384) or not np.isfinite(targets).all():
            raise ValueError("Invalid teacher evaluation targets")
    retrieval = json.loads(retrieval_path.read_text()) if retrieval_path else None
    if retrieval:
        if len({d["id"] for d in retrieval["corpus"]}) != len(retrieval["corpus"]):
            raise ValueError("Duplicate document IDs")
        docs = {d["id"]: d for d in retrieval["corpus"]}
        if any(d["lang"] not in ("en", "fa") or not isinstance(d["text"], str) or not d["text"].strip() for d in docs.values()):
            raise ValueError("Invalid retrieval document")
        for q in retrieval["queries"]:
            if q["lang"] not in ("en", "fa") or not isinstance(q["text"], str) or not q["text"].strip() or not q["relevant"] or any(k not in docs or docs[k]["lang"] != q["lang"] for k in q["relevant"]):
                raise ValueError("Invalid retrieval query/qrels")
            if "contrast" in q and (q["contrast"] not in docs or
                    docs[q["contrast"]]["lang"] != q["lang"] or q["contrast"] in q["relevant"]):
                raise ValueError("Invalid hard-negative document")
    texts = [p[lang] for p in pairs for lang in ("en", "fa")]
    if retrieval:
        texts += [d["text"] for d in retrieval["corpus"]] + [q["text"] for q in retrieval["queries"]]
    sidecar = json.loads(candidate_model.with_suffix(candidate_model.suffix + ".json").read_text())
    meta = sidecar["extra"]["tokenizer"]
    candidate_tokenizer = candidate_model.parent / meta["path"]
    subprocess.run([sys.executable, str(ROOT / "scripts/check-model.py"), str(candidate_model), str(candidate_tokenizer)], check=True)
    meta = {**meta, "path": str(candidate_tokenizer.resolve())}
    tokenizer = load_tokenizer(meta)
    if sidecar["extra"]["data_manifest"]["pairs_sha256"] != evaluation["provenance"]["pairs_sha256"]:
        raise ValueError("Evaluation does not belong to this pilot's held-out split")
    report = {"evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
              "candidate_packed_model_bytes": candidate_model.stat().st_size,
              "retrieval_sha256": hashlib.sha256(retrieval_path.read_bytes()).hexdigest() if retrieval_path else None,
              "retrieval_scope": retrieval.get("scope", "unverified") if retrieval else "missing",
              "pairs": len(pairs), "teacher": evaluation["provenance"]["teacher"], "models": {}}
    for name, glue, path in [("baseline", baseline, ROOT / "engine/assets/tokenizer.json"),
                             ("candidate", candidate, candidate_tokenizer)]:
        result = subprocess.run(["node", str(Path(__file__).with_name("collect-wasm.mjs")), str(glue)],
                                input=json.dumps(texts), capture_output=True, text=True, check=True)
        result = json.loads(result.stdout)
        if name == "candidate":
            if result["tokenizer_sha256"] != meta["sha256"] or result["model_sha256"] != hashlib.sha256(candidate_model.read_bytes()).hexdigest():
                raise ValueError("WASM was built with different model/tokenizer assets")
            expected = tokenizer(texts, truncation=True, padding="max_length", max_length=128)["input_ids"]
            if result["tokens"] != expected:
                raise ValueError("Python/WASM tokenizer parity failed")
        report["models"][name] = score(result, evaluation, retrieval, path)
    a, b = report["models"]["baseline"], report["models"]["candidate"]
    report["gzip_growth_fraction"] = b["gzip_bytes"] / a["gzip_bytes"] - 1
    report["checks"] = {"gzip_growth_at_most_10_percent": report["gzip_growth_fraction"] <= 0.1,
                        "tokenizer_parity": True}
    if retrieval:
        report["checks"]["english_retrieval_retained_97_percent"] = b["languages"]["en"]["retrieval"]["ndcg@10"] >= 0.97*a["languages"]["en"]["retrieval"]["ndcg@10"]
        report["checks"]["persian_retrieval_improved"] = b["languages"]["fa"]["retrieval"]["ndcg@10"] > a["languages"]["fa"]["retrieval"]["ndcg@10"]
    report["release_ready"] = False
    report["note"] = "Pilot comparison only; smoke retrieval and teacher fidelity do not certify general Persian quality. Independently held-out benchmarks and human review are required."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output_path), "checks": report["checks"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Built web-target tern_engine.js")
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "examples/docs-search/wasm/tern_engine.js")
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.evaluation, args.candidate, args.candidate_model, args.baseline, args.retrieval, args.output)
