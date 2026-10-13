"""External STS evaluation of actual WASM embeddings, with pilot overlap filtering."""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/distill"))
from tokenization import bilingual_normalizer


def prepare(pairs_path, output):
    from datasets import load_dataset
    if output.exists():
        raise FileExistsError(f"Preserve the frozen STS set: {output}")
    normalizer = bilingual_normalizer()
    def normalize(text):
        return " ".join(re.sub(r"[^\w\s]", " ", normalizer.normalize_str(text)).split())
    pairs = json.loads(pairs_path.read_text())
    seen = {normalize(pair[lang]) for split in ("train", "val") for pair in pairs[split] for lang in ("en", "fa")}
    benchmarks = {}
    for lang, dataset, revision in [
        ("fa", "MCINext/farsick-sts", "5632877d94c18b8c2f282ffd25afb393c72649ab"),
        ("en", "mteb/stsbenchmark-sts", "96943a16ea6a35129e253c659081cb59daf81b30"),
    ]:
        source = load_dataset(dataset, revision=revision, split="test")
        samples, excluded = [], 0
        for i, row in enumerate(source):
            a, b, score = row["sentence1"], row["sentence2"], float(row["score"])
            if not isinstance(a, str) or not isinstance(b, str) or not a.strip() or not b.strip() or not math.isfinite(score):
                raise ValueError(f"Invalid STS row {dataset}:{i}")
            if normalize(a) in seen or normalize(b) in seen:
                excluded += 1
                continue
            samples.append({"id": i, "sentence1": a, "sentence2": b, "score": score})
        if len(samples) < 3:
            raise ValueError("Insufficient non-overlapping STS pairs")
        benchmarks[lang] = {"dataset": dataset, "revision": revision, "split": "test",
                            "original_pairs": len(source), "excluded_overlap_pairs": excluded, "samples": samples}
    result = {"pilot_pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
              "overlap_method": "Normalized exact sentence match against pilot train and validation. Teacher/baseline pretraining exposure is unknown.",
              "benchmarks": benchmarks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({lang: {k: v for k, v in value.items() if k != "samples"} | {"kept": len(value["samples"])}
                      for lang, value in benchmarks.items()}, indent=2))


def correlations(predicted, expected):
    if len(predicted) < 3 or len(predicted) != len(expected) or not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid STS predictions or labels")
    rho = float(spearmanr(predicted, expected).statistic)
    pearson = float(pearsonr(predicted, expected).statistic)
    return {"pairs": len(predicted), "spearman": rho if math.isfinite(rho) else None,
            "pearson": pearson if math.isfinite(pearson) else None}


def evaluate(dataset_path, models, output):
    dataset = json.loads(dataset_path.read_text())
    texts = list(dict.fromkeys(row[key] for data in dataset["benchmarks"].values()
                              for row in data["samples"] for key in ("sentence1", "sentence2")))
    mapping = {text: i for i, text in enumerate(texts)}
    report = {"dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "pilot_pairs_sha256": dataset["pilot_pairs_sha256"], "overlap_method": dataset["overlap_method"],
              "benchmarks": {lang: {k: v for k, v in data.items() if k != "samples"}
                             for lang, data in dataset["benchmarks"].items()}, "models": {}}
    for name, glue in models.items():
        raw = subprocess.run(["node", str(Path(__file__).with_name("collect-wasm.mjs")), str(glue)],
                             input=json.dumps(texts), capture_output=True, text=True, check=True)
        result = json.loads(raw.stdout)
        vectors = np.asarray(result["embeddings"], dtype=np.float32)
        measured = {"wasm_sha256": result["wasm_sha256"], "config": result["config"], "languages": {}}
        for lang, data in dataset["benchmarks"].items():
            predicted = [float(vectors[mapping[row["sentence1"]]] @ vectors[mapping[row["sentence2"]]]) for row in data["samples"]]
            measured["languages"][lang] = correlations(predicted, [row["score"] for row in data["samples"]])
        report["models"][name] = measured
        print(json.dumps({"model": name, "scores": measured["languages"]}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-from-pairs", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], help="name=web-target-glue.js")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_from_pairs:
        prepare(args.prepare_from_pairs, args.dataset)
    else:
        if not args.model or not args.output:
            parser.error("Evaluation requires --model and --output")
        evaluate(args.dataset, dict(item.split("=", 1) for item in args.model), args.output)
