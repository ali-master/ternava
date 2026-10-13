"""Check frozen evaluation texts against pilot train/validation before scoring."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training/distill"))
from tokenization import bilingual_normalizer


def audit(fixture_path, pairs_path, additional_path=None):
    fixture = json.loads(fixture_path.read_text())
    splits = json.loads(pairs_path.read_text())
    normalize = bilingual_normalizer().normalize_str
    def words(text):
        return tuple(re.sub(r"[^\w\s]", " ", normalize(text)).split())
    evaluation = {item["id"]: words(item["text"]) for item in fixture["corpus"] + fixture["queries"]}
    exact = set(evaluation.values())
    grams = {tokens[i:i+8] for tokens in evaluation.values() for i in range(len(tokens)-7)}
    collisions, shared_grams = [], set()
    count = 0
    for split in ("train", "val"):
        for pair in splits[split]:
            for lang in ("en", "fa"):
                count += 1
                tokens = words(pair[lang])
                if tokens in exact:
                    collisions.append({"split": split, "pair_id": pair["id"], "lang": lang})
                shared_grams.update(tokens[i:i+8] for i in range(len(tokens)-7) if tokens[i:i+8] in grams)
    if additional_path:
        for i, text in enumerate(json.loads(additional_path.read_text())):
            if not isinstance(text, str):
                raise ValueError("Additional training texts must be strings")
            count += 1
            tokens = words(text)
            if tokens in exact:
                collisions.append({"split": "additional-train-val", "index": i})
            shared_grams.update(tokens[j:j+8] for j in range(len(tokens)-7) if tokens[j:j+8] in grams)
    return {"fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
            "additional_sha256": hashlib.sha256(additional_path.read_bytes()).hexdigest() if additional_path else None,
            "texts_checked": len(evaluation), "train_val_sentences": count,
            "normalized_exact_collisions": collisions,
            "shared_eight_word_spans": [" ".join(g) for g in sorted(shared_grams)],
            "method": "Bilingual normalization, punctuation removal and whitespace collapse. Checks exact text and contiguous eight-word spans; cannot establish absence from teacher/baseline pretraining or semantic independence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional", type=Path, help="JSON array of extra train/validation texts")
    args = parser.parse_args()
    report = audit(args.fixture, args.pairs, args.additional)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["normalized_exact_collisions"] or report["shared_eight_word_spans"]:
        raise SystemExit("Held-out overlap detected; review before evaluating")
