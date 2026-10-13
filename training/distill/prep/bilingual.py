"""Prepare a train-only WordPiece vocabulary and a balanced bilingual cache."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import load_dataset
from huggingface_hub import HfApi

from config import load_config
from data import save_cache
from ingest import teacher_encode, tokenize_batch
from tokenization import bilingual_normalizer, load_tokenizer, train_tokenizer


def split_pairs(rows, cfg):
    """Drop repeated endpoints before splitting, so translations never leak."""
    normalize = bilingual_normalizer().normalize_str
    seen_en, seen_fa, pairs = set(), set(), []
    for row in rows:
        en, fa = row.get("english"), row.get("non_english")
        if not isinstance(en, str) or not isinstance(fa, str):
            continue
        en, fa = " ".join(normalize(en).split()), " ".join(normalize(fa).split())
        if not en or not fa or en in seen_en or fa in seen_fa:
            continue
        if not any("a" <= c <= "z" for c in en) or not any("\u0600" <= c <= "\u06ff" for c in fa):
            continue
        seen_en.add(en)
        seen_fa.add(fa)
        pairs.append({"id": hashlib.sha256((en + "\n" + fa).encode()).hexdigest(), "en": en, "fa": fa})
        if len(pairs) == cfg.pairs:
            break
    if len(pairs) < cfg.pairs:
        raise ValueError(f"Only {len(pairs)} usable unique pairs; requested {cfg.pairs}")
    random.Random(cfg.seed).shuffle(pairs)
    n_test, n_val = max(1, round(len(pairs) * cfg.test_ratio)), max(1, round(len(pairs) * cfg.val_ratio))
    return {"test": pairs[:n_test], "val": pairs[n_test:n_test+n_val], "train": pairs[n_test+n_val:]}


def prepare(config_path: Path, pairs_jsonl: Path | None = None, tokenizer_only: bool = False):
    config = load_config(config_path)
    cfg = config.pilot
    if cfg is None or config.train is None:
        raise ValueError("Both pilot and train sections are required")
    if config.train.vocab_size != cfg.vocab_size or config.train.cache_dir.resolve() != cfg.output_dir.resolve():
        raise ValueError("train vocab_size/cache_dir must match pilot vocab_size/output_dir")
    if config.train.contrastive_w != 0:
        raise ValueError("Bilingual translation pairs require contrastive_w=0 (the old repulsion loss separates positives)")
    out = cfg.output_dir
    if (out / f"{config.train.cache_name}.manifest.json").exists():
        raise FileExistsError("Prepared cache exists; choose a new output directory to preserve its provenance")
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "prepare.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["config"] != cfg.model_dump(mode="json"):
            raise ValueError("Preparation config changed; use a new output_dir")
        if pairs_jsonl and hashlib.sha256(pairs_jsonl.read_bytes()).hexdigest() != plan["source"].get("sha256"):
            raise ValueError("Local pair source changed; use a new output_dir")
        if hashlib.sha256((out / "pairs.json").read_bytes()).hexdigest() != plan["pairs_sha256"]:
            raise ValueError("Prepared pairs changed")
        splits = json.loads((out / "pairs.json").read_text())
        metadata = plan["tokenizer"]
    else:
        if pairs_jsonl:
            rows = [json.loads(line) for line in pairs_jsonl.read_text().splitlines() if line.strip()]
            random.Random(cfg.seed).shuffle(rows)
            source = {"path": str(pairs_jsonl.resolve()), "sha256": hashlib.sha256(pairs_jsonl.read_bytes()).hexdigest()}
        else:
            revision = HfApi().dataset_info(cfg.dataset, revision=cfg.dataset_revision).sha
            rows = load_dataset(cfg.dataset, cfg.dataset_config, revision=revision, split="train").shuffle(seed=cfg.seed)
            source = {"dataset": cfg.dataset, "config": cfg.dataset_config, "revision": revision,
                      "fingerprint": rows._fingerprint}
        splits = split_pairs(rows, cfg)
        if len(splits["test"]) < 3:
            raise ValueError("Need at least 3 test pairs; increase pairs or test_ratio")
        metadata = train_tokenizer((p[lang] for p in splits["train"] for lang in ("en", "fa")), out / "tokenizer", cfg.vocab_size)
        (out / "pairs.json").write_text(json.dumps(splits, ensure_ascii=False) + "\n")
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "tokenizers", "sentence-transformers", "bitlinear", "datasets")}
        plan = {"config": cfg.model_dump(mode="json"), "source": source, "tokenizer": metadata, "versions": versions,
                "pairs_sha256": hashlib.sha256((out / "pairs.json").read_bytes()).hexdigest()}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    tokenizer = load_tokenizer(metadata)
    if tokenizer_only:
        print(f"Tokenizer ready: {metadata}; run again without --tokenizer-only to encode the teacher")
        return
    teacher_revision = HfApi().model_info(cfg.teacher_id, revision=cfg.teacher_revision).sha
    prepared, evaluation = {}, []
    for split, pairs in splits.items():
        texts = [p[lang] for p in pairs for lang in ("en", "fa")]
        tokens = tokenize_batch(texts, tokenizer, 128)
        targets = teacher_encode(texts, cfg.teacher_id, teacher_revision, cfg.teacher_batch_size)
        if (targets.shape != (len(texts), config.train.output_dim) or not torch.isfinite(targets).all()
                or not torch.allclose(targets.norm(dim=1), torch.ones(len(texts)), atol=1e-3)):
            raise ValueError("Invalid teacher target shape/values")
        prepared[split] = [
            {"input_ids": tokens["input_ids"][i], "attention_mask": tokens["attention_mask"][i],
             "teacher_emb": targets[i], "source": ("en", "fa")[i % 2], "text": text,
             "pair_id": pairs[i // 2]["id"]}
            for i, text in enumerate(texts)
        ]
        if split == "test":
            evaluation = [{**p, "teacher_en": targets[2*i].tolist(), "teacher_fa": targets[2*i+1].tolist()}
                          for i, p in enumerate(pairs[:cfg.eval_pairs])]
    manifest = {**plan, "teacher": {"id": cfg.teacher_id, "revision": teacher_revision},
                "split_counts": {k: len(v) for k, v in prepared.items()}}
    (out / "evaluation.json").write_text(json.dumps({"provenance": manifest, "pairs": evaluation}, ensure_ascii=False) + "\n")
    save_cache(prepared, manifest, out, config.train.cache_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, help="Local english/non_english pairs instead of downloading OPUS")
    parser.add_argument("--tokenizer-only", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.pairs_jsonl, args.tokenizer_only)
