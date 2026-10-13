"""Create an UNTRAINED 32k mini fixture for cross-runtime parity, not quality."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"training/distill"), str(ROOT/"training/pack")]
from model import StudentEncoder
from tokenization import SPECIAL_TOKENS, bilingual_normalizer, tokenizer_metadata, load_tokenizer
from pack import pack
from unpack import UnpackedModel


def make_fixture(out):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    chars = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789سلامدنیاپژوهشکگچژبرومخهعتصضطظفقزذثئآ؟،؛."))
    tokens = SPECIAL_TOKENS + chars + ["##"+c for c in chars]
    tokens += [f"fixturetoken{i}" for i in range(32000-len(tokens))]
    tk = Tokenizer(models.WordPiece(vocab={t: i for i, t in enumerate(tokens)}, unk_token="[UNK]"))
    tk.normalizer = bilingual_normalizer()
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.add_special_tokens(SPECIAL_TOKENS)
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tk.save(str(out/"tokenizer.json"))
    meta = tokenizer_metadata(out/"tokenizer.json")
    cfg = dict(vocab_size=32000, d_model=256, n_layers=2, n_heads=4, ffn_dim=1024, output_dim=384, dropout=0.0)
    student = StudentEncoder(**cfg)
    torch.save({"model_state": student.state_dict(), "epoch": 0, "tokenizer": meta,
                "config": {**cfg, "enable_qat": True, "run_name": "UNTRAINED-parity-fixture"},
                "data_manifest": {"pairs_sha256": "SYNTHETIC-TEST-ONLY"}}, out/"checkpoint.pt")
    pack(out/"checkpoint.pt", out/"model.bin", "int4")
    assert (out/"model.bin").stat().st_size == 5_034_608
    reference = UnpackedModel.from_bin(out/"model.bin")
    texts = ["", "سلام دنیا", "پژوهش فارسی", "كيك می‌روم", "کیک می روم", "hello world", "hello "*200,
             "علی به رضا پول داد", "رضا به علی پول داد", "رمز API را تغییر بده", "hello [PAD] world"]
    encoded = load_tokenizer(meta)(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        embeddings = reference.forward(encoded["input_ids"], encoded["attention_mask"])
    (out/"parity.json").write_text(json.dumps({"texts": texts, "input_ids": encoded["input_ids"].tolist(),
        "embeddings": embeddings.tolist(), "tokenizer_sha256": meta["sha256"],
        "model_sha256": hashlib.sha256((out/"model.bin").read_bytes()).hexdigest()}, ensure_ascii=False))
    (out/"evaluation.json").write_text(json.dumps({
        "provenance": {"pairs_sha256": "SYNTHETIC-TEST-ONLY", "teacher": "UNTRAINED-FIXTURE-NOT-A-TEACHER"},
        "pairs": [{"id": str(i), "en": texts[en], "fa": texts[fa],
                   "teacher_en": embeddings[en].tolist(), "teacher_fa": embeddings[fa].tolist()}
                  for i, (en, fa) in enumerate([(5, 1), (10, 2), (6, 3)])],
    }, ensure_ascii=False))
    print(f"UNTRAINED fixture ready: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    make_fixture(parser.parse_args().output)
