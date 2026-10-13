"""Offline contracts; synthetic teacher vectors test plumbing, never quality."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "training/distill"), str(ROOT / "training/distill/prep"),
               str(ROOT / "training/pack"), str(ROOT / "eval/quality")]
from config import PilotConfig, TrainConfig, load_config
from tokenization import (bilingual_normalizer, train_tokenizer, load_tokenizer,
                          tokenizer_metadata, validate_tokenizer, check_checkpoint_tokenizer)
from prep.bilingual import split_pairs, prepare
from loss import relational_loss
from model import StudentEncoder
from trainer import Trainer
from data import load_cache, collate_fn, TernDataset
from pack import pack
from unpack import UnpackedModel
from verify import _load_source_model, _get_eval_inputs
eval_spec = importlib.util.spec_from_file_location("bilingual_eval", ROOT / "eval/quality/bilingual.py")
evaluation = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluation)
audit_spec = importlib.util.spec_from_file_location("heldout_audit", ROOT / "eval/quality/audit-heldout.py")
heldout_audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(heldout_audit)
sts_spec = importlib.util.spec_from_file_location("heldout_sts", ROOT / "eval/quality/sts-heldout.py")
heldout_sts = importlib.util.module_from_spec(sts_spec)
sts_spec.loader.exec_module(heldout_sts)

spec = importlib.util.spec_from_file_location("check_model", ROOT / "scripts/check-model.py")
check_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_model)


def rows(n=100):
    return [{"english": f"Question {i} about my account and password",
             "non_english": f"پرسش {i} درباره حساب کاربری و رمز عبور من"} for i in range(n)]


class BilingualTests(unittest.TestCase):
    def test_group_split_and_normalized_dedup(self):
        data = rows(40)
        data.insert(1, {"english": "other", "non_english": data[0]["non_english"].replace("ی", "ي")})
        data.append({"english": data[1]["english"], "non_english": "دیگر"})
        cfg = PilotConfig(pairs=30)
        splits = split_pairs(data, cfg)
        self.assertEqual(splits, split_pairs(data, cfg))
        self.assertEqual(sum(map(len, splits.values())), 30)
        for lang in ("en", "fa"):
            all_text = [p[lang] for group in splits.values() for p in group]
            self.assertEqual(len(all_text), len(set(all_text)))
        with self.assertRaisesRegex(ValueError, "usable unique"):
            split_pairs(rows(5), cfg)

    def test_normalization_vocabulary_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = train_tokenizer((r[k] for r in rows(200) for k in ("english", "non_english")), Path(tmp), 256)
            tk = load_tokenizer(meta)
            self.assertEqual(tk("كيك می‌روم")["input_ids"], tk("کیک می روم")["input_ids"])
            self.assertNotEqual(tk("می‌روم")["input_ids"], tk("میروم")["input_ids"])
            long = tk("hello "*200, truncation=True, padding="max_length", max_length=128)["input_ids"]
            self.assertEqual(len(long), 128)
            self.assertEqual(long[-1], tk.sep_token_id)
            self.assertNotIn(tk.unk_token_id, tk("پژوهش ژاله ؟")["input_ids"])
            check_checkpoint_tokenizer({"tokenizer": meta}, meta)
            with self.assertRaises(ValueError):
                check_checkpoint_tokenizer({}, meta)
            Path(meta["path"]).write_text(Path(meta["path"]).read_text() + " ")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_tokenizer(meta)
            with self.assertRaisesRegex(ValueError, "add training text"):
                train_tokenizer(["hello سلام"], Path(tmp)/"short", 32000)

    def test_singleton_relational_loss_has_finite_gradient(self):
        student = torch.ones(1, 384, requires_grad=True)
        loss = relational_loss(student, student.detach())
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_retrieval_metrics_use_graded_gain_and_reject_missing_qrels(self):
        scores = [[0.1, 0.9, 0.2]]
        metrics = evaluation.retrieval_metrics(scores, [{1: 3, 2: 1}])
        self.assertEqual(metrics["ndcg@10"], 1)
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@10"], 1)
        with self.assertRaises(ValueError):
            evaluation.retrieval_metrics(scores, [{}])

    def test_pilot_configuration(self):
        cfg = load_config(ROOT / "training/distill/configs/pilot-fa-en-32k.yaml")
        self.assertEqual((cfg.train.vocab_size, cfg.train.d_model, cfg.train.n_layers), (32000, 256, 2))
        self.assertEqual(cfg.train.contrastive_w, 0)

    def test_hard_retrieval_reports_ties_and_failures(self):
        corpus = [{"id": "positive"}, {"id": "negative"}]
        query = {"id": "order", "text": "roles", "category": "role-order",
                 "relevant": {"positive": 1}, "contrast": "negative"}
        tied = evaluation.retrieval_details([[0.5, 0.5]], [query], corpus)
        self.assertTrue(tied["queries"][0]["contrast_tied"])
        self.assertEqual(tied["categories"]["role-order"]["contrast_accuracy"], 0)
        wrong = evaluation.retrieval_details([[0.3, 0.8]], [query], corpus)
        self.assertFalse(wrong["queries"][0]["top1_correct"])
        self.assertEqual(wrong["queries"][0]["first_relevant_rank"], 2)

    def test_heldout_audit_detects_normalized_leakage_and_sts_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fixture = out/"fixture.json"
            pairs = out/"pairs.json"
            fixture.write_text(json.dumps({"corpus": [{"id": "a", "text": "کیک می‌روم!"}],
                "queries": [{"id": "q", "text": "zero one two three four five six seven eight nine"}]}))
            pairs.write_text(json.dumps({"train": [{"id": "pair", "en": "one two three four five six seven eight", "fa": "كيك می روم"}], "val": []}))
            result = heldout_audit.audit(fixture, pairs)
            self.assertEqual(result["normalized_exact_collisions"][0]["lang"], "fa")
            self.assertEqual(result["shared_eight_word_spans"], ["one two three four five six seven eight"])
            additional = out/"additional.json"
            additional.write_text(json.dumps(["كيك می روم"]))
            result = heldout_audit.audit(fixture, pairs, additional)
            self.assertEqual(result["normalized_exact_collisions"][-1]["split"], "additional-train-val")
            additional.write_text('[42]')
            with self.assertRaises(ValueError):
                heldout_audit.audit(fixture, pairs, additional)
        result = heldout_sts.correlations([0.1, 0.3, 0.2], [1, 3, 2])
        self.assertAlmostEqual(result["spearman"], 1)
        self.assertAlmostEqual(result["pearson"], 1)
        with self.assertRaises(ValueError):
            heldout_sts.correlations([float("nan"), 1, 2], [1, 2, 3])

    def test_prepare_train_pack_roundtrip(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = {
                "wandb_group": "offline-test", "pilot": {"pairs": 40, "vocab_size": 256,
                    "output_dir": str(out/"cache"), "eval_pairs": 3, "test_ratio": 0.1},
                "train": {"vocab_size": 256, "d_model": 16, "n_layers": 1, "n_heads": 4,
                    "ffn_dim": 32, "output_dim": 384, "cache_dir": str(out/"cache"),
                    "cache_name": "test", "run_name": "offline-test", "batch_size": 4,
                    "epochs": 1, "enable_qat": True, "contrastive_w": 0, "relational_w": 0.1},
            }
            config_path = out/"config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            source = out/"pairs.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in rows(100)))
            def fake_teacher(texts, *args):
                generator = torch.Generator().manual_seed(42)
                return torch.nn.functional.normalize(torch.randn(len(texts), 384, generator=generator), dim=-1)
            with patch("prep.bilingual.HfApi") as api, patch("prep.bilingual.teacher_encode", side_effect=fake_teacher):
                api.return_value.model_info.return_value.sha = "offline-test-teacher"
                prepare(config_path, source, tokenizer_only=True)
                prepare(config_path, source)
                with self.assertRaises(FileExistsError):
                    prepare(config_path, source)
            splits, manifest = load_cache(out/"cache", "test")
            self.assertEqual({s["source"] for s in splits["train"]}, {"en", "fa"})
            train_ids = {s["pair_id"] for s in splits["train"]}
            self.assertFalse(train_ids & {s["pair_id"] for s in splits["test"]})
            cfg = TrainConfig(**config["train"])
            student = StudentEncoder(vocab_size=256, d_model=16, n_layers=1, n_heads=4, ffn_dim=32)
            loader = torch.utils.data.DataLoader(TernDataset(splits["train"][:8]), batch_size=4, collate_fn=collate_fn)
            val = torch.utils.data.DataLoader(TernDataset(splits["val"]), batch_size=4, collate_fn=collate_fn)
            with patch("trainer.wandb.log"):
                trainer = Trainer(student, loader, val, cfg, "cpu", out, tokenizer=manifest["tokenizer"], data_manifest=manifest)
                trainer.train()
            checkpoint = out/"checkpoint_ep1.pt"
            metrics = json.loads((out/"metrics.jsonl").read_text())
            self.assertEqual(metrics["epoch"], 1)
            self.assertGreater(metrics["epoch_seconds"], 0)
            self.assertTrue(0 <= metrics["val/loss"] <= 2)
            with self.assertRaisesRegex(ValueError, "different tokenizer"):
                _get_eval_inputs(out/"cache", "test", checkpoint={})
            _get_eval_inputs(out/"cache", "test", checkpoint=torch.load(checkpoint, weights_only=False))
            packed = out/"model.bin"
            pack(checkpoint, packed, "int4")
            check_model.check(packed, packed.with_suffix(".tokenizer.json"))
            source_model = _load_source_model(checkpoint, "int4")
            reference = UnpackedModel.from_bin(packed)
            batch = next(iter(val))
            with torch.no_grad():
                a = source_model(batch["input_ids"], batch["attention_mask"])
                b = reference.forward(batch["input_ids"], batch["attention_mask"])
            self.assertLess((a-b).abs().max().item(), 5e-4)
            self.assertEqual(torch.load(checkpoint, weights_only=False)["tokenizer"]["sha256"], manifest["tokenizer"]["sha256"])
            wrong = json.loads(packed.with_suffix(".tokenizer.json").read_text())
            vocab = wrong["model"]["vocab"]
            keys = [k for k in vocab if len(k) > 3 and k not in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")][:2]
            vocab[keys[0]], vocab[keys[1]] = vocab[keys[1]], vocab[keys[0]]
            wrong_path = out/"wrong.json"
            wrong_path.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "does not belong"):
                check_model.check(packed, wrong_path)


if __name__ == "__main__":
    unittest.main()
