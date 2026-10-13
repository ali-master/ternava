"""Moving a checkout must preserve tokenizer identity and checkpoint bytes."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training/distill'), str(ROOT / 'training/pack')]
from config import Config, TrainConfig
import train
from model import StudentEncoder
from pack import pack
from tokenization import train_tokenizer, tokenizer_metadata, validate_tokenizer, load_tokenizer


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old = Path(self.tmp.name).resolve() / 'before'
        self.new = Path(self.tmp.name).resolve() / 'after'
        self.relative = Path('training/distill/cache/pilot/tokenizer/tokenizer.json')
        self.path = self.old / self.relative
        texts = [f'پرسش {i} درباره حساب کاربری question {i} about my account' for i in range(200)]
        with patch('tokenization.ROOT', self.old):
            self.meta = train_tokenizer(texts, self.path.parent, 256)
        self.legacy = {**self.meta, 'path': str(self.path)}
        self.old.rename(self.new)

    def test_relative_and_legacy_paths_load_after_real_directory_move(self):
        self.assertEqual(self.meta['path'], str(self.relative))
        original = dict(self.legacy)
        with patch('tokenization.ROOT', self.new):
            expected = load_tokenizer(self.meta)('سلام حساب من')['input_ids']
            self.assertEqual(load_tokenizer(self.legacy)('سلام حساب من')['input_ids'], expected)
            self.assertEqual(tokenizer_metadata(self.new / self.relative), self.meta)
        for relative in (Path('training/distill/runs/pilot/tokenizer.json'),
                         Path('engine/assets/tokenizer.json')):
            with self.subTest(relative=relative), patch('tokenization.ROOT', self.new):
                destination = self.new / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((self.new / self.relative).read_bytes())
                legacy = {**self.meta, 'path': str(self.old / relative)}
                self.assertEqual(validate_tokenizer(legacy), destination)
        self.assertEqual(self.legacy, original)
        self.assertFalse(self.old.exists())

    def test_relocation_does_not_bypass_identity_or_accept_unrelated_paths(self):
        with patch('tokenization.ROOT', self.new):
            for field, value in [('sha256', '0'*64), ('vocab_size', 255), ('max_length', 64)]:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    validate_tokenizer({**self.legacy, field: value})
            with self.assertRaisesRegex(ValueError, 'inside the repository'):
                validate_tokenizer({**self.meta, 'path': '../outside/tokenizer.json'})
            with self.assertRaises(FileNotFoundError):
                validate_tokenizer({**self.meta, 'path': str(self.old/'unrelated/tokenizer.json')})
            (self.new / self.relative).write_text((self.new / self.relative).read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_tokenizer(self.legacy)

    def test_training_entrypoint_snapshots_relocated_tokenizer(self):
        cfg = Config(wandb_group='offline', train=TrainConfig(
            vocab_size=256, d_model=16, n_layers=1, n_heads=4, ffn_dim=32,
            runs_dir=self.new/'training/distill/runs', run_name='relocation', cache_name='fixture'))
        sample = {'input_ids': torch.zeros(128, dtype=torch.long),
                  'teacher_emb': torch.ones(384)}
        splits = {name: [sample] for name in ('train', 'val', 'test')}
        with (patch('tokenization.ROOT', self.new),
              patch('train.load_config', return_value=cfg),
              patch('train.load_cache', return_value=(splits, {'tokenizer': self.legacy})),
              patch('train.best_device', return_value='cpu'),
              patch('train.git_commit', return_value='abc1234'),
              patch('train.wandb'), patch('train.Trainer') as trainer):
            train.main(self.new/'config.yaml')
            metadata = trainer.call_args.kwargs['tokenizer']
            copied = validate_tokenizer(metadata)
            self.assertEqual(copied.read_bytes(), (self.new/self.relative).read_bytes())
            self.assertFalse(Path(metadata['path']).is_absolute())
            trainer.return_value.train.assert_called_once_with()

    def test_packer_resolves_legacy_checkpoint_without_rewriting_it(self):
        torch.set_num_threads(2)
        cfg = dict(vocab_size=256, d_model=16, n_layers=1, n_heads=4,
                   ffn_dim=32, output_dim=384, dropout=0.0)
        model = StudentEncoder(**cfg)
        checkpoint = self.new / 'checkpoint.pt'
        torch.save({'config': {**cfg, 'enable_qat': True}, 'epoch': 0,
                    'model_state': model.state_dict(), 'tokenizer': self.legacy}, checkpoint)
        original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        packed = self.new / 'packed/model.bin'
        with patch('tokenization.ROOT', self.new):
            pack(checkpoint, packed, 'int4')
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), original)
        self.assertEqual(packed.with_suffix('.tokenizer.json').read_bytes(),
                         (self.new / self.relative).read_bytes())
        manifest = json.loads(packed.with_suffix('.bin.json').read_text())
        self.assertEqual(manifest['extra']['tokenizer']['path'], 'model.tokenizer.json')


if __name__ == '__main__':
    unittest.main()
