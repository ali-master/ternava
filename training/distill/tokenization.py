"""The serialized tokenizer is shared by training, evaluation and Rust/WASM."""

import hashlib
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

MAX_LENGTH = 128
ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def bilingual_normalizer():
    # ZWNJ is a boundary, not a character to silently concatenate across.
    return normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("ي", "ی"), normalizers.Replace("ك", "ک"),
        normalizers.Replace("\u200c", " "),
        normalizers.BertNormalizer(lowercase=True, strip_accents=False),
    ])


def train_tokenizer(texts, destination: Path, vocab_size: int = 32_000) -> dict:
    """Train only on train-split text; reject undersized vocabularies."""
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = bilingual_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩"
                    "اآأإءئؤبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیةۀ؟،؛")
    tokenizer.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=alphabet, show_progress=True,
    ))
    # initial_alphabet alone does not reserve continuation pieces (e.g. ##ژ).
    required = list(dict.fromkeys(SPECIAL_TOKENS + alphabet + ["##" + c for c in alphabet]))
    required_set = set(required)
    learned = sorted(tokenizer.get_vocab(), key=tokenizer.token_to_id)
    tokens = required + [t for t in learned if t not in required_set]
    if len(tokens) < vocab_size:
        raise ValueError(f"Corpus produced {len(tokens)} tokens, need {vocab_size}; add training text")
    tokenizer.model = models.WordPiece(vocab={t: i for i, t in enumerate(tokens[:vocab_size])}, unk_token="[UNK]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(length=MAX_LENGTH, pad_id=0, pad_token="[PAD]")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "tokenizer.json"
    tokenizer.save(str(path))
    return tokenizer_metadata(path)


def tokenizer_metadata(path: Path) -> dict:
    path = Path(path).resolve()
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("The packer and engine require [PAD]=0")
    stored_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(stored_path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vocab_size": tokenizer.get_vocab_size(), "max_length": MAX_LENGTH}


def validate_tokenizer(metadata: dict) -> Path:
    path = Path(metadata["path"])
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Relative tokenizer path must stay inside the repository")
    elif not path.is_file():
        # Older checkpoints recorded absolute paths; keep their bytes unchanged.
        for parent in path.parents:
            for anchor in (Path("training/distill/cache"), Path("training/distill/runs"),
                           Path("engine/assets")):
                if parent.parts[-len(anchor.parts):] == anchor.parts:
                    candidate = (ROOT / anchor / path.relative_to(parent)).resolve()
                    if candidate.is_relative_to(ROOT) and candidate.is_file():
                        path = candidate
                        break
            else:
                continue
            break
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    actual = tokenizer_metadata(path)
    if any(actual[key] != metadata[key] for key in ("sha256", "vocab_size", "max_length")):
        raise ValueError("Tokenizer identity mismatch; rebuild the cache/checkpoint instead of reusing token IDs")
    return path


def load_tokenizer(metadata: dict | None = None):
    if metadata is None:
        path = Path(__file__).resolve().parents[2] / "engine/assets/tokenizer.json"
    else:
        path = validate_tokenizer(metadata)
    return PreTrainedTokenizerFast(tokenizer_file=str(path), model_max_length=MAX_LENGTH,
                                  pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")


def check_checkpoint_tokenizer(checkpoint: dict, metadata: dict | None) -> None:
    previous = checkpoint.get("tokenizer")
    if (previous or metadata) and (not previous or not metadata or previous["sha256"] != metadata["sha256"]):
        raise ValueError("Cannot warm-start/resume with a different tokenizer; train the new vocabulary from scratch")
