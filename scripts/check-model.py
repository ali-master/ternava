"""Check the model/tokenizer pair before compiling a self-contained WASM."""
import hashlib
import json
import struct
import sys
from pathlib import Path


def check(model: Path, tokenizer: Path):
    blob = model.read_bytes()
    if len(blob) < 64 or blob[:4] != b"TERN" or hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
        raise ValueError("Invalid model header/checksum")
    vocab_size = struct.unpack_from("<I", blob, 8)[0]
    if struct.unpack_from("<H", blob, 20)[0] != 128:
        raise ValueError("Engine only supports max_seq_len=128")
    raw = tokenizer.read_bytes()
    tk = json.loads(raw)
    vocab = tk["model"]["vocab"]
    if tk["model"]["type"] != "WordPiece" or len(vocab) != vocab_size or vocab.get("[PAD]") != 0:
        raise ValueError("Model/tokenizer vocabulary mismatch")
    if sorted(vocab.values()) != list(range(vocab_size)):
        raise ValueError("Token IDs must be contiguous")
    if any(t["id"] != vocab.get(t["content"]) for t in tk.get("added_tokens", [])):
        raise ValueError("Added token IDs differ from model vocabulary")
    sidecar = model.with_suffix(model.suffix + ".json")
    if sidecar.exists():
        manifest = json.loads(sidecar.read_text())
        if manifest["bin_sha256"] != hashlib.sha256(blob).hexdigest():
            raise ValueError("Model sidecar checksum mismatch")
        metadata = (manifest.get("extra") or {}).get("tokenizer")
    else:
        metadata = None
    if metadata:
        if metadata["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Tokenizer does not belong to this model")
    else:
        legacy = Path(__file__).resolve().parents[1] / "engine/assets/tokenizer.json"
        if vocab_size != 30522 or raw != legacy.read_bytes():
            raise ValueError("Custom tokenizer requires a model sidecar with its SHA256")


if __name__ == "__main__":
    check(Path(sys.argv[1]), Path(sys.argv[2]))
