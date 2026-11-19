# /// script
# dependencies = ["torch", "numpy", "transformers", "tokenizers", "bitlinear==2.4.6"]
# ///
"""Where does tool-search quality go? Run diag.ts's benchmark on training checkpoints.

    MIKROTIK_MCP=... bun diag.ts --dump /tmp/toolsearch.json
    uv run checkpoints.py /tmp/toolsearch.json [run/checkpoint.pt ...]

Each checkpoint is scored as shipped (ternary body + int4 embedding), with the
embedding left fp32 (isolates the int4 cost), and with an fp32 body. The fp32-body
row is only meaningful for QAT-warmup checkpoints; the ternary rows are not. `english-safe.pt` int4 must
reproduce the WASM numbers from diag.ts; if it does not, this script is wrong.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "training/distill/runs"
sys.path.insert(0, str(ROOT / "training/distill"))
import ternary_qat  # noqa: E402
from model import StudentEncoder  # noqa: E402

DEFAULT = [  # training order
    "mini-fa-en-32k-pilot-23da804/checkpoint_ep2.pt",
    "mini-fa-en-32k-pilot-23da804/checkpoint_ep10.pt",
    "mini-fa-en-round2/best.pt",
    "mini-fa-en-round3/best.pt",
    "mini-fa-en-round4/diagnostic-best.pt",
    "mini-fa-en-domain/diagnostic-best.pt",
    "mini-fa-en-reasoning/diagnostic-best.pt",
    "mini-fa-en-reasoning/english-safe.pt",  # shipped in @usex/ternava 0.2.0
]
ARCH = ["vocab_size", "d_model", "n_layers", "n_heads", "ffn_dim", "output_dim", "position_embeddings"]
PILOT_TOKENIZER = RUNS / "mini-fa-en-32k-pilot-23da804/tokenizer.json"


def tokenizer_for(ckpt):
    # Old checkpoints store a stale absolute path to the shared pilot tokenizer; newer ones a repo-relative path.
    path = ROOT / ckpt["tokenizer"]["path"]
    return PreTrainedTokenizerFast(tokenizer_file=str(path if path.is_file() else PILOT_TOKENIZER),
                                   pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")


def load(path: Path, int4: bool, ternary: bool = True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StudentEncoder(**{k: ckpt["config"][k] for k in ARCH if k in ckpt["config"]}, dropout=0)
    model.load_state_dict(ckpt["model_state"])  # before the swap, as pack.py does
    ternary_qat.swap(model)
    ternary_qat.set_lambda(model, 1 if ternary else 0)  # 0 = fp32 body, as in QAT warmup epochs
    if int4:
        ternary_qat.int4_quantize_embedding_(model)
    model.tokenizer = tokenizer_for(ckpt)
    return model.train(False)


@torch.no_grad()
def embed(model, texts):
    out = []
    for i in range(0, len(texts), 64):
        b = model.tokenizer(texts[i:i + 64], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        out.append(model(b["input_ids"], b["attention_mask"]))
    return torch.nn.functional.normalize(torch.cat(out), dim=-1)


def score(model):
    docs = embed(model, data["texts"])
    pairs = [(i, j) for i in range(0, len(docs), 7) for j in range(i + 3, len(docs), 97)]  # as diag.ts
    unrelated = sum(float(docs[i] @ docs[j]) for i, j in pairs) / len(pairs)
    medians, top5, mrr = [], [], []
    for k in range(3):
        order = (embed(model, [c[k] for c in cases]) @ docs.T).argsort(dim=1, descending=True)
        ranks = sorted([names[i] for i in row.tolist()].index(c[3]) + 1 for row, c in zip(order, cases))
        medians.append(ranks[len(ranks) // 2])
        top5.append(sum(r <= 5 for r in ranks))
        mrr.append(sum(1 / r for r in ranks) / len(ranks))  # steadier than the median on 12 queries
    sim = lambda a, b: float(embed(model, [a])[0] @ embed(model, [b])[0])
    return (f"rank EN {medians[0]:>3} FAinf {medians[1]:>3} FAformal {medians[2]:>3} | top5 {top5[0]}/{top5[1]}/{top5[2]}"
            f" | mrr {mrr[0]:.2f}/{mrr[1]:.2f}/{mrr[2]:.2f}"
            f" | unrelated {unrelated:.2f} | order {sim('روتر را خاموش کن', 'خاموش کن را روتر'):.2f}"
            f" | خاموش~روشن {sim('روتر را خاموش کن', 'روتر را روشن کن'):.2f}"
            f" | enable~disable {sim('enable interface', 'disable interface'):.2f}")


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    names, cases = data["names"], data["cases"]
    for rel in sys.argv[2:] or DEFAULT:
        print(rel)
        print(f"  fp32 body, emb fp32: {score(load(RUNS / rel, False, ternary=False))}", flush=True)
        for int4 in (False, True):
            print(f"  ternary,   emb {'int4' if int4 else 'fp32'}: {score(load(RUNS / rel, int4))}", flush=True)
