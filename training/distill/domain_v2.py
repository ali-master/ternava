"""Domain listwise QAT on the v2 pilot, gated by tool-search retention.

    MIKROTIK_MCP=... bun eval/toolsearch/diag.ts --dump /tmp/toolsearch.json
    python training/distill/domain_v2.py --toolsearch /tmp/toolsearch.json [--sts DIR] [--smoke]

--sts DIR adds CoSENT on the STS train splits in DIR (stsb-en-train.jsonl and
synthetic-fa-train.jsonl, one {"sentence1", "sentence2", "score"} object per line).

Unlike domain.py this needs no round2/3/4 cache: the only supervision is the authored
docs/emoji queries, and retention is the frozen v2 pilot's own geometry on OPUS replay.
The earlier logic/ranking terms are left out on purpose: they cost tool-search quality.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval/toolsearch"))
import checkpoints  # noqa: E402
from domain import listwise_loss  # noqa: E402
from gates import rank_suite  # noqa: E402
from native import int4_ste  # noqa: E402
from prep.domain import domain_rows  # noqa: E402
from prep.round2 import normalize  # noqa: E402
from reasoning import geometry_loss  # noqa: E402
from round2 import cosent_loss  # noqa: E402

RUNS = ROOT / "training/distill/runs"
SOURCE = RUNS / "mini-fa-en-32k-pilot-v2-9f183ec/checkpoint_ep10.pt"
PAIRS = ROOT / "training/distill/cache/mini-fa-en-32k-v2/pairs.json"
CORPUS = ROOT / "eval/quality/corpus"
RECIPE = dict(seed=613, epochs=6, steps=250, embedding_lr=1.2e-4, body_lr=3e-6, domain_batch=24, replay_batch=48,
              sts_batch=16, sts_weight=.1,
              loss="1 domain listwise + 1 geometry retention to the frozen v2 pilot on OPUS replay"
                   " (+ sts_weight CoSENT per language on STS train pairs when --sts is given)",
              selection="Highest mean docs/emoji validation NDCG among epochs whose tool-search MRR "
                        "(EN, FA informal, FA formal) each stay within .02 of the v2 pilot. No fresh-test selection.")


def sealed_texts():
    """Every query of the frozen challenge corpora; none may appear in training."""
    texts = set()
    for name in ["persian-domain-hard", "persian-reasoning-hard", "persian-compositional-hard"]:
        for suite in json.loads((CORPUS / f"{name}.json").read_text())["suites"].values():
            texts.update(normalize(q["text"]) for q in suite["queries"])
    for queries in json.loads((CORPUS / "persian-native-demos.json").read_text())["queries"].values():
        texts.update(normalize(q["text"]) for q in queries)
    return texts


def assess(path, fixtures, tools):
    """Score a saved checkpoint exactly as shipped (ternary + int4), on CPU."""
    model = checkpoints.load(path, True)
    result = {}
    for task, f in fixtures.items():
        top1, ndcg, _, _ = rank_suite(model, f["corpus"], f["queries"])
        result[f"{task}_val_top1"], result[f"{task}_val_ndcg"] = int(top1) / len(f["queries"]), float(ndcg)
    docs = checkpoints.embed(model, tools["texts"])
    for k, lang in enumerate(["en", "fa_informal", "fa_formal"]):
        order = (checkpoints.embed(model, [c[k] for c in tools["cases"]]) @ docs.T).argsort(dim=1, descending=True)
        ranks = [[tools["names"][i] for i in row.tolist()].index(c[3]) + 1 for row, c in zip(order, tools["cases"])]
        result[f"tools_{lang}_mrr"] = sum(1 / r for r in ranks) / len(ranks)
        result[f"tools_{lang}_top5"] = sum(r <= 5 for r in ranks)
    result["selection_score"] = (result["docs_val_ndcg"] + result["emoji_val_ndcg"]) / 2
    return result


def train(toolsearch: Path, smoke: bool, sts_dir: Path | None = None):
    torch.manual_seed(RECIPE["seed"])
    rng = random.Random(RECIPE["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run = RUNS / ("mini-fa-en-32k-v2-domain" + ("-sts" if sts_dir else "") + ("-smoke" if smoke else ""))
    if run.exists():
        raise FileExistsError("Preserve all runs")
    tools = json.loads(toolsearch.read_text())
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    tokenizer = checkpoints.tokenizer_for(source)

    def tokens(texts):
        b = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return b["input_ids"].to(device), b["attention_mask"].to(device)

    corpora, rows = domain_rows()
    assert not sealed_texts() & {normalize(q["text"]) for split in rows.values() for q in split}
    fixtures = {t: dict(corpus=corpora[t], queries=[q for q in rows["val"] if q["task"] == t]) for t in corpora}
    tasks = {}
    for task, corpus in corpora.items():
        lookup = {d["id"]: i for i, d in enumerate(corpus)}
        queries = [q for q in rows["train"] if q["task"] == task]
        relevant = torch.zeros(len(queries), len(corpus), dtype=torch.bool)
        for i, q in enumerate(queries):
            relevant[i, [lookup[k] for k in q["relevant"]]] = True
        tasks[task] = (tokens([d["text"] for d in corpus]), [q["text"] for q in queries], relevant.to(device))
    replay = [p[lang] for p in json.loads(PAIRS.read_text())["train"] for lang in ("en", "fa")]
    scored = [[json.loads(line) for line in (sts_dir / f"{name}-train.jsonl").open()]
              for name in ("stsb-en", "synthetic-fa")] if sts_dir else []

    model = checkpoints.load(SOURCE, False)  # fp32 embedding; int4 is applied straight-through below
    model.embedding.register_forward_hook(int4_ste)
    model.to(device)
    anchor = checkpoints.load(SOURCE, True).to(device).requires_grad_(False)
    lrs = [RECIPE["embedding_lr"], RECIPE["body_lr"]]
    optimizer = torch.optim.AdamW([
        {"params": [model.embedding.weight], "lr": lrs[0]},
        {"params": [p for n, p in model.named_parameters() if n != "embedding.weight"], "lr": lrs[1]}], weight_decay=0)

    run.mkdir(parents=True)
    (run / "recipe.json").write_text(json.dumps(RECIPE, indent=2) + "\n")
    baseline = assess(SOURCE, fixtures, tools)
    (run / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print("BASELINE", json.dumps(baseline), flush=True)
    epochs, steps = (1, 2) if smoke else (RECIPE["epochs"], RECIPE["steps"])
    best = -1.0
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        model.train()
        for step in range(steps):
            corpus_tokens, queries, relevant = tasks[["docs", "emoji"][step % 2]]
            pick = torch.randint(len(queries), (RECIPE["domain_batch"],))
            rank = listwise_loss(model(*tokens([queries[i] for i in pick])), model(*corpus_tokens), relevant[pick])
            batch = tokens(rng.sample(replay, RECIPE["replay_batch"]))
            with torch.no_grad():
                target = anchor(*batch)
            retention = geometry_loss(model(*batch), target)
            loss = rank + retention
            for rows in scored:  # one CoSENT term per language
                pairs = rng.sample(rows, RECIPE["sts_batch"])
                v = torch.stack([model(*tokens([r[k] for r in pairs])) for k in ("sentence1", "sentence2")], dim=1)
                loss = loss + RECIPE["sts_weight"] * cosent_loss(v, torch.tensor([r["score"] for r in pairs], device=device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            progress = (epoch - 1) * steps + step
            factor = min(1, (progress + 1) / 50) * (1 - progress / (epochs * steps))
            for group, lr in zip(optimizer.param_groups, lrs):
                group["lr"] = lr * factor
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if (step + 1) % 50 == 0 or smoke:
                print(f"epoch={epoch} step={step + 1} domain={float(rank.detach()):.4f} retention={float(retention.detach()):.4f}", flush=True)
        path = run / f"epoch{epoch}.pt"
        torch.save(dict(epoch=epoch, model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        config={**source["config"], "run_name": run.name}, tokenizer=source["tokenizer"],
                        recipe=RECIPE, source_checkpoint=str(SOURCE.relative_to(ROOT))), path)
        metrics = assess(path, fixtures, tools)
        metrics.update(epoch=epoch, seconds=time.monotonic() - start, eligible=all(
            metrics[f"tools_{lang}_mrr"] >= baseline[f"tools_{lang}_mrr"] - .02 for lang in ["en", "fa_informal", "fa_formal"]))
        print("EPOCH", json.dumps(metrics), flush=True)
        with (run / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if metrics["eligible"] and metrics["selection_score"] > best:
            best = metrics["selection_score"]
            (run / "best.json").write_text(json.dumps(dict(checkpoint=path.name, validation=metrics), indent=2) + "\n")
    print("VERDICT", json.dumps(dict(eligible=best >= 0, best=best, baseline=baseline["selection_score"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolsearch", type=Path, required=True, help="dump from eval/toolsearch/diag.ts --dump")
    parser.add_argument("--sts", type=Path, help="directory with STS train splits; adds a CoSENT term")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(args.toolsearch, args.smoke, args.sts)
