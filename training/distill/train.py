"""Phase 2/3 — training entry point.

Reads a YAML config, loads the cached dataset (output of prep/prepare.py),
builds the StudentEncoder, and hands off to the Trainer.

Phase 2 (fp32 baseline): `--config configs/micro-fp32.yaml`
Phase 3 (QAT, future):   `--config configs/micro.yaml` once QAT is wired

Usage:
    python train.py --config configs/smoke-fp32.yaml      # ~30s pipeline check
    python train.py --config configs/micro-fp32.yaml      # the real run

To run without W&B:
    WANDB_MODE=disabled python train.py --config ...
"""

import argparse
import random
import shutil
import subprocess
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from config  import load_config
from data    import TernDataset, collate_fn, load_cache
from model   import StudentEncoder
from trainer import Trainer
from tokenization import check_checkpoint_tokenizer, validate_tokenizer, tokenizer_metadata


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    if cfg.train is None:
        raise ValueError(f"Config {config_path} has no `train:` section.")
    tcfg = cfg.train
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)

    # ── 1. Load cached dataset ───────────────────────────────────────────────
    print(f"→ Loading cache: {tcfg.cache_dir}/{tcfg.cache_name}.*")
    splits, manifest = load_cache(tcfg.cache_dir, tcfg.cache_name)
    tokenizer_info = manifest.get("tokenizer")
    if tokenizer_info:
        source_tokenizer_path = validate_tokenizer(tokenizer_info)
        if tokenizer_info["vocab_size"] != tcfg.vocab_size:
            raise ValueError("Training vocabulary differs from cached token IDs")
    elif tcfg.vocab_size != 30522:
        raise ValueError("Custom vocabularies require tokenizer provenance in the cache")
    for name, samples in splits.items():
        if not samples:
            raise ValueError(f"Empty {name} split")
        for sample in samples:
            if (sample["input_ids"].numel() != 128 or sample["input_ids"].min() < 0
                    or sample["input_ids"].max() >= tcfg.vocab_size
                    or sample["teacher_emb"].shape != (tcfg.output_dim,)
                    or not torch.isfinite(sample["teacher_emb"]).all()):
                raise ValueError(f"Invalid token IDs/teacher targets in {name} cache")
    print(f"  train: {len(splits['train']):,}  val: {len(splits['val']):,}  test: {len(splits['test']):,}")

    train_loader = DataLoader(
        TernDataset(splits["train"]),
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TernDataset(splits["val"]),
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── 2. Build model ───────────────────────────────────────────────────────
    device = best_device()
    print(f"\n→ Building model (device={device})")
    model = StudentEncoder(
        vocab_size = tcfg.vocab_size,
        d_model    = tcfg.d_model,
        n_layers   = tcfg.n_layers,
        n_heads    = tcfg.n_heads,
        ffn_dim    = tcfg.ffn_dim,
        output_dim = tcfg.output_dim,
        dropout    = tcfg.dropout,
        position_embeddings = tcfg.position_embeddings,
    )
    print(f"  parameters: {model.count_parameters():,}")

    # ── 2b. Warm-start vs resume ─────────────────────────────────────────────
    # init_from:    cross-recipe warm-start. Loads model_state only; optimizer,
    #               scheduler, and global_step reset. Used for fp32 → QAT init.
    # resume_from:  same-recipe continuation. Loads model + optimizer + scheduler
    #               + global_step; loop picks up at ckpt["epoch"] + 1.
    # Both must happen BEFORE Trainer ctor swaps to BitLinear, so BitLinear's
    # shadow weights inherit the loaded values.
    resume_ckpt = None
    if tcfg.resume_from is not None and tcfg.init_from is not None:
        raise ValueError("Cannot set both `resume_from` and `init_from`; pick one.")
    if tcfg.init_from is not None:
        print(f"\n→ Warm-starting from {tcfg.init_from}")
        ckpt = torch.load(tcfg.init_from, weights_only=False, map_location="cpu")
        check_checkpoint_tokenizer(ckpt, tokenizer_info)
        model.load_state_dict(ckpt["model_state"])
        print(f"  loaded weights from epoch {ckpt['epoch']}")
    elif tcfg.resume_from is not None:
        print(f"\n→ Resuming from {tcfg.resume_from}")
        resume_ckpt = torch.load(tcfg.resume_from, weights_only=False, map_location="cpu")
        check_checkpoint_tokenizer(resume_ckpt, tokenizer_info)
        if (resume_ckpt.get("data_manifest") or {}).get("pairs_sha256") != manifest.get("pairs_sha256"):
            raise ValueError("Resume requires the same prepared dataset")
        model.load_state_dict(resume_ckpt["model_state"])
        print(f"  loaded model state from epoch {resume_ckpt['epoch']}")

    # ── 3. Run directory ─────────────────────────────────────────────────────
    short_sha = git_commit()[:7]
    full_run_name = f"{tcfg.run_name}-{short_sha}"
    run_dir = Path(tcfg.runs_dir) / full_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("checkpoint_ep*.pt")) and tcfg.resume_from is None:
        raise FileExistsError(f"Run already has checkpoints: {run_dir}; choose a new run_name or resume")
    if tokenizer_info:
        snapshot = run_dir / "tokenizer.json"
        if source_tokenizer_path != snapshot.resolve():
            shutil.copyfile(source_tokenizer_path, snapshot)
        tokenizer_info = tokenizer_metadata(snapshot)
    print(f"  run dir: {run_dir}")

    # ── 4. W&B (levels 1-3: project, run, job_type) ──────────────────────────
    job_type = "train-qat" if tcfg.enable_qat else "train-fp32"
    wandb.init(
        project = cfg.wandb_project,
        group   = cfg.wandb_group,
        job_type= job_type,
        name    = full_run_name,
        tags    = cfg.wandb_tags,
        config  = {
            **tcfg.model_dump(mode="json"),
            "code_commit":   git_commit(),
            "data_manifest": manifest,
            "device":        device,
        },
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    start_epoch = resume_ckpt["epoch"] if resume_ckpt is not None else 0
    trainer = Trainer(model, train_loader, val_loader, tcfg, device, run_dir, start_epoch=start_epoch,
                      tokenizer=tokenizer_info, data_manifest=manifest)
    if resume_ckpt is not None:
        trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        trainer.scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        trainer._global_step = resume_ckpt["scheduler_state"]["last_epoch"]
        print(f"  ↳ optimizer + scheduler state restored")
        print(f"  ↳ resuming at step {trainer._global_step:,}  LR={trainer.scheduler.get_last_lr()[0]:.2e}")
    trainer.train()

    wandb.finish()
    print(f"\n✓ Done. Checkpoints in {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2/3 — train the student model")
    parser.add_argument("--config", type=Path, required=True, help="path to YAML config")
    args = parser.parse_args()
    main(args.config)
