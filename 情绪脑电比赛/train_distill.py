from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from emotion_merps.distill import (
    GatedTeacherFusion,
    StudentTeacherDistiller,
    gate_entropy,
    load_teacher_cache,
)
from emotion_merps.features import discover_subjects, load_training_features, standardize_from_train
from emotion_merps.model import ASACRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MER-PS with optional MTDP-style distillation.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--output-dir", default="experiments/checkpoints/emotion_mtdp")
    parser.add_argument("--teacher-cache", default=None, help="Optional NPZ with sample_ids and teacher embeddings.")
    parser.add_argument(
        "--teacher-keys",
        default="emotion,eeg,fnirs",
        help="Comma-separated NPZ arrays to use as teachers, e.g. emotion,eeg,fnirs.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--cheb-order", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--teacher-fusion-dim", type=int, default=256)
    parser.add_argument("--teacher-hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.num_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_subjects = discover_subjects(args.data_root)
    val_subjects = all_subjects[-4:]
    train_subjects = all_subjects[:-4]

    print("Loading MER-PS features", flush=True)
    eeg, fnirs, y, sample_subjects, subject_names, sample_ids = load_training_features(
        args.data_root,
        include_sample_ids=True,
        verbose=True,
    )
    train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
    val_idx = np.flatnonzero(np.isin(sample_subjects, val_subjects))
    train_idx = subset_indices(train_idx, args.limit_train_samples, args.seed)
    val_idx = subset_indices(val_idx, args.limit_val_samples, args.seed + 1)

    teacher_arrays = None
    teacher_keys: tuple[str, ...] = ()
    teacher_dims: list[int] = []
    if args.teacher_cache:
        teacher_keys = tuple(key.strip() for key in args.teacher_keys.split(",") if key.strip())
        cache = load_teacher_cache(args.teacher_cache, sample_ids, teacher_keys)
        teacher_arrays = cache.arrays
        teacher_dims = cache.dims

    eeg, fnirs, stats = standardize_from_train(eeg, fnirs, train_idx)
    train_loader = make_loader(eeg, fnirs, y, train_idx, args.batch_size, True, teacher_arrays)
    val_loader = make_loader(eeg, fnirs, y, val_idx, args.batch_size, False, teacher_arrays)

    model_config = {
        "eeg_nodes": int(eeg.shape[1]),
        "eeg_features": int(eeg.shape[2]),
        "fnirs_nodes": int(fnirs.shape[1]),
        "fnirs_features": int(fnirs.shape[2]),
        "output_dim": 2,
        "hidden_dim": args.hidden_dim,
        "cheb_order": args.cheb_order,
        "heads": args.heads,
        "dropout": args.dropout,
        "projection_dim": args.projection_dim,
        "temperature": args.temperature,
    }
    model = ASACRegressor(**model_config).to(device)

    teacher_fusion = None
    distiller = None
    parameters = list(model.parameters())
    if teacher_arrays is not None:
        teacher_fusion = GatedTeacherFusion(
            teacher_dims,
            fusion_dim=args.teacher_fusion_dim,
            hidden_dim=args.teacher_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        distiller = StudentTeacherDistiller(
            student_dim=model.embedding_dim,
            teacher_dim=args.teacher_fusion_dim,
            hidden_dim=args.teacher_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        parameters += list(teacher_fusion.parameters()) + list(distiller.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    summary = {
        "subjects": subject_names,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "model_config": model_config,
        "teacher_cache": args.teacher_cache,
        "teacher_keys": teacher_keys,
        "teacher_dims": teacher_dims,
    }
    print(json.dumps(summary, indent=2), flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            contrastive_weight=args.contrastive_weight,
            distill_weight=args.distill_weight,
            gate_entropy_weight=args.gate_entropy_weight,
            teacher_fusion=teacher_fusion,
            distiller=distiller,
            epoch=epoch,
            log_every=args.log_every,
        )
        val_stats = evaluate(model, val_loader, device, epoch=epoch, log_every=args.log_every)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_mse={val_stats['mse']:.6f} val_raw_mae={val_stats['raw_mae']:.3f}",
            flush=True,
        )
        if val_stats["mse"] < best_val:
            best_val = val_stats["mse"]
            save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                stats,
                train_subjects,
                val_subjects,
                args,
                val_stats,
                teacher_keys,
                teacher_dims,
            )
    print(f"Saved {checkpoint_path}", flush=True)


def subset_indices(indices: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or indices.size <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def make_loader(
    eeg: np.ndarray,
    fnirs: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    teacher_arrays: tuple[np.ndarray, ...] | None = None,
) -> DataLoader:
    tensors = [
        torch.from_numpy(np.ascontiguousarray(eeg[indices])).float(),
        torch.from_numpy(np.ascontiguousarray(fnirs[indices])).float(),
        torch.from_numpy(np.ascontiguousarray(y[indices])).float(),
    ]
    if teacher_arrays is not None:
        tensors.extend(
            torch.from_numpy(np.ascontiguousarray(array[indices])).float()
            for array in teacher_arrays
        )
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(
    model: ASACRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    contrastive_weight: float,
    distill_weight: float,
    gate_entropy_weight: float,
    teacher_fusion: GatedTeacherFusion | None,
    distiller: StudentTeacherDistiller | None,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    if teacher_fusion is not None:
        teacher_fusion.train()
    if distiller is not None:
        distiller.train()

    total = 0
    loss_sum = 0.0
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        eeg, fnirs, target, *teachers = batch
        eeg = eeg.to(device)
        fnirs = fnirs.to(device)
        target = target.to(device)
        teachers = [teacher.to(device) for teacher in teachers]

        optimizer.zero_grad(set_to_none=True)
        prediction, contrastive_loss, embeddings = model(eeg, fnirs, return_embeddings=True)
        loss = F.mse_loss(prediction, target) + contrastive_weight * contrastive_loss

        if teacher_fusion is not None and distiller is not None and teachers:
            fused_teacher, weights = teacher_fusion(teachers)
            loss = loss + distill_weight * distiller(embeddings["pooled"], fused_teacher)
            if gate_entropy_weight:
                loss = loss - gate_entropy_weight * gate_entropy(weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = int(target.size(0))
        total += batch_size
        loss_sum += float(loss.detach()) * batch_size
        if batch_index == 1 or batch_index == total_batches or batch_index % log_every == 0:
            print(
                f"[train] epoch {epoch} batch {batch_index}/{total_batches} "
                f"loss={loss_sum / max(total, 1):.6f}",
                flush=True,
            )
    return loss_sum / max(total, 1)


@torch.no_grad()
def evaluate(
    model: ASACRegressor,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    mse_sum = 0.0
    raw_abs_sum = 0.0
    raw_count = 0
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        eeg, fnirs, target = batch[:3]
        eeg = eeg.to(device)
        fnirs = fnirs.to(device)
        target = target.to(device)
        prediction, _ = model(eeg, fnirs)
        batch_size = int(target.size(0))
        total += batch_size
        mse_sum += float(F.mse_loss(prediction, target, reduction="sum"))
        raw_abs_sum += float(torch.abs((prediction - target) * 254.0).sum())
        raw_count += int(target.numel())
        if batch_index == 1 or batch_index == total_batches or batch_index % log_every == 0:
            print(f"[valid] epoch {epoch} batch {batch_index}/{total_batches}", flush=True)
    return {
        "mse": mse_sum / max(raw_count, 1),
        "raw_mae": raw_abs_sum / max(raw_count, 1),
    }


def save_checkpoint(
    path: Path,
    model: ASACRegressor,
    model_config: dict[str, object],
    stats: dict[str, np.ndarray],
    train_subjects: list[str],
    val_subjects: list[str],
    args: argparse.Namespace,
    val_stats: dict[str, float],
    teacher_keys: tuple[str, ...],
    teacher_dims: list[int],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "standardization": {key: value.tolist() for key, value in stats.items()},
            "target_names": ("valence", "arousal"),
            "label_scale": {"min": 1.0, "max": 255.0},
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "training_args": vars(args),
            "teacher_keys": teacher_keys,
            "teacher_dims": teacher_dims,
            "validation": val_stats,
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
