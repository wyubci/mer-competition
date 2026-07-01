from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.graph_mamba import GraphMambaResidualRegressor  # noqa: E402
from tools.run_iteration_experiments import (  # noqa: E402
    expand_subjects,
    load_labels,
    predict_video_time_mean,
    score,
    smooth_predictions,
)


SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph-Mamba residual model on MER-PS.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_002_graph_mamba.json")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/graph_mamba/best_model.pt")
    parser.add_argument("--target-mode", choices=("all", "valence", "arousal"), default="valence")
    parser.add_argument(
        "--input-modality",
        choices=("both", "eeg", "fnirs", "none"),
        default="both",
        help="Modality ablation switch. Non-selected signal tensors are zeroed before the model.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--graph-hidden", type=int, default=32)
    parser.add_argument(
        "--graph-encoder",
        choices=(
            "adaptive",
            "signed",
            "functional",
            "hybrid_functional",
            "sparse_functional",
            "sparse_hybrid_functional",
        ),
        default="adaptive",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("pool", "cross_asac", "modal_gate", "local_global", "pool_local_global"),
        default="pool",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--cheb-order", type=int, default=2)
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument(
        "--temporal-block",
        choices=(
            "mamba",
            "gated_ssm",
            "conv_mixer",
            "patch_tst",
            "timesnet",
            "itransformer",
            "timemixer",
            "fourier",
            "ssm_itransformer",
            "time_itransformer",
            "hybrid_temporal",
        ),
        default="mamba",
    )
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument(
        "--multiscale-windows",
        default="1",
        help="Comma-separated temporal feature windows per trial, e.g. 1,3,5,9. "
        "Window 1 keeps the original per-second feature.",
    )
    parser.add_argument("--scale-fusion", choices=("concat", "gated"), default="concat")
    parser.add_argument(
        "--feature-norm",
        choices=("none", "trial", "subject"),
        default="none",
        help="Unsupervised feature normalization scope applied after cache loading.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="Per-trial training-time probability of dropping each modality. Disabled at validation.",
    )
    parser.add_argument(
        "--teacher-cache",
        default=None,
        help="Optional NPZ teacher cache aligned by sample_id. Used only during training.",
    )
    parser.add_argument(
        "--teacher-keys",
        default="semantic",
        help="Comma-separated teacher arrays from --teacher-cache, e.g. semantic or emotion,sam,affectgpt.",
    )
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--teacher-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--pretrained-graph",
        default=None,
        help="Optional masked-graph pretraining checkpoint with encoder weights.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--residual-target-scale",
        type=float,
        default=1.0,
        help="Scale residual regression targets during training. Evaluation still maps output by 127.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    labels = load_labels(Path(args.data_root), train_subjects + val_subjects)
    train_label_ids = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_label_ids]).astype(
        np.float32
    )
    teacher_keys: tuple[str, ...] = ()
    teacher_lookup: dict[str, np.ndarray] | None = None
    teacher_dim = 0
    if args.teacher_cache:
        teacher_keys = tuple(key.strip() for key in args.teacher_keys.split(",") if key.strip())
        teacher_lookup, teacher_dim = load_teacher_lookup(Path(args.teacher_cache), teacher_keys)

    multiscale_windows = parse_windows(args.multiscale_windows)
    examples, summary = load_trial_examples(
        Path(args.feature_cache),
        train_subjects,
        val_subjects,
        train_label_ids,
        y_train_label_order,
        multiscale_windows,
        args.feature_norm,
        teacher_lookup,
    )
    train_examples = [example for example in examples if example["subject"] in train_subjects]
    val_examples = [example for example in examples if example["subject"] in val_subjects]
    train_loader = DataLoader(
        TrialDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trials,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        TrialDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trials,
        num_workers=args.num_workers,
    )

    model = GraphMambaResidualRegressor(
        eeg_nodes=int(summary["eeg_nodes"]),
        eeg_features=int(summary["eeg_feature_dim"]),
        fnirs_nodes=int(summary["fnirs_nodes"]),
        fnirs_features=int(summary["fnirs_feature_dim"]),
        graph_hidden=args.graph_hidden,
        d_model=args.d_model,
        cheb_order=args.cheb_order,
        mamba_layers=args.mamba_layers,
        temporal_block=args.temporal_block,
        d_state=args.d_state,
        dropout=args.dropout,
        graph_encoder=args.graph_encoder,
        fusion_mode=args.fusion_mode,
        eeg_scale_count=len(multiscale_windows) if args.scale_fusion == "gated" else 1,
        fnirs_scale_count=len(multiscale_windows) if args.scale_fusion == "gated" else 1,
    ).to(device)
    pretrained_info = None
    if args.pretrained_graph:
        pretrained_info = load_pretrained_graph(model, Path(args.pretrained_graph), device)
    teacher_projector = None
    parameters: list[nn.Parameter] = list(model.parameters())
    if teacher_dim and args.distill_weight > 0.0:
        teacher_projector = SequenceTeacherProjector(
            d_model=args.d_model,
            teacher_dim=teacher_dim,
            hidden_dim=args.teacher_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        parameters += list(teacher_projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction="none")

    best: dict[str, object] | None = None
    best_payload: dict[str, object] | None = None
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            target_mode=args.target_mode,
            input_modality=args.input_modality,
            modality_dropout=args.modality_dropout,
            residual_target_scale=args.residual_target_scale,
            teacher_projector=teacher_projector,
            distill_weight=args.distill_weight,
        )
        stats, payload = evaluate(model, val_loader, device, args.target_mode, args.input_modality)
        stats["epoch"] = epoch
        stats["train_loss"] = round(train_loss, 6)
        if best is None or float(stats["overall_mae"]) < float(best["overall_mae"]):
            best = stats
            best_payload = payload
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "teacher_projector_state": (
                        teacher_projector.state_dict() if teacher_projector is not None else None
                    ),
                    "args": vars(args),
                    "feature_summary": summary,
                    "best": best,
                },
                checkpoint,
            )
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_mae={stats['overall_mae']:.4f} "
                f"valence={stats['valence_mae']:.4f} arousal={stats['arousal_mae']:.4f}",
                flush=True,
            )

    assert best is not None and best_payload is not None
    smooth_results = make_smoothing_results(best_payload)
    output = {
        "model": "GraphMambaResidualRegressor",
        "target_mode": args.target_mode,
        "input_modality": args.input_modality,
        "fusion_mode": args.fusion_mode,
        "modality_dropout": args.modality_dropout,
        "teacher_cache": args.teacher_cache,
        "teacher_keys": list(teacher_keys),
        "teacher_dim": teacher_dim,
        "distill_weight": args.distill_weight,
        "pretrained_graph": args.pretrained_graph,
        "pretrained_info": pretrained_info,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_trials": len(train_examples),
            "val_trials": len(val_examples),
            "train_samples": int(sum(len(example["sample_ids"]) for example in train_examples)),
            "val_samples": int(sum(len(example["sample_ids"]) for example in val_examples)),
        },
        "feature_summary": summary,
        "best": best,
        "smoothing_results": smooth_results,
        "checkpoint": str(checkpoint),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def load_trial_examples(
    feature_cache: Path,
    train_subjects: list[str],
    val_subjects: list[str],
    train_label_ids: list[str],
    y_train_label_order: np.ndarray,
    multiscale_windows: tuple[int, ...],
    feature_norm: str,
    teacher_lookup: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with np.load(feature_cache, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y_raw = data["y_raw"].astype(np.float32)
        sample_subjects = data["sample_subjects"].astype(str)
        sample_ids = data["sample_ids"].astype(str)
        summary = json.loads(str(data["summary"].item()))

    eeg_shape = tuple(int(value) for value in summary.get("eeg_shape", (x.shape[0], 64, 5)))
    fnirs_shape = tuple(int(value) for value in summary.get("fnirs_shape", (x.shape[0], 51, 9)))
    if len(eeg_shape) != 3 or len(fnirs_shape) != 3:
        raise ValueError(f"Invalid cached feature shapes: eeg={eeg_shape}, fnirs={fnirs_shape}")
    eeg_nodes, eeg_features = eeg_shape[1], eeg_shape[2]
    fnirs_nodes, fnirs_features = fnirs_shape[1], fnirs_shape[2]
    eeg_dim = eeg_nodes * eeg_features
    fnirs_dim = fnirs_nodes * fnirs_features
    if x.shape[1] != eeg_dim + fnirs_dim:
        raise ValueError(
            f"Feature cache dimension mismatch: x has {x.shape[1]}, "
            f"expected {eeg_dim + fnirs_dim} from summary"
        )
    eeg = x[:, :eeg_dim].reshape(-1, eeg_nodes, eeg_features)
    fnirs = x[:, eeg_dim : eeg_dim + fnirs_dim].reshape(-1, fnirs_nodes, fnirs_features)
    relevant = set(train_subjects + val_subjects)
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids.astype(str)):
        subject, video, timestamp = parse_sample_id(sample_id)
        if subject in relevant:
            grouped[(subject, video)].append((timestamp, index))

    examples: list[dict[str, object]] = []
    for (subject, video), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        items = sorted(items)
        indices = np.asarray([index for _, index in items], dtype=np.int64)
        ids = sample_ids[indices].astype(str).tolist()
        prior = predict_video_time_mean(train_label_ids, y_train_label_order, ids)
        eeg_trial = make_multiscale_features(eeg[indices], multiscale_windows)
        fnirs_trial = make_multiscale_features(fnirs[indices], multiscale_windows)
        example = {
            "subject": subject,
            "video": video,
            "sample_ids": ids,
            "eeg": eeg_trial,
            "fnirs": fnirs_trial,
            "y_raw": y_raw[indices],
            "prior": prior.astype(np.float32),
            "residual": ((y_raw[indices] - prior) / 127.0).astype(np.float32),
        }
        if teacher_lookup is not None:
            missing = [sample_id for sample_id in ids if sample_id not in teacher_lookup]
            if missing:
                preview = ", ".join(missing[:3])
                raise ValueError(f"Teacher cache is missing {len(missing)} ids, e.g. {preview}")
            example["teacher"] = np.stack([teacher_lookup[sample_id] for sample_id in ids]).astype(
                np.float32
            )
        examples.append(example)
    apply_feature_norm(examples, feature_norm)
    summary = dict(summary)
    summary["multiscale_windows"] = list(multiscale_windows)
    summary["feature_norm"] = feature_norm
    summary["eeg_nodes"] = int(examples[0]["eeg"].shape[1]) if examples else 64
    summary["fnirs_nodes"] = int(examples[0]["fnirs"].shape[1]) if examples else 51
    summary["eeg_feature_dim"] = int(examples[0]["eeg"].shape[-1]) if examples else 5
    summary["fnirs_feature_dim"] = int(examples[0]["fnirs"].shape[-1]) if examples else 9
    summary["teacher_dim"] = int(examples[0].get("teacher", np.zeros((1, 0))).shape[-1]) if examples else 0
    return examples, summary


def load_teacher_lookup(path: Path, keys: tuple[str, ...]) -> tuple[dict[str, np.ndarray], int]:
    if not keys:
        raise ValueError("--teacher-keys must contain at least one key when --teacher-cache is set")
    with np.load(path, allow_pickle=False) as data:
        if "sample_ids" not in data:
            raise ValueError(f"{path} must contain sample_ids")
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        arrays = []
        for key in keys:
            if key not in data:
                raise ValueError(f"{path} does not contain teacher key '{key}'")
            value = data[key].astype(np.float32)
            if value.shape[0] != sample_ids.shape[0] or value.ndim != 2:
                raise ValueError(
                    f"Teacher key '{key}' must have shape [N, D], got {value.shape}"
                )
            arrays.append(value)
    merged = np.concatenate(arrays, axis=1).astype(np.float32)
    lookup = {sample_id: merged[index] for index, sample_id in enumerate(sample_ids)}
    return lookup, int(merged.shape[1])


def load_pretrained_graph(
    model: GraphMambaResidualRegressor,
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    info: dict[str, object] = {"path": str(path), "loaded": []}
    for key, module_name in (("eeg_encoder", "eeg_encoder"), ("fnirs_encoder", "fnirs_encoder")):
        state = checkpoint.get(key)
        if state is None:
            continue
        module = getattr(model, module_name)
        try:
            module.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            info[f"{key}_error"] = str(exc).splitlines()[0]
        else:
            info["loaded"].append(key)
    for key, parameter_name in (
        ("eeg_scale_logits", "eeg_scale_logits"),
        ("fnirs_scale_logits", "fnirs_scale_logits"),
    ):
        value = checkpoint.get(key)
        parameter = getattr(model, parameter_name)
        if value is None or parameter is None:
            continue
        if tuple(value.shape) == tuple(parameter.shape):
            with torch.no_grad():
                parameter.copy_(value.to(device=device, dtype=parameter.dtype))
            info["loaded"].append(key)
        else:
            info[f"{key}_error"] = f"shape mismatch {tuple(value.shape)} vs {tuple(parameter.shape)}"
    print(f"Loaded pretrained graph: {json.dumps(info, ensure_ascii=False)}", flush=True)
    return info


class TrialDataset(Dataset):
    def __init__(self, examples: list[dict[str, object]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


class SequenceTeacherProjector(nn.Module):
    def __init__(self, d_model: int, teacher_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def collate_trials(batch: list[dict[str, object]]) -> dict[str, object]:
    max_len = max(len(item["sample_ids"]) for item in batch)
    eeg_nodes = int(batch[0]["eeg"].shape[1])
    fnirs_nodes = int(batch[0]["fnirs"].shape[1])
    eeg_features = int(batch[0]["eeg"].shape[-1])
    fnirs_features = int(batch[0]["fnirs"].shape[-1])
    has_teacher = "teacher" in batch[0]
    teacher_dim = int(batch[0]["teacher"].shape[-1]) if has_teacher else 0
    out = {
        "sample_ids": [],
        "eeg": torch.zeros(len(batch), max_len, eeg_nodes, eeg_features, dtype=torch.float32),
        "fnirs": torch.zeros(len(batch), max_len, fnirs_nodes, fnirs_features, dtype=torch.float32),
        "y_raw": torch.zeros(len(batch), max_len, 2, dtype=torch.float32),
        "prior": torch.zeros(len(batch), max_len, 2, dtype=torch.float32),
        "residual": torch.zeros(len(batch), max_len, 2, dtype=torch.float32),
        "mask": torch.zeros(len(batch), max_len, dtype=torch.float32),
    }
    if has_teacher:
        out["teacher"] = torch.zeros(len(batch), max_len, teacher_dim, dtype=torch.float32)
    for row, item in enumerate(batch):
        length = len(item["sample_ids"])
        out["sample_ids"].append(item["sample_ids"])
        out["eeg"][row, :length] = torch.from_numpy(item["eeg"])
        out["fnirs"][row, :length] = torch.from_numpy(item["fnirs"])
        out["y_raw"][row, :length] = torch.from_numpy(item["y_raw"])
        out["prior"][row, :length] = torch.from_numpy(item["prior"])
        out["residual"][row, :length] = torch.from_numpy(item["residual"])
        if has_teacher:
            out["teacher"][row, :length] = torch.from_numpy(item["teacher"])
        out["mask"][row, :length] = 1.0
    return out


def train_epoch(
    model: GraphMambaResidualRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    target_mode: str,
    input_modality: str,
    modality_dropout: float,
    residual_target_scale: float,
    teacher_projector: SequenceTeacherProjector | None,
    distill_weight: float,
) -> float:
    model.train()
    if teacher_projector is not None:
        teacher_projector.train()
    total = 0.0
    loss_sum = 0.0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        fnirs = batch["fnirs"].to(device)
        eeg, fnirs = apply_input_modality(eeg, fnirs, input_modality)
        eeg, fnirs = apply_modality_dropout(eeg, fnirs, modality_dropout)
        target = batch["residual"].to(device) * residual_target_scale
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if teacher_projector is not None and distill_weight > 0.0 and "teacher" in batch:
            output, features = model(eeg, fnirs, mask=mask, return_features=True)
        else:
            output = model(eeg, fnirs, mask=mask)
            features = None
        loss_by_dim = loss_fn(output, target)
        dim_mask = make_dim_mask(target_mode, output.device).view(1, 1, 2)
        loss = (loss_by_dim * mask.unsqueeze(-1) * dim_mask).sum()
        denom = (mask.sum() * dim_mask.sum()).clamp_min(1.0)
        loss = loss / denom
        if teacher_projector is not None and features is not None and distill_weight > 0.0:
            teacher = batch["teacher"].to(device)
            teacher_pred = teacher_projector(features)
            loss = loss + distill_weight * masked_teacher_loss(teacher_pred, teacher, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if teacher_projector is not None:
            torch.nn.utils.clip_grad_norm_(teacher_projector.parameters(), 5.0)
        optimizer.step()
        weight = float(mask.sum())
        total += weight
        loss_sum += float(loss.detach()) * weight
    return loss_sum / max(total, 1.0)


def masked_teacher_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    student = F.normalize(student, dim=-1, eps=1e-6)
    teacher = F.normalize(teacher.detach(), dim=-1, eps=1e-6)
    loss = F.mse_loss(student, teacher, reduction="none").mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: GraphMambaResidualRegressor,
    loader: DataLoader,
    device: torch.device,
    target_mode: str,
    input_modality: str,
) -> tuple[dict[str, object], dict[str, object]]:
    model.eval()
    all_ids: list[str] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prior: list[np.ndarray] = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        fnirs = batch["fnirs"].to(device)
        eeg, fnirs = apply_input_modality(eeg, fnirs, input_modality)
        mask = batch["mask"].to(device)
        output = model(eeg, fnirs, mask=mask).cpu().numpy()
        y_raw = batch["y_raw"].numpy()
        prior = batch["prior"].numpy()
        mask_np = batch["mask"].numpy().astype(bool)
        pred = prior.copy()
        if target_mode in ("all", "valence"):
            pred[..., 0] = prior[..., 0] + output[..., 0] * 127.0
        if target_mode in ("all", "arousal"):
            pred[..., 1] = prior[..., 1] + output[..., 1] * 127.0
        pred = np.clip(pred, 1.0, 255.0)
        for row, ids in enumerate(batch["sample_ids"]):
            length = int(mask_np[row].sum())
            all_ids.extend(ids[:length])
            all_true.append(y_raw[row, :length])
            all_pred.append(pred[row, :length])
            all_prior.append(prior[row, :length])
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    stats = score(
        f"GraphMambaResidual_{target_mode}",
        y_true,
        y_pred,
        "Adaptive channel graph encoders plus Mamba-like SSM temporal residual model.",
    )
    prior_all = np.concatenate(all_prior, axis=0)
    return stats, {"sample_ids": all_ids, "y_true": y_true, "y_pred": y_pred, "prior": prior_all}


def make_smoothing_results(payload: dict[str, object]) -> list[dict[str, object]]:
    sample_ids = payload["sample_ids"]
    y_true = payload["y_true"]
    y_pred = payload["y_pred"]
    prior = payload["prior"]
    results = []
    for window in (3, 5, 9):
        smooth = smooth_predictions(sample_ids, y_pred, window=window)
        results.append(
            score(
                f"GraphMambaResidual_smooth{window}",
                y_true,
                smooth,
                f"Graph-Mamba residual with moving-average smoothing window={window}.",
            )
        )
    residual = y_pred - prior
    for scale in (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        4.0,
    ):
        scaled = np.clip(prior + scale * residual, 1.0, 255.0)
        results.append(
            score(
                f"GraphMambaResidual_scale{scale:.2f}",
                y_true,
                scaled,
                f"Scale residual by {scale:.2f} before optional smoothing.",
            )
        )
        for window in (3, 5, 9):
            smooth = smooth_predictions(sample_ids, scaled, window=window)
            results.append(
                score(
                    f"GraphMambaResidual_scale{scale:.2f}_smooth{window}",
                    y_true,
                    smooth,
                    f"Scale residual by {scale:.2f}, then moving-average smoothing window={window}.",
                )
            )
    for scale in (1.0, 1.5, 2.0, 2.5, 2.75, 3.0):
        for clip in (5.0, 10.0, 15.0, 20.0, 30.0, 40.0):
            clipped = np.clip(prior + np.clip(scale * residual, -clip, clip), 1.0, 255.0)
            results.append(
                score(
                    f"GraphMambaResidual_scale{scale:.2f}_clip{clip:.0f}",
                    y_true,
                    clipped,
                    f"Scale residual by {scale:.2f}, clip raw residual to +/-{clip:.0f}.",
                )
            )
            for window in (5, 9):
                smooth = smooth_predictions(sample_ids, clipped, window=window)
                results.append(
                    score(
                        f"GraphMambaResidual_scale{scale:.2f}_clip{clip:.0f}_smooth{window}",
                        y_true,
                        smooth,
                        f"Scale residual by {scale:.2f}, clip to +/-{clip:.0f}, then smooth window={window}.",
                    )
                )
    return results


def parse_windows(text: str) -> tuple[int, ...]:
    windows = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        window = int(stripped)
        if window < 1:
            raise ValueError("--multiscale-windows values must be positive")
        windows.append(window)
    if not windows:
        windows = [1]
    deduped = sorted(set(windows))
    return tuple(deduped)


def make_multiscale_features(features: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    if windows == (1,):
        return features.astype(np.float32, copy=False)
    scales = []
    for window in windows:
        if window == 1:
            scales.append(features)
        else:
            scales.append(temporal_moving_average(features, window))
    return np.concatenate(scales, axis=-1).astype(np.float32, copy=False)


def apply_feature_norm(examples: list[dict[str, object]], scope: str) -> None:
    if scope == "none":
        return
    if scope == "trial":
        for example in examples:
            example["eeg"] = normalize_feature_block(example["eeg"])
            example["fnirs"] = normalize_feature_block(example["fnirs"])
        return
    if scope == "subject":
        by_subject: dict[str, list[dict[str, object]]] = defaultdict(list)
        for example in examples:
            by_subject[str(example["subject"])].append(example)
        for subject_examples in by_subject.values():
            eeg_blocks = [example["eeg"].reshape(-1, example["eeg"].shape[-1]) for example in subject_examples]
            fnirs_blocks = [
                example["fnirs"].reshape(-1, example["fnirs"].shape[-1]) for example in subject_examples
            ]
            eeg_mean, eeg_std = feature_stats(np.concatenate(eeg_blocks, axis=0))
            fnirs_mean, fnirs_std = feature_stats(np.concatenate(fnirs_blocks, axis=0))
            for example in subject_examples:
                example["eeg"] = ((example["eeg"] - eeg_mean) / eeg_std).astype(np.float32)
                example["fnirs"] = ((example["fnirs"] - fnirs_mean) / fnirs_std).astype(np.float32)
        return
    raise ValueError(f"Unsupported feature norm scope: {scope}")


def normalize_feature_block(features: np.ndarray) -> np.ndarray:
    mean, std = feature_stats(features.reshape(-1, features.shape[-1]))
    return ((features - mean) / std).astype(np.float32)


def feature_stats(flat_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = flat_features.mean(axis=0, keepdims=True).reshape(1, 1, -1)
    std = np.maximum(flat_features.std(axis=0, keepdims=True), 1e-6).reshape(1, 1, -1)
    return mean.astype(np.float32), std.astype(np.float32)


def temporal_moving_average(features: np.ndarray, window: int) -> np.ndarray:
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(features, ((pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0, dtype=np.float64)
    cumsum = np.concatenate([np.zeros_like(cumsum[:1]), cumsum], axis=0)
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return smoothed.astype(np.float32)


def make_dim_mask(target_mode: str, device: torch.device) -> torch.Tensor:
    if target_mode == "all":
        return torch.tensor([1.0, 1.0], device=device)
    if target_mode == "valence":
        return torch.tensor([1.0, 0.0], device=device)
    if target_mode == "arousal":
        return torch.tensor([0.0, 1.0], device=device)
    raise ValueError(target_mode)


def apply_input_modality(
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    input_modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_modality == "both":
        return eeg, fnirs
    if input_modality == "eeg":
        return eeg, torch.zeros_like(fnirs)
    if input_modality == "fnirs":
        return torch.zeros_like(eeg), fnirs
    if input_modality == "none":
        return torch.zeros_like(eeg), torch.zeros_like(fnirs)
    raise ValueError(input_modality)


def apply_modality_dropout(
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probability <= 0.0:
        return eeg, fnirs
    probability = min(max(float(probability), 0.0), 1.0)
    batch = eeg.size(0)
    shape = (batch, 1, 1, 1)
    drop_eeg = torch.rand(shape, device=eeg.device) < probability
    drop_fnirs = torch.rand(shape, device=fnirs.device) < probability
    both_dropped = drop_eeg & drop_fnirs
    restore_eeg = both_dropped & (torch.rand(shape, device=eeg.device) < 0.5)
    restore_fnirs = both_dropped & ~restore_eeg
    drop_eeg = drop_eeg & ~restore_eeg
    drop_fnirs = drop_fnirs & ~restore_fnirs
    eeg = eeg * (~drop_eeg).to(dtype=eeg.dtype)
    fnirs = fnirs * (~drop_fnirs).to(dtype=fnirs.dtype)
    return eeg, fnirs


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    match = SAMPLE_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id: {sample_id}")
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
