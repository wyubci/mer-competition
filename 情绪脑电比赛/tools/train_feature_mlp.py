from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_iteration_experiments import (  # noqa: E402
    expand_subjects,
    load_labels,
    predict_video_time_mean,
    score,
    smooth_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP models on cached MER-PS signal features.")
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/results/iteration_001_mlp.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_subjects = expand_subjects(args.train_subjects)
    val_subjects = expand_subjects(args.val_subjects)
    data_root = Path(args.data_root)

    labels = load_labels(data_root, train_subjects + val_subjects)
    train_ids_label_order = [
        sample_id for sample_id in labels if sample_id.split("_V", 1)[0] in train_subjects
    ]
    y_train_label_order = np.stack([labels[sample_id] for sample_id in train_ids_label_order]).astype(
        np.float32
    )

    with np.load(args.feature_cache, allow_pickle=False) as data:
        x = data["x"].astype(np.float32)
        y_raw = data["y_raw"].astype(np.float32)
        sample_subjects = data["sample_subjects"].astype(str)
        sample_ids = data["sample_ids"].astype(str)
        summary = json.loads(str(data["summary"].item()))

    train_idx = np.flatnonzero(np.isin(sample_subjects, train_subjects))
    val_idx = np.flatnonzero(np.isin(sample_subjects, val_subjects))
    train_ids = sample_ids[train_idx].astype(str).tolist()
    val_ids = sample_ids[val_idx].astype(str).tolist()
    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y_raw[train_idx]
    y_val = y_raw[val_idx]
    train_prior = predict_video_time_mean(train_ids_label_order, y_train_label_order, train_ids)
    val_prior = predict_video_time_mean(train_ids_label_order, y_train_label_order, val_ids)

    results = []
    raw_result = train_one(
        name="MLPFeatureRaw",
        x_train=x_train,
        y_train=(y_train - 1.0) / 254.0,
        x_val=x_val,
        y_val=y_val,
        val_ids=val_ids,
        args=args,
        mode="raw",
        val_prior=None,
    )
    results.extend(raw_result)

    residual_result = train_one(
        name="MLPFeatureResidual",
        x_train=x_train,
        y_train=(y_train - train_prior) / 127.0,
        x_val=x_val,
        y_val=y_val,
        val_ids=val_ids,
        args=args,
        mode="residual",
        val_prior=val_prior,
    )
    results.extend(residual_result)
    results = sorted(results, key=lambda item: float(item["overall_mae"]))

    output = {
        "feature_cache": args.feature_cache,
        "feature_summary": summary,
        "split": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "train_samples": int(train_idx.size),
            "val_samples": int(val_idx.size),
        },
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def train_one(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_ids: list[str],
    args: argparse.Namespace,
    mode: str,
    val_prior: np.ndarray | None,
) -> list[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best: dict[str, object] | None = None
    best_pred: np.ndarray | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(xb.size(0))
            count += int(xb.size(0))

        pred = predict(model, x_val, device, mode, val_prior)
        stats = score(name, y_val, pred, f"{mode} MLP on cached EEG/fNIRS ASAC features.")
        stats["epoch"] = epoch
        stats["train_loss"] = round(loss_sum / max(count, 1), 6)
        if best is None or float(stats["overall_mae"]) < float(best["overall_mae"]):
            best = stats
            best_pred = pred
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"{name} epoch={epoch} train_loss={stats['train_loss']} "
                f"val_mae={stats['overall_mae']}",
                flush=True,
            )

    assert best is not None and best_pred is not None
    results = [best]
    for window in (3, 5, 9):
        smooth = smooth_predictions(val_ids, best_pred, window=window)
        smooth_stats = score(
            f"{name}_best_smooth{window}",
            y_val,
            smooth,
            f"Best {name} prediction with moving-average smoothing window={window}.",
        )
        smooth_stats["source_epoch"] = best["epoch"]
        results.append(smooth_stats)
        if mode == "residual" and val_prior is not None:
            valence_only = val_prior.copy()
            valence_only[:, 0] = smooth[:, 0]
            valence_stats = score(
                f"{name}_valence_only_smooth{window}",
                y_val,
                valence_only,
                f"Use smoothed {name} only for valence; keep VideoTimeMean arousal.",
            )
            valence_stats["source_epoch"] = best["epoch"]
            results.append(valence_stats)

            arousal_only = val_prior.copy()
            arousal_only[:, 1] = smooth[:, 1]
            arousal_stats = score(
                f"{name}_arousal_only_smooth{window}",
                y_val,
                arousal_only,
                f"Use smoothed {name} only for arousal; keep VideoTimeMean valence.",
            )
            arousal_stats["source_epoch"] = best["epoch"]
            results.append(arousal_stats)
    return results


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict(
    model: FeatureMLP,
    x: np.ndarray,
    device: torch.device,
    mode: str,
    val_prior: np.ndarray | None,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, x.shape[0], 4096):
        xb = torch.from_numpy(x[start : start + 4096]).to(device)
        outputs.append(model(xb).cpu().numpy())
    output = np.concatenate(outputs, axis=0)
    if mode == "raw":
        pred = output * 254.0 + 1.0
    elif mode == "residual":
        if val_prior is None:
            raise ValueError("val_prior is required for residual mode")
        pred = val_prior + output * 127.0
    else:
        raise ValueError(mode)
    return np.clip(pred, 1.0, 255.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
