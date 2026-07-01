from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emotion_merps.graph_mamba import GraphMambaResidualRegressor  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels  # noqa: E402
from tools.train_graph_mamba import (  # noqa: E402
    TrialDataset,
    collate_trials,
    load_trial_examples,
    parse_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked EEG/fNIRS graph pretraining for MER-PS."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--feature-cache", default="experiments/features/asac_features_20_4.npz")
    parser.add_argument("--train-subjects", default="test_1-test_20")
    parser.add_argument("--val-subjects", default="test_21-test_24")
    parser.add_argument("--output", default="experiments/checkpoints/pretrain/masked_graph.pt")
    parser.add_argument("--summary-output", default="experiments/results/iteration_031_masked_graph_pretrain.json")
    parser.add_argument("--epochs", type=int, default=5)
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
        default="hybrid_functional",
    )
    parser.add_argument("--cheb-order", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
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
        default="gated_ssm",
    )
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--multiscale-windows", default="1,5,9")
    parser.add_argument("--scale-fusion", choices=("concat", "gated"), default="gated")
    parser.add_argument("--feature-norm", choices=("none", "trial", "subject"), default="none")
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class MaskedGraphPretrainer(nn.Module):
    def __init__(
        self,
        eeg_nodes: int,
        eeg_features: int,
        fnirs_nodes: int,
        fnirs_features: int,
        graph_hidden: int,
        d_model: int,
        cheb_order: int,
        mamba_layers: int,
        temporal_block: str,
        d_state: int,
        dropout: float,
        graph_encoder: str,
        eeg_scale_count: int,
        fnirs_scale_count: int,
    ):
        super().__init__()
        self.backbone = GraphMambaResidualRegressor(
            eeg_nodes=eeg_nodes,
            eeg_features=eeg_features,
            fnirs_nodes=fnirs_nodes,
            fnirs_features=fnirs_features,
            graph_hidden=graph_hidden,
            d_model=d_model,
            cheb_order=cheb_order,
            mamba_layers=mamba_layers,
            temporal_block=temporal_block,
            d_state=d_state,
            dropout=dropout,
            graph_encoder=graph_encoder,
            fusion_mode="pool",
            eeg_scale_count=eeg_scale_count,
            fnirs_scale_count=fnirs_scale_count,
        )
        self.eeg_decoder = nn.Sequential(
            nn.LayerNorm(graph_hidden),
            nn.Linear(graph_hidden, graph_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, self.backbone.eeg_base_features),
        )
        self.fnirs_decoder = nn.Sequential(
            nn.LayerNorm(graph_hidden),
            nn.Linear(graph_hidden, graph_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, self.backbone.fnirs_base_features),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        eeg_mask = make_node_mask(eeg.size(0), eeg.size(1), mask_ratio, eeg.device)
        fnirs_mask = make_node_mask(fnirs.size(0), fnirs.size(1), mask_ratio, fnirs.device)
        eeg_input = eeg.masked_fill(eeg_mask.unsqueeze(-1), 0.0)
        fnirs_input = fnirs.masked_fill(fnirs_mask.unsqueeze(-1), 0.0)

        with torch.no_grad():
            eeg_target = self._fuse_eeg(eeg)
            fnirs_target = self._fuse_fnirs(fnirs)
        eeg_encoded = self.backbone.eeg_encoder(self._fuse_eeg(eeg_input))
        fnirs_encoded = self.backbone.fnirs_encoder(self._fuse_fnirs(fnirs_input))
        eeg_loss = masked_node_mse(self.eeg_decoder(eeg_encoded), eeg_target, eeg_mask)
        fnirs_loss = masked_node_mse(self.fnirs_decoder(fnirs_encoded), fnirs_target, fnirs_mask)
        loss = eeg_loss + fnirs_loss
        return loss, {
            "eeg_loss": float(eeg_loss.detach()),
            "fnirs_loss": float(fnirs_loss.detach()),
        }

    def _fuse_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.backbone._fuse_scales(
            eeg,
            self.backbone.eeg_scale_count,
            self.backbone.eeg_base_features,
            self.backbone.eeg_scale_logits,
        )

    def _fuse_fnirs(self, fnirs: torch.Tensor) -> torch.Tensor:
        return self.backbone._fuse_scales(
            fnirs,
            self.backbone.fnirs_scale_count,
            self.backbone.fnirs_base_features,
            self.backbone.fnirs_scale_logits,
        )


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
    windows = parse_windows(args.multiscale_windows)
    examples, summary = load_trial_examples(
        Path(args.feature_cache),
        train_subjects,
        val_subjects,
        train_label_ids,
        y_train_label_order,
        windows,
        args.feature_norm,
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

    scale_count = len(windows) if args.scale_fusion == "gated" else 1
    model = MaskedGraphPretrainer(
        eeg_nodes=int(summary.get("eeg_nodes", 64)),
        eeg_features=int(summary["eeg_feature_dim"]),
        fnirs_nodes=int(summary.get("fnirs_nodes", 51)),
        fnirs_features=int(summary["fnirs_feature_dim"]),
        graph_hidden=args.graph_hidden,
        d_model=args.d_model,
        cheb_order=args.cheb_order,
        mamba_layers=args.mamba_layers,
        temporal_block=args.temporal_block,
        d_state=args.d_state,
        dropout=args.dropout,
        graph_encoder=args.graph_encoder,
        eeg_scale_count=scale_count,
        fnirs_scale_count=scale_count,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, device, args.mask_ratio, optimizer)
        val_stats = run_epoch(model, val_loader, device, args.mask_ratio, None)
        row = {
            "epoch": epoch,
            "train_loss": round(train_stats["loss"], 6),
            "train_eeg_loss": round(train_stats["eeg_loss"], 6),
            "train_fnirs_loss": round(train_stats["fnirs_loss"], 6),
            "val_loss": round(val_stats["loss"], 6),
            "val_eeg_loss": round(val_stats["eeg_loss"], 6),
            "val_fnirs_loss": round(val_stats["fnirs_loss"], 6),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save(
                {
                    "args": vars(args),
                    "feature_summary": summary,
                    "eeg_encoder": model.backbone.eeg_encoder.state_dict(),
                    "fnirs_encoder": model.backbone.fnirs_encoder.state_dict(),
                    "eeg_scale_logits": (
                        model.backbone.eeg_scale_logits.detach().cpu()
                        if model.backbone.eeg_scale_logits is not None
                        else None
                    ),
                    "fnirs_scale_logits": (
                        model.backbone.fnirs_scale_logits.detach().cpu()
                        if model.backbone.fnirs_scale_logits is not None
                        else None
                    ),
                    "best": row,
                },
                output_path,
            )

    summary_output = {
        "method": "masked_graph_pretraining",
        "args": vars(args),
        "feature_summary": summary,
        "best": min(history, key=lambda item: item["val_loss"]),
        "history": history,
        "checkpoint": str(output_path),
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_output, ensure_ascii=False, indent=2), flush=True)


def run_epoch(
    model: MaskedGraphPretrainer,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "eeg_loss": 0.0, "fnirs_loss": 0.0}
    count = 0
    with torch.enable_grad() if training else torch.no_grad():
        for batch in loader:
            eeg, fnirs = flatten_valid(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, parts = model(eeg, fnirs, mask_ratio)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = int(eeg.size(0))
            count += batch_size
            totals["loss"] += float(loss.detach()) * batch_size
            totals["eeg_loss"] += parts["eeg_loss"] * batch_size
            totals["fnirs_loss"] += parts["fnirs_loss"] * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def flatten_valid(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch["mask"].bool()
    eeg = batch["eeg"][mask].to(device)
    fnirs = batch["fnirs"][mask].to(device)
    return eeg, fnirs


def make_node_mask(batch: int, nodes: int, ratio: float, device: torch.device) -> torch.Tensor:
    mask = torch.rand(batch, nodes, device=device) < ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        random_nodes = torch.randint(0, nodes, (int(empty.sum()),), device=device)
        mask[empty, random_nodes] = True
    return mask


def masked_node_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(prediction, target.detach(), reduction="none").mean(dim=-1)
    return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
