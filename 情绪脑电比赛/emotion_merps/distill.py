from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TeacherCache:
    sample_ids: np.ndarray
    keys: tuple[str, ...]
    arrays: tuple[np.ndarray, ...]

    @property
    def dims(self) -> list[int]:
        return [int(array.shape[1]) for array in self.arrays]


def load_teacher_cache(
    path: str | Path,
    sample_ids: Sequence[str],
    keys: Sequence[str],
) -> TeacherCache:
    """Load an NPZ teacher cache and align it to training sample_ids.

    Expected NPZ format:
      sample_ids: string array shaped [N]
      <teacher_key>: float array shaped [N, D]

    Typical teacher keys are: emotion, eeg, fnirs.
    """
    data = np.load(path, allow_pickle=False)
    if "sample_ids" not in data:
        raise ValueError(f"{path} must contain a sample_ids array")

    cache_ids = np.asarray(data["sample_ids"]).astype(str)
    index = {sample_id: idx for idx, sample_id in enumerate(cache_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Teacher cache is missing {len(missing)} sample_ids, e.g. {preview}")

    aligned_indices = np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64)
    arrays: list[np.ndarray] = []
    for key in keys:
        if key not in data:
            raise ValueError(f"{path} does not contain teacher key '{key}'")
        array = np.asarray(data[key], dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != cache_ids.shape[0]:
            raise ValueError(
                f"Teacher key '{key}' must have shape [N, D], got {array.shape}"
            )
        arrays.append(np.ascontiguousarray(array[aligned_indices]))
    return TeacherCache(np.asarray(sample_ids).astype(str), tuple(keys), tuple(arrays))


class GatedTeacherFusion(nn.Module):
    """Fuse heterogeneous teacher embeddings with a learnable gate."""

    def __init__(
        self,
        teacher_dims: Sequence[int],
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not teacher_dims:
            raise ValueError("At least one teacher dimension is required.")
        self.teacher_dims = tuple(int(dim) for dim in teacher_dims)
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, fusion_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dim in self.teacher_dims
            ]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * len(self.teacher_dims)),
            nn.Linear(fusion_dim * len(self.teacher_dims), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.teacher_dims)),
        )

    def forward(self, teachers: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(teachers) != len(self.projectors):
            raise ValueError(f"Expected {len(self.projectors)} teachers, got {len(teachers)}")
        projected = [projector(tensor) for projector, tensor in zip(self.projectors, teachers)]
        gate_logits = self.gate(torch.cat(projected, dim=1))
        weights = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(projected, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class StudentTeacherDistiller(nn.Module):
    """Project student embeddings and align them to fused teacher embeddings."""

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        student_projected = self.projector(student)
        cosine = 1.0 - F.cosine_similarity(student_projected, teacher.detach(), dim=1).mean()
        mse = F.mse_loss(
            F.normalize(student_projected, dim=1),
            F.normalize(teacher.detach(), dim=1),
        )
        return cosine + mse


def gate_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Small diagnostic regularizer; higher entropy means less teacher collapse."""
    weights = weights.clamp_min(1e-8)
    return -(weights * weights.log()).sum(dim=1).mean()
