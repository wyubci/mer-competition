from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def normalize_adjacency(adjacency: torch.Tensor, symmetric: bool = True) -> torch.Tensor:
    adjacency = F.relu(adjacency)
    if symmetric:
        adjacency = adjacency + adjacency.transpose(0, 1)
    adjacency = adjacency + torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def chebyshev_supports(adjacency: torch.Tensor, order: int) -> list[torch.Tensor]:
    if order < 1:
        raise ValueError("Chebyshev order must be >= 1")
    supports = [torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)]
    if order == 1:
        return supports
    supports.append(adjacency)
    for _ in range(2, order):
        supports.append(torch.matmul(supports[-1], adjacency))
    return supports


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("nm,bmf->bnf", support, x)
        return torch.matmul(propagated, self.weight) + self.bias


class AdaptiveChebGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        cheb_order: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.cheb_order = cheb_order
        self.input_norm = nn.BatchNorm1d(in_features)
        self.adjacency = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.convs = nn.ModuleList(
            [GraphConvolution(in_features, hidden_dim) for _ in range(cheb_order)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        adjacency = normalize_adjacency(self.adjacency)
        supports = chebyshev_supports(adjacency, self.cheb_order)
        out = sum(conv(x, support) for conv, support in zip(self.convs, supports))
        out = F.gelu(out)
        out = self.output_norm(out)
        return self.dropout(out)


class CrossModalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.2, ff_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query=query, key=context, value=context, need_weights=False)
        x = self.norm1(query + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg_embedding: torch.Tensor, fnirs_embedding: torch.Tensor) -> torch.Tensor:
        batch_size = eeg_embedding.size(0)
        if batch_size < 2:
            return eeg_embedding.new_zeros(())
        z_eeg = F.normalize(eeg_embedding, dim=1)
        z_fnirs = F.normalize(fnirs_embedding, dim=1)
        logits = torch.matmul(z_eeg, z_fnirs.t()) / self.temperature
        labels = torch.arange(batch_size, device=eeg_embedding.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class ASACRegressor(nn.Module):
    """ASAC-style EEG-fNIRS fusion network adapted for continuous MER_PS labels."""

    def __init__(
        self,
        eeg_nodes: int,
        eeg_features: int,
        fnirs_nodes: int,
        fnirs_features: int,
        output_dim: int = 2,
        hidden_dim: int = 64,
        cheb_order: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        projection_dim: int = 64,
        temperature: float = 0.5,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

        self.eeg_encoder = AdaptiveChebGraphEncoder(
            eeg_nodes, eeg_features, hidden_dim, cheb_order, dropout
        )
        self.fnirs_encoder = AdaptiveChebGraphEncoder(
            fnirs_nodes, fnirs_features, hidden_dim, cheb_order, dropout
        )
        self.eeg_to_fnirs = CrossModalBlock(hidden_dim, heads, dropout)
        self.fnirs_to_eeg = CrossModalBlock(hidden_dim, heads, dropout)
        self.eeg_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.fnirs_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.alignment_loss = ContrastiveAlignmentLoss(temperature)
        pooled_dim = hidden_dim * 8
        self.regressor = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eeg_graph = self.eeg_encoder(eeg)
        fnirs_graph = self.fnirs_encoder(fnirs)

        eeg_global = eeg_graph.mean(dim=1)
        fnirs_global = fnirs_graph.mean(dim=1)
        contrastive_loss = self.alignment_loss(
            self.eeg_projector(eeg_global),
            self.fnirs_projector(fnirs_global),
        )

        eeg_cross = self.eeg_to_fnirs(eeg_graph, fnirs_graph)
        fnirs_cross = self.fnirs_to_eeg(fnirs_graph, eeg_graph)
        pooled = torch.cat(
            [
                eeg_graph.mean(dim=1),
                eeg_graph.amax(dim=1),
                fnirs_graph.mean(dim=1),
                fnirs_graph.amax(dim=1),
                eeg_cross.mean(dim=1),
                eeg_cross.amax(dim=1),
                fnirs_cross.mean(dim=1),
                fnirs_cross.amax(dim=1),
            ],
            dim=1,
        )
        prediction = torch.sigmoid(self.regressor(pooled))
        return prediction, contrastive_loss
