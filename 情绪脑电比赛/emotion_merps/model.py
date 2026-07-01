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


def normalize_batched_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adjacency = F.relu(adjacency)
    eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency + eye.unsqueeze(0)
    degree = adjacency.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)


def sparsify_batched_adjacency(adjacency: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k >= adjacency.size(-1):
        return adjacency
    values, indices = torch.topk(adjacency, k=max(1, top_k), dim=-1)
    sparse = torch.zeros_like(adjacency).scatter(-1, indices, values)
    return torch.maximum(sparse, sparse.transpose(1, 2))


def functional_adjacency(x: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    normalized = F.normalize(x, dim=-1, eps=1e-6)
    adjacency = torch.bmm(normalized, normalized.transpose(1, 2))
    if top_k is not None:
        adjacency = sparsify_batched_adjacency(adjacency, top_k)
    return normalize_batched_adjacency(adjacency)


def batched_chebyshev_supports(adjacency: torch.Tensor, order: int) -> list[torch.Tensor]:
    if order < 1:
        raise ValueError("Chebyshev order must be >= 1")
    batch, nodes = adjacency.shape[:2]
    eye = torch.eye(nodes, device=adjacency.device, dtype=adjacency.dtype)
    supports = [eye.unsqueeze(0).expand(batch, -1, -1)]
    if order == 1:
        return supports
    supports.append(adjacency)
    for _ in range(2, order):
        supports.append(torch.bmm(supports[-1], adjacency))
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


class BatchedGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        propagated = torch.bmm(support, x)
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


class FunctionalChebGraphEncoder(nn.Module):
    """Chebyshev graph encoder with per-sample functional-connectivity adjacency."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        cheb_order: int = 3,
        dropout: float = 0.2,
        top_k: int | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.cheb_order = cheb_order
        self.top_k = top_k
        self.input_norm = nn.BatchNorm1d(in_features)
        self.convs = nn.ModuleList(
            [BatchedGraphConvolution(in_features, hidden_dim) for _ in range(cheb_order)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        adjacency = functional_adjacency(x, self.top_k)
        supports = batched_chebyshev_supports(adjacency, self.cheb_order)
        out = sum(conv(x, support) for conv, support in zip(self.convs, supports))
        out = F.gelu(out)
        out = self.output_norm(out)
        return self.dropout(out)


class HybridFunctionalChebGraphEncoder(nn.Module):
    """Mixes learned static topology with per-sample functional connectivity."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        cheb_order: int = 3,
        dropout: float = 0.2,
        top_k: int | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.cheb_order = cheb_order
        self.top_k = top_k
        self.input_norm = nn.BatchNorm1d(in_features)
        self.adjacency = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.functional_logit = nn.Parameter(torch.zeros(()))
        self.convs = nn.ModuleList(
            [BatchedGraphConvolution(in_features, hidden_dim) for _ in range(cheb_order)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        dynamic = functional_adjacency(x, self.top_k)
        static = normalize_adjacency(self.adjacency).unsqueeze(0).expand(x.size(0), -1, -1)
        dynamic_weight = torch.sigmoid(self.functional_logit)
        adjacency = normalize_batched_adjacency(
            dynamic_weight * dynamic + (1.0 - dynamic_weight) * static
        )
        supports = batched_chebyshev_supports(adjacency, self.cheb_order)
        out = sum(conv(x, support) for conv, support in zip(self.convs, supports))
        out = F.gelu(out)
        out = self.output_norm(out)
        return self.dropout(out)


class SparseFunctionalChebGraphEncoder(FunctionalChebGraphEncoder):
    def __init__(self, *args: object, top_k: int = 8, **kwargs: object):
        super().__init__(*args, top_k=top_k, **kwargs)


class SparseHybridFunctionalChebGraphEncoder(HybridFunctionalChebGraphEncoder):
    def __init__(self, *args: object, top_k: int = 8, **kwargs: object):
        super().__init__(*args, top_k=top_k, **kwargs)


class SignedAdaptiveChebGraphEncoder(nn.Module):
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
        self.positive_adjacency = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.negative_adjacency = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.positive_convs = nn.ModuleList(
            [GraphConvolution(in_features, hidden_dim) for _ in range(cheb_order)]
        )
        self.negative_convs = nn.ModuleList(
            [GraphConvolution(in_features, hidden_dim) for _ in range(cheb_order)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.positive_adjacency)
        nn.init.xavier_uniform_(self.negative_adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        positive = normalize_adjacency(self.positive_adjacency)
        negative = normalize_adjacency(self.negative_adjacency)
        positive_supports = chebyshev_supports(positive, self.cheb_order)
        negative_supports = chebyshev_supports(negative, self.cheb_order)
        out = sum(
            conv(x, support)
            for conv, support in zip(self.positive_convs, positive_supports)
        )
        out = out - sum(
            conv(x, support)
            for conv, support in zip(self.negative_convs, negative_supports)
        )
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
        self.embedding_dim = pooled_dim
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

    def encode(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg_graph = self.eeg_encoder(eeg)
        fnirs_graph = self.fnirs_encoder(fnirs)

        eeg_global = eeg_graph.mean(dim=1)
        fnirs_global = fnirs_graph.mean(dim=1)
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
        return {
            "eeg_graph": eeg_graph,
            "fnirs_graph": fnirs_graph,
            "eeg_global": eeg_global,
            "fnirs_global": fnirs_global,
            "eeg_projected": self.eeg_projector(eeg_global),
            "fnirs_projected": self.fnirs_projector(fnirs_global),
            "eeg_cross": eeg_cross,
            "fnirs_cross": fnirs_cross,
            "pooled": pooled,
        }

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        return_embeddings: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.encode(eeg, fnirs)
        contrastive_loss = self.alignment_loss(
            embeddings["eeg_projected"],
            embeddings["fnirs_projected"],
        )
        pooled = embeddings["pooled"]
        prediction = torch.sigmoid(self.regressor(pooled))
        if return_embeddings:
            return prediction, contrastive_loss, embeddings
        return prediction, contrastive_loss
