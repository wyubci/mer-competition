from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from emotion_merps.model import (
    AdaptiveChebGraphEncoder,
    CrossModalBlock,
    FunctionalChebGraphEncoder,
    HybridFunctionalChebGraphEncoder,
    SignedAdaptiveChebGraphEncoder,
    SparseFunctionalChebGraphEncoder,
    SparseHybridFunctionalChebGraphEncoder,
)


class GatedSSMBlock(nn.Module):
    """A lightweight Mamba-like gated state-space block.

    This is a dependency-free fallback for Codabench. It keeps the useful
    ingredients for this project: gated channel mixing, local temporal
    convolution, and a fast diagonal recurrent state over trial time.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=inner_dim,
        )
        self.state_logit = nn.Parameter(torch.zeros(inner_dim))
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        values, gates = self.in_proj(x_norm).chunk(2, dim=-1)
        values = self.conv(values.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        values = F.silu(values)
        gates = torch.sigmoid(gates)

        state_keep = torch.sigmoid(self.state_logit).view(1, -1)
        state = values.new_zeros(values.size(0), values.size(-1))
        outputs = []
        for time_index in range(values.size(1)):
            current = values[:, time_index]
            if mask is not None:
                current = current * mask[:, time_index : time_index + 1]
            state = state_keep * state + (1.0 - state_keep) * current
            outputs.append(state * gates[:, time_index])
        y = torch.stack(outputs, dim=1)
        x = residual + self.dropout(self.out_proj(y))
        return x + self.ffn(x)


class MambaSelectiveScanBlock(nn.Module):
    """Pure PyTorch Mamba block with input-dependent selective scan.

    This follows the Mamba mixer structure closely while avoiding custom CUDA
    kernels: input projection, causal depthwise convolution, input-dependent
    dt/B/C parameters, diagonal state matrix A, skip D, output gate z, and
    output projection. It is slower than mamba-ssm but works on Windows and is
    suitable for MER-PS trial lengths (~60-170 steps).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expansion: int = 2,
        dt_rank: int | str = "auto",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expansion * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._init_dt()

    def _init_dt(self) -> None:
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Inverse softplus of a small positive delta keeps early dynamics stable.
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(1e-3), math.log(1e-1))
        ).clamp_min(1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        x_in, z = self.in_proj(x_norm).chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)
        if mask is not None:
            x_conv = x_conv * mask.unsqueeze(-1)

        projected = self.x_proj(x_conv)
        dt_raw, b_param, c_param = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        y = self._selective_scan(x_conv, delta, b_param, c_param, mask)
        y = y * F.silu(z)
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return residual + self.dropout(self.out_proj(y))

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, steps, _ = x.shape
        a = -torch.exp(self.A_log.float()).to(x.dtype)
        state = x.new_zeros(batch, self.d_inner, self.d_state)
        outputs = []
        for time_index in range(steps):
            delta_t = delta[:, time_index]
            x_t = x[:, time_index]
            b_t = b_param[:, time_index]
            c_t = c_param[:, time_index]
            d_a = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            d_b_x = delta_t.unsqueeze(-1) * b_t.unsqueeze(1) * x_t.unsqueeze(-1)
            new_state = d_a * state + d_b_x
            if mask is not None:
                keep = mask[:, time_index].view(batch, 1, 1).to(dtype=x.dtype)
                state = torch.where(keep.bool(), new_state, state)
            else:
                state = new_state
            y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1) + self.D * x_t
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class TemporalConvMixerBlock(nn.Module):
    """ModernTCN/PatchMixer-style depthwise temporal mixer for small data."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        dilation: int = 1,
        expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.temporal = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=d_model,
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        mixed = self.temporal(x_norm.transpose(1, 2)).transpose(1, 2)
        if mixed.size(1) != x.size(1):
            mixed = mixed[:, : x.size(1)]
        mixed = F.gelu(mixed) * self.gate(x_norm)
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.channel_mixer(x)


class PatchTSTLiteBlock(nn.Module):
    """PatchTST-inspired temporal mixer over latent graph features.

    Trial lengths are short enough that a small Transformer over overlapping
    temporal patches is practical. The block projects each patch to one token,
    mixes patch tokens, then overlap-adds decoded patches back to per-second
    latent features.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int = 8,
        stride: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.norm = nn.LayerNorm(d_model)
        self.patch_proj = nn.Linear(d_model * patch_len, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_decode = nn.Linear(d_model, d_model * patch_len)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
        patches, valid_steps = self._make_patches(x_norm)
        tokens = self.patch_proj(patches.reshape(patches.size(0), patches.size(1), -1))
        patch_mask = None
        if mask is not None:
            patch_mask = self._patch_mask(mask, tokens.size(1))
        tokens = self.encoder(tokens, src_key_padding_mask=patch_mask)
        decoded = self.patch_decode(tokens).reshape(
            tokens.size(0),
            tokens.size(1),
            self.patch_len,
            x.size(-1),
        )
        mixed = self._overlap_add(decoded, valid_steps)[:, : x.size(1)]
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.ffn(x)

    def _make_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        steps = x.size(1)
        if steps <= self.patch_len:
            padded_steps = self.patch_len
        else:
            patch_count = math.ceil((steps - self.patch_len) / self.stride) + 1
            padded_steps = (patch_count - 1) * self.stride + self.patch_len
        if padded_steps > steps:
            x = F.pad(x, (0, 0, 0, padded_steps - steps))
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches.transpose(2, 3), padded_steps

    def _patch_mask(self, mask: torch.Tensor, patch_count: int) -> torch.Tensor:
        if mask.size(1) <= self.patch_len:
            padded_steps = self.patch_len
        else:
            padded_steps = (patch_count - 1) * self.stride + self.patch_len
        if padded_steps > mask.size(1):
            mask = F.pad(mask, (0, padded_steps - mask.size(1)))
        patch_valid = mask.unfold(dimension=1, size=self.patch_len, step=self.stride).amax(dim=-1)
        return ~patch_valid.bool()

    def _overlap_add(self, patches: torch.Tensor, steps: int) -> torch.Tensor:
        batch, patch_count, _, d_model = patches.shape
        output = patches.new_zeros(batch, steps, d_model)
        counts = patches.new_zeros(steps)
        for patch_index in range(patch_count):
            start = patch_index * self.stride
            end = start + self.patch_len
            output[:, start:end] = output[:, start:end] + patches[:, patch_index]
            counts[start:end] = counts[start:end] + 1.0
        return output / counts.clamp_min(1.0).view(1, -1, 1)


class TimesNetLiteBlock(nn.Module):
    """TimesNet-inspired temporal 2D variation block.

    The block detects dominant periods with FFT, reshapes the sequence into
    period grids, applies a small 2D convolution, and combines periods using
    amplitude weights.
    """

    def __init__(
        self,
        d_model: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)
        self.period_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
        periods, weights = self._dominant_periods(x_norm)
        outputs = []
        for period in periods:
            outputs.append(self._period_conv(x_norm, int(period))[:, : x.size(1)])
        if outputs:
            stacked = torch.stack(outputs, dim=-1)
            mixed = (stacked * weights.view(1, 1, 1, -1)).sum(dim=-1)
        else:
            mixed = x_norm
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.ffn(x)

    def _dominant_periods(self, x: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        steps = x.size(1)
        if steps < 2:
            return [1], x.new_ones(1)
        spectrum = torch.fft.rfft(x.float(), dim=1).abs().mean(dim=(0, 2))
        spectrum[0] = 0.0
        top_k = min(self.top_k, max(1, spectrum.numel() - 1))
        values, indices = torch.topk(spectrum, k=top_k)
        periods = [max(1, math.ceil(steps / max(1, int(index)))) for index in indices.tolist()]
        weights = torch.softmax(values.to(dtype=x.dtype), dim=0)
        return periods, weights

    def _period_conv(self, x: torch.Tensor, period: int) -> torch.Tensor:
        steps = x.size(1)
        padded_steps = math.ceil(steps / period) * period
        if padded_steps > steps:
            x = F.pad(x, (0, 0, 0, padded_steps - steps))
        grid = x.reshape(x.size(0), padded_steps // period, period, x.size(-1))
        grid = grid.permute(0, 3, 1, 2)
        grid = self.period_conv(grid)
        return grid.permute(0, 2, 3, 1).reshape(x.size(0), padded_steps, x.size(-1))


class ITransformerLiteBlock(nn.Module):
    """iTransformer-inspired inverted variable attention.

    Each latent dimension is treated as a variate token. A compact pooled
    lookback representation is used for attention across variables, then the
    resulting variable gates modulate the original temporal sequence.
    """

    def __init__(
        self,
        d_model: int,
        bins: int = 16,
        token_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bins = bins
        self.norm = nn.LayerNorm(d_model)
        self.token_proj = nn.Linear(bins, token_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
            nn.Sigmoid(),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
        tokens = F.adaptive_avg_pool1d(x_norm.transpose(1, 2), self.bins)
        tokens = self.token_proj(tokens)
        tokens = self.encoder(tokens)
        gate = self.gate(tokens).transpose(1, 2)
        value = self.value(tokens).transpose(1, 2)
        mixed = x_norm * gate + value
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.ffn(x)


class TimeMixerLiteBlock(nn.Module):
    """TimeMixer-inspired decomposable multiscale temporal mixing.

    The block separates a moving-average trend from residual variation, mixes
    both branches with different receptive fields, and lets a small gate choose
    how much low-frequency versus high-frequency evidence to inject.
    """

    def __init__(
        self,
        d_model: int,
        short_kernel: int = 3,
        long_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.short_kernel = short_kernel
        self.long_kernel = long_kernel
        self.norm = nn.LayerNorm(d_model)
        self.short_mixer = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            padding=short_kernel // 2,
            groups=d_model,
        )
        self.trend_mixer = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=long_kernel,
            padding=long_kernel // 2,
            groups=d_model,
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
        trend = self._moving_average(x_norm, self.long_kernel)
        seasonal = x_norm - trend
        seasonal_mixed = self.short_mixer(seasonal.transpose(1, 2)).transpose(1, 2)
        trend_mixed = self.trend_mixer(trend.transpose(1, 2)).transpose(1, 2)
        if seasonal_mixed.size(1) != x.size(1):
            seasonal_mixed = seasonal_mixed[:, : x.size(1)]
        if trend_mixed.size(1) != x.size(1):
            trend_mixed = trend_mixed[:, : x.size(1)]
        gate = self.gate(torch.cat([seasonal_mixed, trend_mixed], dim=-1))
        mixed = gate * seasonal_mixed + (1.0 - gate) * trend_mixed
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.channel_mixer(x)

    @staticmethod
    def _moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        left = kernel_size // 2
        right = kernel_size - 1 - left
        padded = F.pad(x.transpose(1, 2), (left, right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).transpose(1, 2)


class FourierLiteBlock(nn.Module):
    """FITS-inspired low-frequency residual mixer.

    Continuous MER-PS labels are slow 1 Hz trajectories, so a compact frequency
    branch is a natural complement to local SSM/conv blocks. The branch keeps a
    few low-frequency coefficients and learns per-channel amplitude/phase
    scaling before returning to the time domain.
    """

    def __init__(
        self,
        d_model: int,
        keep_ratio: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.norm = nn.LayerNorm(d_model)
        self.real_scale = nn.Parameter(torch.zeros(d_model))
        self.imag_scale = nn.Parameter(torch.zeros(d_model))
        self.time_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        if mask is not None:
            x_norm = x_norm * mask.unsqueeze(-1)
        freq = torch.fft.rfft(x_norm.float(), dim=1)
        keep = max(1, int(math.ceil(freq.size(1) * self.keep_ratio)))
        filtered = torch.zeros_like(freq)
        phase_scale = torch.complex(
            1.0 + self.real_scale.float(),
            self.imag_scale.float(),
        ).view(1, 1, -1)
        filtered[:, :keep] = freq[:, :keep] * phase_scale
        mixed = torch.fft.irfft(filtered, n=x.size(1), dim=1).to(dtype=x.dtype)
        gate = self.time_gate(x_norm)
        mixed = gate * mixed
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = residual + self.dropout(mixed)
        return x + self.ffn(x)


class SSMITransformerLiteBlock(nn.Module):
    """Local recurrent smoothing followed by inverted variable attention."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ssm = GatedSSMBlock(d_model=d_model, dropout=dropout)
        self.itransformer = ITransformerLiteBlock(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.itransformer(self.ssm(x, mask=mask), mask=mask)


class TimeITransformerLiteBlock(nn.Module):
    """Trend/seasonal decomposition followed by variable-wise attention."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.timemixer = TimeMixerLiteBlock(d_model=d_model, dropout=dropout)
        self.itransformer = ITransformerLiteBlock(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.itransformer(self.timemixer(x, mask=mask), mask=mask)


class HybridTemporalGateBlock(nn.Module):
    """Gated mixture of the useful temporal inductive biases.

    Branches are converted to residual deltas before mixing. This keeps the
    block close to the existing zero-init residual training setup while letting
    the model choose among local recurrence, variable attention, decomposition,
    and low-frequency evidence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                GatedSSMBlock(d_model=d_model, dropout=dropout),
                ITransformerLiteBlock(d_model=d_model, dropout=dropout),
                TimeMixerLiteBlock(d_model=d_model, dropout=dropout),
                FourierLiteBlock(d_model=d_model, dropout=dropout),
            ]
        )
        self.branch_logits = nn.Parameter(torch.tensor([0.0, 0.25, 0.0, -0.5]))
        self.mix_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        base = self.mix_norm(x)
        if mask is not None:
            base = base * mask.unsqueeze(-1)
        deltas = [branch(base, mask=mask) - base for branch in self.branches]
        stacked = torch.stack(deltas, dim=-1)
        weights = torch.softmax(self.branch_logits, dim=0).to(dtype=x.dtype)
        mixed = (stacked * weights.view(1, 1, 1, -1)).sum(dim=-1)
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1)
        x = x + self.dropout(mixed)
        return x + self.ffn(x)


class GraphMambaResidualRegressor(nn.Module):
    """Graph spatial encoder + Mamba-like temporal residual regressor."""

    def __init__(
        self,
        eeg_nodes: int = 64,
        eeg_features: int = 5,
        fnirs_nodes: int = 51,
        fnirs_features: int = 9,
        graph_hidden: int = 32,
        d_model: int = 128,
        cheb_order: int = 2,
        mamba_layers: int = 2,
        temporal_block: str = "mamba",
        d_state: int = 16,
        dropout: float = 0.15,
        output_dim: int = 2,
        graph_encoder: str = "adaptive",
        fusion_mode: str = "pool",
        eeg_scale_count: int = 1,
        fnirs_scale_count: int = 1,
    ):
        super().__init__()
        if eeg_features % eeg_scale_count != 0:
            raise ValueError("eeg_features must be divisible by eeg_scale_count")
        if fnirs_features % fnirs_scale_count != 0:
            raise ValueError("fnirs_features must be divisible by fnirs_scale_count")
        self.eeg_scale_count = eeg_scale_count
        self.fnirs_scale_count = fnirs_scale_count
        self.eeg_base_features = eeg_features // eeg_scale_count
        self.fnirs_base_features = fnirs_features // fnirs_scale_count
        if eeg_scale_count > 1:
            self.eeg_scale_logits = nn.Parameter(torch.zeros(eeg_scale_count))
        else:
            self.register_parameter("eeg_scale_logits", None)
        if fnirs_scale_count > 1:
            self.fnirs_scale_logits = nn.Parameter(torch.zeros(fnirs_scale_count))
        else:
            self.register_parameter("fnirs_scale_logits", None)
        if graph_encoder == "adaptive":
            encoder_cls = AdaptiveChebGraphEncoder
        elif graph_encoder == "signed":
            encoder_cls = SignedAdaptiveChebGraphEncoder
        elif graph_encoder == "functional":
            encoder_cls = FunctionalChebGraphEncoder
        elif graph_encoder == "hybrid_functional":
            encoder_cls = HybridFunctionalChebGraphEncoder
        elif graph_encoder == "sparse_functional":
            encoder_cls = SparseFunctionalChebGraphEncoder
        elif graph_encoder == "sparse_hybrid_functional":
            encoder_cls = SparseHybridFunctionalChebGraphEncoder
        else:
            raise ValueError(f"Unsupported graph_encoder: {graph_encoder}")
        self.eeg_encoder = encoder_cls(
            eeg_nodes, self.eeg_base_features, graph_hidden, cheb_order=cheb_order, dropout=dropout
        )
        self.fnirs_encoder = encoder_cls(
            fnirs_nodes,
            self.fnirs_base_features,
            graph_hidden,
            cheb_order=cheb_order,
            dropout=dropout,
        )
        self.fusion_mode = fusion_mode
        if fusion_mode == "pool":
            pooled_dim = graph_hidden * 4
        elif fusion_mode == "cross_asac":
            self.eeg_to_fnirs = CrossModalBlock(graph_hidden, heads=4, dropout=dropout)
            self.fnirs_to_eeg = CrossModalBlock(graph_hidden, heads=4, dropout=dropout)
            pooled_dim = graph_hidden * 8
        elif fusion_mode == "modal_gate":
            self.eeg_fusion_proj = nn.Sequential(
                nn.LayerNorm(graph_hidden * 2),
                nn.Linear(graph_hidden * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fnirs_fusion_proj = nn.Sequential(
                nn.LayerNorm(graph_hidden * 2),
                nn.Linear(graph_hidden * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.modal_gate = nn.Sequential(
                nn.LayerNorm(graph_hidden * 4),
                nn.Linear(graph_hidden * 4, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )
            pooled_dim = 0
        elif fusion_mode == "local_global":
            self.eeg_region_logits = nn.Parameter(torch.zeros(8))
            self.fnirs_region_logits = nn.Parameter(torch.zeros(6))
            pooled_dim = graph_hidden * 8
        elif fusion_mode == "pool_local_global":
            self.eeg_region_logits = nn.Parameter(torch.zeros(8))
            self.fnirs_region_logits = nn.Parameter(torch.zeros(6))
            pooled_dim = graph_hidden * 12
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if pooled_dim:
            self.fusion = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.fusion = None
        block_cls: type[nn.Module]
        if temporal_block == "mamba":
            block_cls = MambaSelectiveScanBlock
            self.temporal = nn.ModuleList(
                [
                    block_cls(d_model=d_model, d_state=d_state, dropout=dropout)
                    for _ in range(mamba_layers)
                ]
            )
        elif temporal_block == "gated_ssm":
            block_cls = GatedSSMBlock
            self.temporal = nn.ModuleList(
                [block_cls(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "conv_mixer":
            self.temporal = nn.ModuleList(
                [
                    TemporalConvMixerBlock(
                        d_model=d_model,
                        dilation=2**layer_index,
                        dropout=dropout,
                    )
                    for layer_index in range(mamba_layers)
                ]
            )
        elif temporal_block == "patch_tst":
            self.temporal = nn.ModuleList(
                [PatchTSTLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "timesnet":
            self.temporal = nn.ModuleList(
                [TimesNetLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "itransformer":
            self.temporal = nn.ModuleList(
                [ITransformerLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "timemixer":
            self.temporal = nn.ModuleList(
                [TimeMixerLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "fourier":
            self.temporal = nn.ModuleList(
                [FourierLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "ssm_itransformer":
            self.temporal = nn.ModuleList(
                [SSMITransformerLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "time_itransformer":
            self.temporal = nn.ModuleList(
                [TimeITransformerLiteBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        elif temporal_block == "hybrid_temporal":
            self.temporal = nn.ModuleList(
                [HybridTemporalGateBlock(d_model=d_model, dropout=dropout) for _ in range(mamba_layers)]
            )
        else:
            raise ValueError(f"Unsupported temporal_block: {temporal_block}")
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )
        self._reset_residual_head()

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        batch, steps = eeg.shape[:2]
        eeg_flat = eeg.reshape(batch * steps, eeg.size(2), eeg.size(3))
        fnirs_flat = fnirs.reshape(batch * steps, fnirs.size(2), fnirs.size(3))
        eeg_flat = self._fuse_scales(
            eeg_flat,
            self.eeg_scale_count,
            self.eeg_base_features,
            self.eeg_scale_logits,
        )
        fnirs_flat = self._fuse_scales(
            fnirs_flat,
            self.fnirs_scale_count,
            self.fnirs_base_features,
            self.fnirs_scale_logits,
        )

        eeg_graph = self.eeg_encoder(eeg_flat)
        fnirs_graph = self.fnirs_encoder(fnirs_flat)
        eeg_summary = torch.cat([eeg_graph.mean(dim=1), eeg_graph.amax(dim=1)], dim=1)
        fnirs_summary = torch.cat([fnirs_graph.mean(dim=1), fnirs_graph.amax(dim=1)], dim=1)
        pooled_parts = [eeg_summary, fnirs_summary]
        if self.fusion_mode == "cross_asac":
            eeg_cross = self.eeg_to_fnirs(eeg_graph, fnirs_graph)
            fnirs_cross = self.fnirs_to_eeg(fnirs_graph, eeg_graph)
            pooled_parts.extend(
                [
                    eeg_cross.mean(dim=1),
                    eeg_cross.amax(dim=1),
                    fnirs_cross.mean(dim=1),
                    fnirs_cross.amax(dim=1),
                ]
            )
        if self.fusion_mode == "modal_gate":
            eeg_token = self.eeg_fusion_proj(eeg_summary)
            fnirs_token = self.fnirs_fusion_proj(fnirs_summary)
            gates = torch.softmax(self.modal_gate(torch.cat([eeg_summary, fnirs_summary], dim=1)), dim=1)
            x = gates[:, :1] * eeg_token + gates[:, 1:] * fnirs_token
        elif self.fusion_mode == "local_global":
            pooled_parts = [
                self._local_global_summary(eeg_graph, self.eeg_region_logits),
                self._local_global_summary(fnirs_graph, self.fnirs_region_logits),
            ]
            pooled = torch.cat(pooled_parts, dim=1)
            x = self.fusion(pooled)
        elif self.fusion_mode == "pool_local_global":
            pooled_parts.extend(
                [
                    self._local_global_summary(eeg_graph, self.eeg_region_logits),
                    self._local_global_summary(fnirs_graph, self.fnirs_region_logits),
                ]
            )
            pooled = torch.cat(pooled_parts, dim=1)
            x = self.fusion(pooled)
        else:
            pooled = torch.cat(pooled_parts, dim=1)
            x = self.fusion(pooled)
        x = x.reshape(batch, steps, -1)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        for block in self.temporal:
            x = block(x, mask=mask)
            if mask is not None:
                x = x * mask.unsqueeze(-1)
        prediction = torch.tanh(self.head(x))
        if return_features:
            return prediction, x
        return prediction

    @staticmethod
    def _local_global_summary(x: torch.Tensor, region_logits: torch.Tensor) -> torch.Tensor:
        batch, nodes, hidden = x.shape
        groups = int(region_logits.numel())
        padded_nodes = math.ceil(nodes / groups) * groups
        if padded_nodes > nodes:
            pad = x.new_zeros(batch, padded_nodes - nodes, hidden)
            x_group = torch.cat([x, pad], dim=1)
        else:
            x_group = x
        region = x_group.reshape(batch, groups, padded_nodes // groups, hidden).mean(dim=2)
        weights = torch.softmax(region_logits, dim=0).view(1, groups, 1)
        local_weighted = (region * weights).sum(dim=1)
        local_max = region.amax(dim=1)
        global_mean = x.mean(dim=1)
        global_max = x.amax(dim=1)
        return torch.cat([local_weighted, local_max, global_mean, global_max], dim=1)

    def _reset_residual_head(self) -> None:
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _fuse_scales(
        x: torch.Tensor,
        scale_count: int,
        base_features: int,
        logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if scale_count == 1:
            return x
        weights = torch.softmax(logits, dim=0).view(1, 1, scale_count, 1)
        x = x.reshape(x.size(0), x.size(1), scale_count, base_features)
        return (x * weights).sum(dim=2)
