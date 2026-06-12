import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.seasonal_period import PeriodicInputConditioner, SeasonalPeriodModulator


def _bool_arg(args, name, default):
    value = getattr(args, name, default)
    return bool(default) if value is None else bool(value)


class SigmaEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, sigma, batch_size=None, device=None, dtype=None):
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(-1)
        if batch_size is not None and sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        elif batch_size is not None and sigma.numel() != batch_size:
            sigma = sigma[:1].expand(batch_size)

        c_noise = sigma.clamp_min(1e-8).log() / 4.0
        half_dim = self.embed_dim // 2
        if half_dim > 0:
            freq = torch.exp(
                -math.log(10000.0) * torch.arange(half_dim, device=c_noise.device, dtype=torch.float32)
                / max(half_dim - 1, 1)
            )
            emb = c_noise[:, None] * freq[None, :]
            emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        else:
            emb = c_noise[:, None]
        if emb.shape[1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[1]))
        emb = emb.to(dtype=dtype or torch.float32)
        return self.proj(emb)


class STResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation
        self.norm = nn.GroupNorm(num_groups=8 if channels % 8 == 0 else 1, num_channels=channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=3, padding=padding, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.emb = nn.Linear(channels, channels * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, sigma_emb):
        shift, scale = self.emb(sigma_emb).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        h = F.silu(self.depthwise(h))
        h = self.pointwise(h)
        h = self.dropout(h)
        return x + h


class VariableRelationBlock(nn.Module):
    def __init__(self, channels, rank=8, beta=0.10, init_beta=0.0, no_self=True):
        super().__init__()
        self.channels = int(channels)
        self.rank = max(1, min(int(rank), self.channels))
        self.beta = max(float(beta), 1e-6)
        self.no_self = bool(no_self)
        self.left = nn.Parameter(torch.randn(self.channels, self.rank) * 0.02)
        self.right = nn.Parameter(torch.randn(self.rank, self.channels) * 0.02)
        init = max(min(float(init_beta) / self.beta, 0.999), -0.999)
        self.beta_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))

    def forward(self, x):
        if self.channels <= 1:
            return x, x.new_tensor(0.0), x.new_tensor(0.0)
        relation = torch.matmul(self.left, self.right).to(device=x.device, dtype=x.dtype)
        if self.no_self:
            relation = relation * (1.0 - torch.eye(self.channels, device=x.device, dtype=x.dtype))
        relation = relation / relation.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        beta = self.beta * torch.tanh(self.beta_raw.to(device=x.device, dtype=x.dtype))
        related = torch.einsum("blk,km->blm", x, relation)
        out = x + beta * related
        return torch.nan_to_num(out), beta, relation.abs().mean()


class AdaptiveMovingAverage(nn.Module):
    def __init__(self, input_channels, kernels=None, affine=True, eps=1e-5):
        super().__init__()
        self.input_channels = int(input_channels)
        self.kernels = [1, 2, 4, 6, 12] if kernels is None else [max(1, int(k)) for k in kernels]
        self.kernel_mlp = nn.Linear(1, len(self.kernels))
        self.affine = bool(affine)
        self.eps = eps
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.input_channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.input_channels))

    @staticmethod
    def _moving_average(x_ts, kernel):
        length = x_ts.shape[1]
        kernel = min(max(1, int(kernel)), max(1, length))
        if kernel <= 1 or length <= 1:
            return x_ts
        left = kernel // 2
        right = (kernel - 1) // 2
        x = torch.cat(
            [
                x_ts[:, 0:1, :].repeat(1, left, 1),
                x_ts,
                x_ts[:, -1:, :].repeat(1, right, 1),
            ],
            dim=1,
        )
        x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=kernel, stride=1)
        return x.permute(0, 2, 1)

    def decompose(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        valid = [(idx, k) for idx, k in enumerate(self.kernels) if k <= max(1, x_ts.shape[1])]
        if not valid:
            valid = [(0, 1)]
        indices = [idx for idx, _ in valid]
        trends = torch.stack([self._moving_average(x_ts, k) for _, k in valid], dim=-1)
        logits = self.kernel_mlp(x_ts.unsqueeze(-1))[..., indices]
        weights = torch.softmax(logits, dim=-1).to(dtype=x_ts.dtype)
        trend = (trends * weights).sum(dim=-1)
        if self.affine:
            weight = self.affine_weight.to(device=x_ts.device, dtype=x_ts.dtype).view(1, 1, -1)
            bias = self.affine_bias.to(device=x_ts.device, dtype=x_ts.dtype).view(1, 1, -1)
            trend = trend * weight + bias
        season = x_ts - trend
        return torch.nan_to_num(season), torch.nan_to_num(trend)

    def restore(self, season, trend):
        season = torch.nan_to_num(season)
        trend = torch.nan_to_num(trend)
        if self.affine:
            weight = self.affine_weight.to(device=trend.device, dtype=trend.dtype).view(1, 1, -1)
            bias = self.affine_bias.to(device=trend.device, dtype=trend.dtype).view(1, 1, -1)
            trend = (trend - bias) / (weight + self.eps * self.eps)
        return torch.nan_to_num(season + trend)


class MultiScaleTemporalContextTokenAdapter(nn.Module):
    def __init__(
        self,
        token_channels,
        scales=None,
        heads=4,
        max_scale=0.015,
        init_scale=0.003,
        length_mid=96.0,
        length_tau=32.0,
        dropout=0.0,
        zero_init=True,
    ):
        super().__init__()
        self.scales = [max(1, int(s)) for s in ([4, 16] if scales is None else scales)]
        self.max_scale = max(float(max_scale), 1e-6)
        self.length_mid = float(length_mid)
        self.length_tau = max(float(length_tau), 1e-6)
        heads = max(1, min(int(heads), int(token_channels)))
        while token_channels % heads != 0 and heads > 1:
            heads -= 1
        self.trend_attn = nn.ModuleDict()
        self.season_attn = nn.ModuleDict()
        for scale in self.scales:
            key = f"s{scale}"
            self.trend_attn[key] = nn.MultiheadAttention(token_channels, heads, batch_first=True, dropout=dropout)
            self.season_attn[key] = nn.MultiheadAttention(token_channels, heads, batch_first=True, dropout=dropout)
        self.trend_proj = nn.Sequential(
            nn.LayerNorm(token_channels, elementwise_affine=False),
            nn.Linear(token_channels, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, token_channels),
        )
        self.season_proj = nn.Sequential(
            nn.LayerNorm(token_channels, elementwise_affine=False),
            nn.Linear(token_channels, token_channels),
            nn.SiLU(),
            nn.Linear(token_channels, token_channels),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.trend_proj[-1].weight)
            nn.init.zeros_(self.trend_proj[-1].bias)
            nn.init.zeros_(self.season_proj[-1].weight)
            nn.init.zeros_(self.season_proj[-1].bias)

    @staticmethod
    def _scale_is_valid(seq_len, scale):
        if scale >= 16:
            return seq_len >= 128
        if scale >= 4:
            return seq_len >= 32
        return seq_len >= max(4, scale * 2)

    @staticmethod
    def _downsample_tokens(tokens, scale):
        seq_len = int(tokens.shape[1])
        x = tokens.permute(0, 2, 1)
        pad = (scale - seq_len % scale) % scale
        if pad > 0:
            x = F.pad(x, (0, pad), mode="replicate")
        pooled = F.avg_pool1d(x, kernel_size=scale, stride=scale)
        return pooled.permute(0, 2, 1)

    @staticmethod
    def _upsample_tokens(tokens, target_len):
        x = tokens.permute(0, 2, 1)
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return x.permute(0, 2, 1)

    def _length_gate(self, ref, seq_len):
        value = (float(seq_len) - self.length_mid) / self.length_tau
        return torch.sigmoid(ref.new_tensor(value))

    def _branch_residual(self, tokens, attn_modules):
        seq_len = int(tokens.shape[1])
        residuals = []
        used_scales = []
        for scale in self.scales:
            if not self._scale_is_valid(seq_len, scale):
                continue
            key = f"s{scale}"
            pooled = self._downsample_tokens(tokens, scale)
            attended, _ = attn_modules[key](pooled, pooled, pooled, need_weights=False)
            residuals.append(self._upsample_tokens(attended - pooled, seq_len))
            used_scales.append(scale)
        if not residuals:
            return torch.zeros_like(tokens), used_scales
        return torch.stack(residuals, dim=0).mean(dim=0), used_scales

    def forward(self, trend_tokens, season_tokens):
        seq_len = int(trend_tokens.shape[1])
        trend_res, trend_used = self._branch_residual(trend_tokens, self.trend_attn)
        season_res, season_used = self._branch_residual(season_tokens, self.season_attn)
        used_scales = sorted(set(trend_used + season_used))
        if not used_scales:
            zero = trend_tokens.new_tensor(0.0)
            return trend_tokens, season_tokens, {
                "mstc_token_enabled": zero,
                "mstc_token_length_gate": self._length_gate(trend_tokens, seq_len),
                "mstc_token_strength": zero,
                "mstc_token_scale": zero,
                "mstc_token_norm": zero,
                "mstc_token_scale_count": 0.0,
                "mstc_token_max_scale_used": 0.0,
            }
        length_gate = self._length_gate(trend_tokens, seq_len)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=trend_tokens.device, dtype=trend_tokens.dtype))
        strength = torch.nan_to_num(length_gate * scale)
        trend_update = self.trend_proj(trend_res)
        season_update = self.season_proj(season_res)
        trend_tokens = trend_tokens + strength * trend_update
        season_tokens = season_tokens + strength * season_update
        update_norm = 0.5 * (
            torch.nan_to_num(trend_update).square().mean().sqrt()
            + torch.nan_to_num(season_update).square().mean().sqrt()
        )
        details = {
            "mstc_token_enabled": trend_tokens.new_tensor(1.0),
            "mstc_token_length_gate": torch.nan_to_num(length_gate),
            "mstc_token_scale": torch.nan_to_num(scale.abs()),
            "mstc_token_strength": torch.nan_to_num(strength.abs()),
            "mstc_token_norm": torch.nan_to_num(update_norm),
            "mstc_token_scale_count": float(len(used_scales)),
            "mstc_token_max_scale_used": float(max(used_scales)),
        }
        return torch.nan_to_num(trend_tokens), torch.nan_to_num(season_tokens), details


class TransFusionTemporalEncoding(nn.Module):
    def __init__(self, latent_dim, max_len=4096):
        super().__init__()
        latent_dim = int(latent_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(latent_dim, 1))
        )
        pe = torch.zeros(max_len, latent_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if latent_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.shape[1] <= self.pe.shape[1]:
            return x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)
        extra = x.shape[1] - self.pe.shape[1]
        tail = self.pe[:, -1:].expand(1, extra, -1)
        pe = torch.cat([self.pe, tail], dim=1)
        return x + pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)


class TransFusionTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        latent_dim=64,
        num_heads=4,
        num_layers=2,
        ff_size=256,
        dropout=0.0,
        patch_short=1,
        patch_mid=4,
        patch_long=16,
        mid_threshold=48,
        long_threshold=200,
        length_mid=96.0,
        length_tau=32.0,
        input_clip=None,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.patch_short = max(1, int(patch_short))
        self.patch_mid = max(1, int(patch_mid))
        self.patch_long = max(1, int(patch_long))
        self.mid_threshold = max(1, int(mid_threshold))
        self.long_threshold = max(self.mid_threshold + 1, int(long_threshold))
        self.length_mid = float(length_mid)
        self.length_tau = max(float(length_tau), 1e-6)
        self.input_clip = None if input_clip is None else max(float(input_clip), 0.0)
        heads = max(1, min(int(num_heads), self.latent_dim))
        while self.latent_dim % heads != 0 and heads > 1:
            heads -= 1
        self.in_proj = nn.Linear(self.input_channels, self.latent_dim)
        self.sigma_embed = SigmaEmbedding(self.latent_dim)
        self.pos = TransFusionTemporalEncoding(self.latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=heads,
            dim_feedforward=max(int(ff_size), self.latent_dim * 2),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.out_norm = nn.LayerNorm(self.latent_dim, elementwise_affine=False)

    def patch_size(self, seq_len):
        seq_len = int(seq_len)
        if seq_len <= self.mid_threshold:
            return self.patch_short
        if seq_len <= self.long_threshold:
            return self.patch_mid
        return self.patch_long

    def length_gate(self, ref, seq_len):
        value = (float(seq_len) - self.length_mid) / self.length_tau
        return torch.sigmoid(ref.new_tensor(value))

    @staticmethod
    def _pool_patches(x, patch):
        seq_len = int(x.shape[1])
        patch = max(1, int(patch))
        if patch <= 1:
            return x, 0
        y = x.permute(0, 2, 1)
        pad = (patch - seq_len % patch) % patch
        if pad > 0:
            y = F.pad(y, (0, pad), mode="replicate")
        y = F.avg_pool1d(y, kernel_size=patch, stride=patch)
        return y.permute(0, 2, 1), pad

    @staticmethod
    def upsample_tokens(tokens, target_len):
        if tokens.shape[1] == target_len:
            return tokens
        y = tokens.permute(0, 2, 1)
        y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.permute(0, 2, 1)

    def forward(self, x_ts, sigma):
        x_ts = torch.nan_to_num(x_ts)
        if self.input_clip is not None and self.input_clip > 0:
            x_ts = x_ts.clamp(min=-self.input_clip, max=self.input_clip)
        batch, seq_len, channels = x_ts.shape
        if channels != self.input_channels:
            raise ValueError(f"TFDA expected {self.input_channels} channels, got {channels}.")
        patch = self.patch_size(seq_len)
        pooled, _ = self._pool_patches(x_ts, patch)
        tokens = self.in_proj(pooled)
        sigma_token = self.sigma_embed(sigma, batch_size=batch, device=x_ts.device, dtype=x_ts.dtype).unsqueeze(1)
        tokens = torch.cat([sigma_token.to(tokens.dtype), tokens], dim=1)
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        tokens = self.out_norm(tokens[:, 1:, :])
        return torch.nan_to_num(tokens), {
            "patch": float(patch),
            "token_count": float(tokens.shape[1]),
            "length_gate": self.length_gate(x_ts, seq_len),
        }


class TransFusionFilmConditioner(nn.Module):
    def __init__(
        self,
        input_channels,
        emb_channels,
        latent_dim=64,
        num_heads=4,
        num_layers=2,
        ff_size=256,
        dropout=0.0,
        patch_short=1,
        patch_mid=4,
        patch_long=16,
        mid_threshold=48,
        long_threshold=200,
        length_mid=96.0,
        length_tau=32.0,
        max_scale=0.008,
        init_scale=0.0015,
        input_clip=3.0,
        zero_init=True,
    ):
        super().__init__()
        self.max_scale = max(float(max_scale), 1e-6)
        self.encoder = TransFusionTemporalEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_size=ff_size,
            dropout=dropout,
            patch_short=patch_short,
            patch_mid=patch_mid,
            patch_long=patch_long,
            mid_threshold=mid_threshold,
            long_threshold=long_threshold,
            length_mid=length_mid,
            length_tau=length_tau,
            input_clip=input_clip,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(int(latent_dim) * 3, elementwise_affine=False),
            nn.Linear(int(latent_dim) * 3, int(latent_dim) * 2),
            nn.SiLU(),
            nn.Linear(int(latent_dim) * 2, int(emb_channels)),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x_ts, sigma):
        tokens, enc_details = self.encoder(x_ts, sigma)
        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        max_abs = tokens.abs().amax(dim=1)
        feat = torch.cat([mean, std, max_abs], dim=1)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=x_ts.device, dtype=x_ts.dtype))
        length_gate = enc_details["length_gate"].to(device=x_ts.device, dtype=x_ts.dtype)
        st_film = torch.nan_to_num(length_gate * scale * self.proj(feat))
        return st_film, {
            "tfda_film_enabled": x_ts.new_tensor(1.0),
            "tfda_film_length_gate": torch.nan_to_num(length_gate),
            "tfda_film_scale": torch.nan_to_num(scale.abs()),
            "tfda_film_norm": torch.nan_to_num(st_film).square().mean().sqrt(),
            "tfda_film_patch": enc_details["patch"],
            "tfda_film_token_count": enc_details["token_count"],
        }


class TransFusionTokenContextAdapter(nn.Module):
    def __init__(
        self,
        token_channels,
        latent_dim=64,
        num_heads=4,
        num_layers=2,
        ff_size=256,
        dropout=0.0,
        patch_short=1,
        patch_mid=4,
        patch_long=16,
        mid_threshold=48,
        long_threshold=200,
        length_mid=96.0,
        length_tau=32.0,
        max_scale=0.006,
        init_scale=0.001,
        zero_init=True,
    ):
        super().__init__()
        self.max_scale = max(float(max_scale), 1e-6)
        kwargs = dict(
            input_channels=int(token_channels),
            latent_dim=int(latent_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_size=int(ff_size),
            dropout=float(dropout),
            patch_short=patch_short,
            patch_mid=patch_mid,
            patch_long=patch_long,
            mid_threshold=mid_threshold,
            long_threshold=long_threshold,
            length_mid=length_mid,
            length_tau=length_tau,
            input_clip=None,
        )
        self.trend_encoder = TransFusionTemporalEncoder(**kwargs)
        self.season_encoder = TransFusionTemporalEncoder(**kwargs)
        self.trend_proj = nn.Sequential(
            nn.LayerNorm(int(latent_dim), elementwise_affine=False),
            nn.Linear(int(latent_dim), int(token_channels)),
        )
        self.season_proj = nn.Sequential(
            nn.LayerNorm(int(latent_dim), elementwise_affine=False),
            nn.Linear(int(latent_dim), int(token_channels)),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.trend_proj[-1].weight)
            nn.init.zeros_(self.trend_proj[-1].bias)
            nn.init.zeros_(self.season_proj[-1].weight)
            nn.init.zeros_(self.season_proj[-1].bias)

    def forward(self, trend_tokens, season_tokens, sigma):
        seq_len = int(trend_tokens.shape[1])
        trend_ctx, trend_details = self.trend_encoder(trend_tokens, sigma)
        season_ctx, season_details = self.season_encoder(season_tokens, sigma)
        trend_ctx = self.trend_encoder.upsample_tokens(trend_ctx, seq_len)
        season_ctx = self.season_encoder.upsample_tokens(season_ctx, seq_len)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=trend_tokens.device, dtype=trend_tokens.dtype))
        length_gate = trend_details["length_gate"].to(device=trend_tokens.device, dtype=trend_tokens.dtype)
        strength = torch.nan_to_num(length_gate * scale)
        trend_update = self.trend_proj(trend_ctx)
        season_update = self.season_proj(season_ctx)
        trend_tokens = trend_tokens + strength * trend_update
        season_tokens = season_tokens + strength * season_update
        update_norm = 0.5 * (
            torch.nan_to_num(trend_update).square().mean().sqrt()
            + torch.nan_to_num(season_update).square().mean().sqrt()
        )
        return torch.nan_to_num(trend_tokens), torch.nan_to_num(season_tokens), {
            "tfda_token_enabled": trend_tokens.new_tensor(1.0),
            "tfda_token_length_gate": torch.nan_to_num(length_gate),
            "tfda_token_scale": torch.nan_to_num(scale.abs()),
            "tfda_token_strength": torch.nan_to_num(strength.abs()),
            "tfda_token_norm": torch.nan_to_num(update_norm),
            "tfda_token_patch": trend_details["patch"],
            "tfda_token_token_count": trend_details["token_count"],
            "tfda_token_season_patch": season_details["patch"],
        }


class STDenoiser(nn.Module):
    def __init__(
        self,
        input_channels,
        st_channels=64,
        st_res_layers=2,
        st_nheads=4,
        st_freq_tier=1,
        kernels=None,
        zero_init=True,
        use_var_relation=True,
        var_relation_rank=8,
        var_relation_beta=0.10,
        var_relation_init_beta=0.0,
        var_relation_no_self=True,
        lma_affine=True,
        use_period_branch=True,
        period_candidates=None,
        period_min=2,
        period_max=None,
        period_temperature=0.35,
        period_max_scale=0.20,
        st_dropout=0.0,
        use_mstc_token_adapter=False,
        mstc_token_scales=None,
        mstc_token_heads=4,
        mstc_token_max_scale=0.015,
        mstc_token_init_scale=0.003,
        mstc_token_length_mid=96.0,
        mstc_token_length_tau=32.0,
        mstc_token_dropout=0.0,
        mstc_token_zero_init=True,
        use_tfda_token_context=False,
        tfda_latent_dim=64,
        tfda_num_heads=4,
        tfda_num_layers=2,
        tfda_ff_size=256,
        tfda_dropout=0.0,
        tfda_patch_short=1,
        tfda_patch_mid=4,
        tfda_patch_long=16,
        tfda_mid_threshold=48,
        tfda_long_threshold=200,
        tfda_length_mid=96.0,
        tfda_length_tau=32.0,
        tfda_token_max_scale=0.006,
        tfda_token_init_scale=0.001,
        tfda_zero_init=True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.kernels = [1, 2, 4, 6, 12] if kernels is None else [max(1, int(k)) for k in kernels]
        self.lma = AdaptiveMovingAverage(input_channels, self.kernels, affine=lma_affine)
        self.sigma_embed = SigmaEmbedding(st_channels)
        self.trend_in = nn.Conv1d(input_channels, st_channels, kernel_size=1)
        self.season_in = nn.Conv1d(input_channels, st_channels, kernel_size=1)
        self.context_trend_in = nn.Conv1d(input_channels, st_channels, kernel_size=1)
        self.context_season_in = nn.Conv1d(input_channels, st_channels, kernel_size=1)
        self.trend_blocks = nn.ModuleList([STResidualBlock(st_channels, dilation=1, dropout=st_dropout) for _ in range(st_res_layers)])
        season_dilation = max(1, int(st_freq_tier))
        self.season_blocks = nn.ModuleList(
            [STResidualBlock(st_channels, dilation=season_dilation, dropout=st_dropout) for _ in range(st_res_layers)]
        )
        self.period_modulator = (
            SeasonalPeriodModulator(
                channels=st_channels,
                candidate_periods=period_candidates,
                min_period=period_min,
                max_period=period_max,
                temperature=period_temperature,
                max_scale=period_max_scale,
                zero_init=zero_init,
            )
            if use_period_branch
            else None
        )
        self.mstc_token_adapter = (
            MultiScaleTemporalContextTokenAdapter(
                token_channels=st_channels,
                scales=mstc_token_scales,
                heads=mstc_token_heads,
                max_scale=mstc_token_max_scale,
                init_scale=mstc_token_init_scale,
                length_mid=mstc_token_length_mid,
                length_tau=mstc_token_length_tau,
                dropout=mstc_token_dropout,
                zero_init=mstc_token_zero_init,
            )
            if use_mstc_token_adapter
            else None
        )
        self.tfda_token_adapter = (
            TransFusionTokenContextAdapter(
                token_channels=st_channels,
                latent_dim=tfda_latent_dim,
                num_heads=tfda_num_heads,
                num_layers=tfda_num_layers,
                ff_size=tfda_ff_size,
                dropout=tfda_dropout,
                patch_short=tfda_patch_short,
                patch_mid=tfda_patch_mid,
                patch_long=tfda_patch_long,
                mid_threshold=tfda_mid_threshold,
                long_threshold=tfda_long_threshold,
                length_mid=tfda_length_mid,
                length_tau=tfda_length_tau,
                max_scale=tfda_token_max_scale,
                init_scale=tfda_token_init_scale,
                zero_init=tfda_zero_init,
            )
            if use_tfda_token_context
            else None
        )
        heads = max(1, min(int(st_nheads), st_channels))
        while st_channels % heads != 0 and heads > 1:
            heads -= 1
        self.trend_cross_attn = nn.MultiheadAttention(st_channels, heads, batch_first=True)
        self.season_cross_attn = nn.MultiheadAttention(st_channels, heads, batch_first=True)
        self.trend_norm = nn.LayerNorm(st_channels, elementwise_affine=False)
        self.season_norm = nn.LayerNorm(st_channels, elementwise_affine=False)
        self.trend_mlp = nn.Sequential(
            nn.Linear(st_channels, st_channels * 4),
            nn.SiLU(),
            nn.Linear(st_channels * 4, st_channels),
        )
        self.season_mlp = nn.Sequential(
            nn.Linear(st_channels, st_channels * 4),
            nn.SiLU(),
            nn.Linear(st_channels * 4, st_channels),
        )
        self.trend_out = nn.Sequential(
            nn.Linear(st_channels, st_channels),
            nn.SiLU(),
            nn.Linear(st_channels, input_channels),
        )
        self.season_out = nn.Sequential(
            nn.Linear(st_channels, st_channels),
            nn.SiLU(),
            nn.Linear(st_channels, input_channels),
        )
        self.confidence_out = nn.Sequential(
            nn.Linear(st_channels, st_channels),
            nn.SiLU(),
            nn.Linear(st_channels, input_channels),
        )
        self.var_relation = (
            VariableRelationBlock(
                input_channels,
                rank=var_relation_rank,
                beta=var_relation_beta,
                init_beta=var_relation_init_beta,
                no_self=var_relation_no_self,
            )
            if use_var_relation and input_channels > 1
            else None
        )
        if zero_init:
            for proj in (self.trend_out[-1], self.season_out[-1]):
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            nn.init.zeros_(self.confidence_out[-1].weight)
            nn.init.zeros_(self.confidence_out[-1].bias)

    def decompose(self, x_ts):
        season, trend = self.lma.decompose(x_ts)
        return trend, season

    def forward(self, base_ts_hat, sigma, residual_context=None, return_details=False):
        base_ts_hat = torch.nan_to_num(base_ts_hat)
        batch, _, channels = base_ts_hat.shape
        if channels != self.input_channels:
            raise ValueError(f"STDenoiser expected {self.input_channels} variables, got {channels}.")
        if residual_context is None:
            residual_context = torch.zeros_like(base_ts_hat)
        else:
            residual_context = torch.nan_to_num(residual_context.to(device=base_ts_hat.device, dtype=base_ts_hat.dtype))
            if residual_context.shape != base_ts_hat.shape:
                raise ValueError(
                    f"STDenoiser residual context shape {tuple(residual_context.shape)} "
                    f"does not match base_ts_hat shape {tuple(base_ts_hat.shape)}."
                )

        sigma_emb = self.sigma_embed(
            sigma, batch_size=batch, device=base_ts_hat.device, dtype=base_ts_hat.dtype
        ).to(base_ts_hat.dtype)
        trend, season_ts = self.decompose(base_ts_hat)
        context_trend, context_season = self.decompose(residual_context)
        trend = self.trend_in(trend.permute(0, 2, 1)) + self.context_trend_in(context_trend.permute(0, 2, 1))
        season = self.season_in(season_ts.permute(0, 2, 1)) + self.context_season_in(context_season.permute(0, 2, 1))

        for block in self.trend_blocks:
            trend = block(trend, sigma_emb)
        for block in self.season_blocks:
            season = block(season, sigma_emb)

        trend_tokens = trend.permute(0, 2, 1)
        season_tokens = season.permute(0, 2, 1)
        period_details = {}
        if self.period_modulator is not None:
            season_tokens, period_details = self.period_modulator(season_ts, season_tokens)
        mstc_token_details = {}
        if self.mstc_token_adapter is not None:
            trend_tokens, season_tokens, mstc_token_details = self.mstc_token_adapter(trend_tokens, season_tokens)
        tfda_token_details = {}
        if self.tfda_token_adapter is not None:
            trend_tokens, season_tokens, tfda_token_details = self.tfda_token_adapter(
                trend_tokens, season_tokens, sigma
            )
        trend_cross, _ = self.trend_cross_attn(trend_tokens, season_tokens, season_tokens, need_weights=False)
        season_cross, _ = self.season_cross_attn(season_tokens, trend_tokens, trend_tokens, need_weights=False)
        trend_tokens = trend_tokens + trend_cross
        season_tokens = season_tokens + season_cross
        trend_tokens = trend_tokens + self.trend_mlp(self.trend_norm(trend_tokens))
        season_tokens = season_tokens + self.season_mlp(self.season_norm(season_tokens))
        trend_delta_ts = self.trend_out(trend_tokens)
        season_delta_ts = self.season_out(season_tokens)
        delta_raw_ts = self.lma.restore(season_delta_ts, trend_delta_ts)
        relation_beta = base_ts_hat.new_tensor(0.0)
        relation_norm = base_ts_hat.new_tensor(0.0)
        if self.var_relation is not None:
            delta_raw_ts, relation_beta, relation_norm = self.var_relation(delta_raw_ts)
        confidence = torch.sigmoid(self.confidence_out(0.5 * (trend_tokens + season_tokens)))
        delta_ts = torch.nan_to_num(delta_raw_ts * confidence)
        if return_details:
            return delta_ts, {
                "delta_raw_ts": torch.nan_to_num(delta_raw_ts),
                "trend_delta_ts": torch.nan_to_num(trend_delta_ts),
                "season_delta_ts": torch.nan_to_num(season_delta_ts),
                "confidence": torch.nan_to_num(confidence),
                "relation_beta": relation_beta,
                "relation_norm": relation_norm,
                "period_details": period_details,
                "mstc_token_details": mstc_token_details,
                "tfda_token_details": tfda_token_details,
            }
        return delta_ts


class STFeatureConditioner(nn.Module):
    def __init__(
        self,
        input_channels,
        emb_channels,
        hidden_channels=64,
        kernels=None,
        max_scale=0.10,
        init_scale=0.02,
        input_clip=3.0,
        zero_init=True,
    ):
        super().__init__()
        self.max_scale = max(float(max_scale), 1e-6)
        self.input_clip = None if input_clip is None else max(float(input_clip), 0.0)
        self.lma = AdaptiveMovingAverage(
            input_channels,
            [1, 2, 4, 6, 12] if kernels is None else kernels,
            affine=False,
        )
        self.sigma_embed = SigmaEmbedding(hidden_channels)
        self.trend_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.season_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_channels * 5),
            nn.Linear(hidden_channels * 5, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, emb_channels),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _pool_stats(x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        return mean, std

    def forward(self, x_ts, sigma):
        x_ts = torch.nan_to_num(x_ts)
        if self.input_clip is not None and self.input_clip > 0:
            x_ts = x_ts.clamp(min=-self.input_clip, max=self.input_clip)
        batch = x_ts.shape[0]
        season, trend = self.lma.decompose(x_ts)
        trend_h = F.silu(self.trend_conv(trend.permute(0, 2, 1)))
        season_h = F.silu(self.season_conv(season.permute(0, 2, 1)))
        trend_mean, trend_std = self._pool_stats(trend_h)
        season_mean, season_std = self._pool_stats(season_h)
        sigma_h = self.sigma_embed(sigma, batch_size=batch, device=x_ts.device, dtype=x_ts.dtype)
        feat = torch.cat([trend_mean, trend_std, season_mean, season_std, sigma_h], dim=-1)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=x_ts.device, dtype=x_ts.dtype))
        st_film = scale * self.proj(feat)
        return torch.nan_to_num(st_film), {
            "st_film_scale": torch.nan_to_num(scale).abs(),
            "st_film_norm": torch.nan_to_num(st_film).square().mean().sqrt(),
        }


class TransitionContextFilmConditioner(nn.Module):
    def __init__(
        self,
        input_channels,
        emb_channels,
        hidden_channels=64,
        scales=None,
        max_scale=0.012,
        init_scale=0.002,
        input_clip=3.0,
        length_mid=96.0,
        length_tau=32.0,
        zero_init=True,
    ):
        super().__init__()
        self.scales = [max(1, int(s)) for s in ([4, 16] if scales is None else scales)]
        self.max_scale = max(float(max_scale), 1e-6)
        self.input_clip = None if input_clip is None else max(float(input_clip), 0.0)
        self.length_mid = float(length_mid)
        self.length_tau = max(float(length_tau), 1e-6)
        hidden_channels = max(8, int(hidden_channels))
        self.sigma_embed = SigmaEmbedding(hidden_channels)
        self.context_conv = nn.Conv1d(input_channels * 4, hidden_channels, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_channels * 5),
            nn.Linear(hidden_channels * 5, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, emb_channels),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _scale_is_valid(seq_len, scale):
        if scale >= 16:
            return seq_len >= 128
        if scale >= 4:
            return seq_len >= 64
        return seq_len >= max(8, scale * 4)

    def _length_gate(self, ref, seq_len):
        value = (float(seq_len) - self.length_mid) / self.length_tau
        return torch.sigmoid(ref.new_tensor(value))

    def _transition_features(self, x_ts, scale):
        smooth = AdaptiveMovingAverage._moving_average(x_ts, scale)
        diff = torch.zeros_like(smooth)
        if smooth.shape[1] > scale:
            diff[:, scale:, :] = smooth[:, scale:, :] - smooth[:, :-scale, :]
        accel = torch.zeros_like(diff)
        if diff.shape[1] > scale:
            accel[:, scale:, :] = diff[:, scale:, :] - diff[:, :-scale, :]
        residual = x_ts - smooth
        return torch.cat([smooth, diff, accel, residual], dim=-1)

    @staticmethod
    def _pool_stats(x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        max_abs = x.abs().amax(dim=-1)
        tail = x[..., -max(1, x.shape[-1] // 4):].mean(dim=-1)
        return mean, std, max_abs, tail

    def forward(self, x_ts, sigma):
        x_ts = torch.nan_to_num(x_ts)
        if self.input_clip is not None and self.input_clip > 0:
            x_ts = x_ts.clamp(min=-self.input_clip, max=self.input_clip)
        batch, seq_len, _ = x_ts.shape
        valid_scales = [scale for scale in self.scales if self._scale_is_valid(seq_len, scale)]
        length_gate = self._length_gate(x_ts, seq_len)
        if not valid_scales:
            zero = x_ts.new_zeros(batch, self.proj[-1].out_features)
            return zero, {
                "transition_context_enabled": x_ts.new_tensor(0.0),
                "transition_context_length_gate": torch.nan_to_num(length_gate),
                "transition_context_scale": x_ts.new_tensor(0.0),
                "transition_context_norm": x_ts.new_tensor(0.0),
                "transition_context_scale_count": 0.0,
                "transition_context_max_scale_used": 0.0,
            }
        pooled_stats = []
        for scale_value in valid_scales:
            feat = self._transition_features(x_ts, scale_value)
            hidden = F.silu(self.context_conv(feat.permute(0, 2, 1)))
            pooled_stats.append(torch.stack(self._pool_stats(hidden), dim=1))
        stats = torch.stack(pooled_stats, dim=1)
        mean_avg = stats[:, :, 0, :].mean(dim=1)
        std_avg = stats[:, :, 1, :].mean(dim=1)
        max_avg = stats[:, :, 2, :].mean(dim=1)
        tail_avg = stats[:, :, 3, :].mean(dim=1)
        sigma_h = self.sigma_embed(sigma, batch_size=batch, device=x_ts.device, dtype=x_ts.dtype)
        context = torch.cat([mean_avg, std_avg, max_avg, tail_avg, sigma_h], dim=-1)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=x_ts.device, dtype=x_ts.dtype))
        st_film = torch.nan_to_num(length_gate * scale * self.proj(context))
        return st_film, {
            "transition_context_enabled": x_ts.new_tensor(1.0),
            "transition_context_length_gate": torch.nan_to_num(length_gate),
            "transition_context_scale": torch.nan_to_num(scale).abs(),
            "transition_context_norm": torch.nan_to_num(st_film).square().mean().sqrt(),
            "transition_context_scale_count": float(len(valid_scales)),
            "transition_context_max_scale_used": float(max(valid_scales)),
        }


class STEDMPrecondWrapper(nn.Module):
    def __init__(self, base_net, ts_to_img, img_to_ts, args):
        super().__init__()
        self.base_net = base_net
        self.ts_to_img = ts_to_img
        self.img_to_ts = img_to_ts
        self.current_epoch = 0
        self.st_warmup_epochs = max(int(getattr(args, "st_warmup_epochs", 200)), 0)
        self.total_epochs = max(int(getattr(args, "epochs", 0) or 0), 0)
        self.st_input_noise = float(getattr(args, "st_input_noise", 0.0) or 0.0)
        self.use_late_decay = _bool_arg(args, "use_late_decay", False)
        self.late_decay_st_strength = _bool_arg(args, "late_decay_st_strength", False)
        self.late_decay_start_epoch = max(int(getattr(args, "late_decay_start_epoch", 0) or 0), 0)
        self.late_decay_start_ratio = float(getattr(args, "late_decay_start_ratio", 0.70))
        self.late_decay_min_scale = min(1.0, max(0.0, float(getattr(args, "late_decay_min_scale", 1.0))))
        self.late_decay_power = max(float(getattr(args, "late_decay_power", 1.0)), 1e-6)
        self.use_sigma_gate = bool(getattr(args, "st_sigma_gate", True))
        self.use_context_sigma_gate = bool(getattr(args, "st_context_sigma_gate", True))
        self.sigma_mid = float(getattr(args, "st_sigma_mid", 0.0))
        self.sigma_scale = max(float(getattr(args, "st_sigma_scale", 2.0)), 1e-6)
        self.alpha_max = max(float(getattr(args, "st_alpha_max", 0.2)), 1e-6)
        self.use_feature_fusion = _bool_arg(args, "st_feature_fusion", True)
        self.use_feature_sigma_gate = _bool_arg(args, "st_feature_sigma_gate", True)
        self.feature_sigma_mid = float(getattr(args, "st_feature_sigma_mid", -1.0))
        self.feature_sigma_scale = max(float(getattr(args, "st_feature_sigma_scale", 0.75)), 1e-6)
        self.feature_warmup_epochs = max(
            int(getattr(args, "st_feature_warmup_epochs", self.st_warmup_epochs) or 0),
            0,
        )
        self.feature_norm_max = max(float(getattr(args, "st_feature_norm_max", 0.04)), 0.0)
        self.use_trust_gate = _bool_arg(args, "st_trust_gate", False)
        self.trust_max = max(float(getattr(args, "st_trust_max", 0.60)), 1e-6)
        trust_init = float(getattr(args, "st_trust_init", 0.10))
        trust_init = min(max(trust_init, 1e-6), self.trust_max - 1e-6)
        trust_ratio = min(max(trust_init / self.trust_max, 1e-6), 1.0 - 1e-6)
        self.st_trust_raw = None
        if self.use_trust_gate and _bool_arg(args, "st_trust_learnable", True):
            self.st_trust_raw = nn.Parameter(
                torch.tensor(math.log(trust_ratio / (1.0 - trust_ratio)), dtype=torch.float32)
            )
        elif self.use_trust_gate:
            self.register_buffer("st_trust_value", torch.tensor(trust_init, dtype=torch.float32))
        self.period_input_warmup_epochs = max(
            int(getattr(args, "st_period_input_warmup_epochs", self.st_warmup_epochs)),
            0,
        )
        self.period_input_alpha_max = max(float(getattr(args, "st_period_input_alpha_max", 0.05)), 1e-6)

        st_alpha = float(getattr(args, "st_alpha", 0.05))
        st_alpha = max(min(st_alpha, self.alpha_max - 1e-6), -self.alpha_max + 1e-6)
        if _bool_arg(args, "st_alpha_learnable", True):
            init = math.atanh(st_alpha / self.alpha_max)
            self.st_alpha_raw = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        else:
            self.register_buffer("st_alpha_value", torch.tensor(st_alpha, dtype=torch.float32))
            self.st_alpha_raw = None

        input_channels = self._infer_ts_channels(args)
        period_input_alpha = float(getattr(args, "st_period_input_alpha", 0.02))
        period_input_alpha = max(
            min(period_input_alpha, self.period_input_alpha_max - 1e-6),
            -self.period_input_alpha_max + 1e-6,
        )
        if _bool_arg(args, "st_period_input_alpha_learnable", True):
            init = math.atanh(period_input_alpha / self.period_input_alpha_max)
            self.period_input_alpha_raw = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        else:
            self.register_buffer("period_input_alpha_value", torch.tensor(period_input_alpha, dtype=torch.float32))
            self.period_input_alpha_raw = None

        kernels = getattr(args, "ds_lma_kernels", [1, 2, 4, 6, 12])
        self.st_denoiser = STDenoiser(
            input_channels=input_channels,
            st_channels=int(getattr(args, "st_channels", 64)),
            st_res_layers=int(getattr(args, "st_res_layers", 2)),
            st_nheads=int(getattr(args, "st_nheads", 4)),
            st_freq_tier=int(getattr(args, "st_freq_tier", 1)),
            kernels=kernels,
            zero_init=_bool_arg(args, "st_zero_init", True),
            use_var_relation=_bool_arg(args, "st_var_relation", True),
            var_relation_rank=int(getattr(args, "st_var_relation_rank", 8)),
            var_relation_beta=float(getattr(args, "st_var_relation_beta", 0.10)),
            var_relation_init_beta=float(getattr(args, "st_var_relation_init_beta", 0.0)),
            var_relation_no_self=_bool_arg(args, "st_var_relation_no_self", True),
            lma_affine=_bool_arg(args, "st_lma_affine", True),
            use_period_branch=_bool_arg(args, "st_period_branch", True),
            period_candidates=getattr(args, "st_period_candidates", None),
            period_min=int(getattr(args, "st_period_min", 2)),
            period_max=getattr(args, "st_period_max", None),
            period_temperature=float(getattr(args, "st_period_temperature", 0.35)),
            period_max_scale=float(getattr(args, "st_period_max_scale", 0.20)),
            st_dropout=float(getattr(args, "st_dropout", 0.0) or 0.0),
            use_mstc_token_adapter=_bool_arg(args, "use_mstc_token_adapter", False),
            mstc_token_scales=getattr(args, "mstc_token_scales", None),
            mstc_token_heads=int(getattr(args, "mstc_token_heads", 4)),
            mstc_token_max_scale=float(getattr(args, "mstc_token_max_scale", 0.015)),
            mstc_token_init_scale=float(getattr(args, "mstc_token_init_scale", 0.003)),
            mstc_token_length_mid=float(getattr(args, "mstc_token_length_mid", 96.0)),
            mstc_token_length_tau=float(getattr(args, "mstc_token_length_tau", 32.0)),
            mstc_token_dropout=float(getattr(args, "mstc_token_dropout", 0.0) or 0.0),
            mstc_token_zero_init=_bool_arg(args, "mstc_token_zero_init", True),
            use_tfda_token_context=_bool_arg(args, "use_tfda_token_context", False),
            tfda_latent_dim=int(getattr(args, "tfda_latent_dim", 64)),
            tfda_num_heads=int(getattr(args, "tfda_num_heads", 4)),
            tfda_num_layers=int(getattr(args, "tfda_num_layers", 2)),
            tfda_ff_size=int(getattr(args, "tfda_ff_size", 256)),
            tfda_dropout=float(getattr(args, "tfda_dropout", 0.0) or 0.0),
            tfda_patch_short=int(getattr(args, "tfda_patch_short", 1)),
            tfda_patch_mid=int(getattr(args, "tfda_patch_mid", 4)),
            tfda_patch_long=int(getattr(args, "tfda_patch_long", 16)),
            tfda_mid_threshold=int(getattr(args, "tfda_mid_threshold", 48)),
            tfda_long_threshold=int(getattr(args, "tfda_long_threshold", 200)),
            tfda_length_mid=float(getattr(args, "tfda_length_mid", 96.0)),
            tfda_length_tau=float(getattr(args, "tfda_length_tau", 32.0)),
            tfda_token_max_scale=float(getattr(args, "tfda_token_max_scale", 0.006)),
            tfda_token_init_scale=float(getattr(args, "tfda_token_init_scale", 0.001)),
            tfda_zero_init=_bool_arg(args, "tfda_zero_init", True),
        )
        self.period_input_conditioner = (
            PeriodicInputConditioner(
                input_channels=input_channels,
                hidden_channels=int(getattr(args, "st_period_input_channels", getattr(args, "st_channels", 64))),
                candidate_periods=getattr(args, "st_period_candidates", None),
                min_period=int(getattr(args, "st_period_min", 2)),
                max_period=getattr(args, "st_period_max", None),
                temperature=float(getattr(args, "st_period_temperature", 0.45)),
                max_scale=float(getattr(args, "st_period_input_max_scale", 0.20)),
                init_scale=float(getattr(args, "st_period_input_init_scale", 0.02)),
                zero_init=_bool_arg(args, "st_zero_init", True),
            )
            if _bool_arg(args, "st_period_input_condition", False)
            else None
        )
        base_model = getattr(self.base_net, "model", None)
        map_layer1 = getattr(base_model, "map_layer1", None)
        emb_channels = int(getattr(map_layer1, "out_features", 0) or 0)
        if emb_channels <= 0:
            emb_channels = int(getattr(args, "unet_channels", 128)) * 4
        self.feature_conditioner = (
            STFeatureConditioner(
                input_channels=input_channels,
                emb_channels=emb_channels,
                hidden_channels=int(getattr(args, "st_feature_channels", getattr(args, "st_channels", 64))),
                kernels=kernels,
                max_scale=float(getattr(args, "st_feature_scale_max", 0.10)),
                init_scale=float(getattr(args, "st_feature_init_scale", 0.02)),
                input_clip=getattr(args, "st_feature_input_clip", 3.0),
                zero_init=_bool_arg(args, "st_feature_zero_init", True),
            )
            if self.use_feature_fusion
            else None
        )
        self.transition_context_conditioner = (
            TransitionContextFilmConditioner(
                input_channels=input_channels,
                emb_channels=emb_channels,
                hidden_channels=int(getattr(args, "transition_context_hidden", getattr(args, "st_feature_channels", 64))),
                scales=getattr(args, "transition_context_scales", None),
                max_scale=float(getattr(args, "transition_context_max_scale", 0.012)),
                init_scale=float(getattr(args, "transition_context_init_scale", 0.002)),
                input_clip=getattr(args, "transition_context_input_clip", 3.0),
                length_mid=float(getattr(args, "transition_context_length_mid", 96.0)),
                length_tau=float(getattr(args, "transition_context_length_tau", 32.0)),
                zero_init=_bool_arg(args, "transition_context_zero_init", True),
            )
            if _bool_arg(args, "use_transition_context_film", False)
            else None
        )
        self.tfda_film_conditioner = (
            TransFusionFilmConditioner(
                input_channels=input_channels,
                emb_channels=emb_channels,
                latent_dim=int(getattr(args, "tfda_latent_dim", 64)),
                num_heads=int(getattr(args, "tfda_num_heads", 4)),
                num_layers=int(getattr(args, "tfda_num_layers", 2)),
                ff_size=int(getattr(args, "tfda_ff_size", 256)),
                dropout=float(getattr(args, "tfda_dropout", 0.0) or 0.0),
                patch_short=int(getattr(args, "tfda_patch_short", 1)),
                patch_mid=int(getattr(args, "tfda_patch_mid", 4)),
                patch_long=int(getattr(args, "tfda_patch_long", 16)),
                mid_threshold=int(getattr(args, "tfda_mid_threshold", 48)),
                long_threshold=int(getattr(args, "tfda_long_threshold", 200)),
                length_mid=float(getattr(args, "tfda_length_mid", 96.0)),
                length_tau=float(getattr(args, "tfda_length_tau", 32.0)),
                max_scale=float(getattr(args, "tfda_max_scale", 0.008)),
                init_scale=float(getattr(args, "tfda_init_scale", 0.0015)),
                input_clip=getattr(args, "tfda_input_clip", 3.0),
                zero_init=_bool_arg(args, "tfda_zero_init", True),
            )
            if _bool_arg(args, "use_tfda_film", False)
            else None
        )
        self._clear_st_state()

    @staticmethod
    def _infer_ts_channels(args):
        input_channels = int(getattr(args, "input_channels"))
        if bool(getattr(args, "use_stft", False)):
            return max(1, input_channels // 2)
        return input_channels

    @property
    def sigma_min(self):
        return self.base_net.sigma_min

    @property
    def sigma_max(self):
        return self.base_net.sigma_max

    def round_sigma(self, sigma):
        return self.base_net.round_sigma(sigma)

    def _effective_alpha(self, ref):
        if self.st_alpha_raw is None:
            return self.st_alpha_value.to(device=ref.device, dtype=ref.dtype)
        return self.alpha_max * torch.tanh(self.st_alpha_raw.to(device=ref.device, dtype=ref.dtype))

    def _period_input_alpha(self, ref):
        if self.period_input_alpha_raw is None:
            return self.period_input_alpha_value.to(device=ref.device, dtype=ref.dtype)
        raw = self.period_input_alpha_raw.to(device=ref.device, dtype=ref.dtype)
        return self.period_input_alpha_max * torch.tanh(raw)

    def _trust_gate(self, ref):
        if not self.use_trust_gate:
            return ref.new_tensor(1.0)
        if self.st_trust_raw is None:
            return self.st_trust_value.to(device=ref.device, dtype=ref.dtype)
        raw = self.st_trust_raw.to(device=ref.device, dtype=ref.dtype)
        return self.trust_max * torch.sigmoid(raw)

    def _warmup_gate(self, ref):
        if self.st_warmup_epochs <= 0:
            return ref.new_tensor(1.0)
        gate = min(1.0, max(0.0, float(self.current_epoch) / float(self.st_warmup_epochs)))
        return ref.new_tensor(gate)

    def _late_decay_gate(self, ref):
        if not self.use_late_decay or not self.late_decay_st_strength:
            return ref.new_tensor(1.0)
        if self.total_epochs <= 0:
            return ref.new_tensor(1.0)
        start_epoch = self.late_decay_start_epoch
        if start_epoch <= 0:
            start_epoch = int(max(1.0, float(self.total_epochs) * self.late_decay_start_ratio))
        if float(self.current_epoch) <= float(start_epoch):
            return ref.new_tensor(1.0)
        denom = max(float(self.total_epochs - start_epoch), 1.0)
        progress = min(1.0, max(0.0, (float(self.current_epoch) - float(start_epoch)) / denom))
        progress = progress ** self.late_decay_power
        scale = 1.0 - progress * (1.0 - self.late_decay_min_scale)
        return ref.new_tensor(scale)

    def _period_input_warmup_gate(self, ref):
        if self.period_input_warmup_epochs <= 0:
            return ref.new_tensor(1.0)
        gate = min(1.0, max(0.0, float(self.current_epoch) / float(self.period_input_warmup_epochs)))
        return ref.new_tensor(gate)

    def _feature_warmup_gate(self, ref):
        if self.feature_warmup_epochs <= 0:
            return ref.new_tensor(1.0)
        gate = min(1.0, max(0.0, float(self.current_epoch) / float(self.feature_warmup_epochs)))
        return ref.new_tensor(gate)

    def _sigma_gate(self, sigma, ref):
        if not self.use_sigma_gate:
            return ref.new_tensor(1.0)
        sigma = torch.as_tensor(sigma, device=ref.device, dtype=ref.dtype).clamp_min(1e-8)
        gate = torch.sigmoid((self.sigma_mid - sigma.log()) / self.sigma_scale)
        gate = gate.reshape(-1)
        batch = int(ref.shape[0])
        if gate.numel() == 1 and batch > 1:
            gate = gate.expand(batch)
        elif gate.numel() != batch:
            gate = gate[:1].expand(batch)
        return gate.reshape(-1, 1, 1, 1)

    def _feature_sigma_gate(self, sigma, ref):
        if not self.use_feature_sigma_gate:
            return ref.new_tensor(1.0)
        sigma = torch.as_tensor(sigma, device=ref.device, dtype=ref.dtype).clamp_min(1e-8)
        gate = torch.sigmoid((self.feature_sigma_mid - sigma.log()) / self.feature_sigma_scale)
        gate = gate.reshape(-1)
        batch = int(ref.shape[0])
        if gate.numel() == 1 and batch > 1:
            gate = gate.expand(batch)
        elif gate.numel() != batch:
            gate = gate[:1].expand(batch)
        return gate.reshape(-1, 1)

    def _limit_feature_norm(self, st_film):
        if self.feature_norm_max <= 0:
            return st_film
        norm = (st_film.square().mean(dim=1, keepdim=True) + 1e-8).sqrt()
        scale = (self.feature_norm_max / norm).clamp(max=1.0)
        return st_film * scale

    @staticmethod
    def _match_img_shape(delta_img, ref_img):
        if delta_img.shape == ref_img.shape:
            return delta_img
        delta_img = delta_img[..., : ref_img.shape[-2], : ref_img.shape[-1]]
        pad_h = ref_img.shape[-2] - delta_img.shape[-2]
        pad_w = ref_img.shape[-1] - delta_img.shape[-1]
        if pad_h > 0 or pad_w > 0:
            delta_img = F.pad(delta_img, (0, max(0, pad_w), 0, max(0, pad_h)))
        if delta_img.shape[1] > ref_img.shape[1]:
            delta_img = delta_img[:, : ref_img.shape[1]]
        elif delta_img.shape[1] < ref_img.shape[1]:
            pad_c = ref_img.shape[1] - delta_img.shape[1]
            delta_img = F.pad(delta_img, (0, 0, 0, 0, 0, pad_c))
        return delta_img

    def _clear_st_state(self):
        self.last_st_state = None

    def pop_st_state(self):
        state = self.last_st_state
        self._clear_st_state()
        return state

    def forward(self, x_img, sigma, class_labels=None, **kwargs):
        self._clear_st_state()
        x_img = x_img.to(torch.float32)
        noisy_ts = self.img_to_ts(x_img)
        x_for_base = x_img
        period_input_details = {}
        period_input_gate = x_img.new_tensor(0.0)
        period_input_alpha = x_img.new_tensor(0.0)
        late_decay_gate = self._late_decay_gate(x_img)
        if self.period_input_conditioner is not None:
            _, noisy_season = self.st_denoiser.decompose(noisy_ts.detach())
            cond_ts, period_input_details = self.period_input_conditioner(noisy_season)
            cond_img = self.ts_to_img(cond_ts)
            cond_img = self._match_img_shape(cond_img, x_img).to(device=x_img.device, dtype=x_img.dtype)
            period_input_gate = (
                self._period_input_warmup_gate(x_img)
                * self._sigma_gate(sigma, x_img)
                * late_decay_gate
            )
            period_input_alpha = self._period_input_alpha(x_img)
            x_for_base = x_img + period_input_gate * period_input_alpha * torch.nan_to_num(cond_img)

        st_film = None
        feature_details = {"st_feature_enabled": x_img.new_tensor(0.0)}
        if self.feature_conditioner is not None:
            st_film, feature_details = self.feature_conditioner(noisy_ts.detach(), sigma)
            raw_st_film = st_film
            feature_gate = self._feature_warmup_gate(x_img) * self._feature_sigma_gate(sigma, x_img) * late_decay_gate
            st_film = self._limit_feature_norm(torch.nan_to_num(st_film * feature_gate.to(st_film.dtype)))
            feature_details["st_feature_enabled"] = x_img.new_tensor(1.0)
            feature_details["st_feature_gate"] = torch.nan_to_num(feature_gate).mean()
            feature_details["st_film_raw_norm"] = torch.nan_to_num(raw_st_film).square().mean().sqrt()
            feature_details["st_film_gated_norm"] = torch.nan_to_num(st_film).square().mean().sqrt()
        if self.transition_context_conditioner is not None:
            transition_film, transition_details = self.transition_context_conditioner(noisy_ts.detach(), sigma)
            raw_transition_film = transition_film
            transition_gate = self._feature_warmup_gate(x_img) * self._feature_sigma_gate(sigma, x_img) * late_decay_gate
            transition_film = torch.nan_to_num(transition_film * transition_gate.to(transition_film.dtype))
            st_film = transition_film if st_film is None else torch.nan_to_num(st_film + transition_film)
            st_film = self._limit_feature_norm(st_film)
            feature_details["transition_context_gate"] = torch.nan_to_num(transition_gate).mean()
            feature_details["transition_context_raw_norm"] = torch.nan_to_num(raw_transition_film).square().mean().sqrt()
            feature_details["transition_context_gated_norm"] = torch.nan_to_num(transition_film).square().mean().sqrt()
            for key, value in transition_details.items():
                feature_details[key] = value
        if self.tfda_film_conditioner is not None:
            tfda_film, tfda_details = self.tfda_film_conditioner(noisy_ts.detach(), sigma)
            raw_tfda_film = tfda_film
            tfda_gate = self._feature_warmup_gate(x_img) * self._feature_sigma_gate(sigma, x_img) * late_decay_gate
            tfda_film = torch.nan_to_num(tfda_film * tfda_gate.to(tfda_film.dtype))
            st_film = tfda_film if st_film is None else torch.nan_to_num(st_film + tfda_film)
            st_film = self._limit_feature_norm(st_film)
            feature_details["tfda_film_gate"] = torch.nan_to_num(tfda_gate).mean()
            feature_details["tfda_film_raw_norm"] = torch.nan_to_num(raw_tfda_film).square().mean().sqrt()
            feature_details["tfda_film_gated_norm"] = torch.nan_to_num(tfda_film).square().mean().sqrt()
            for key, value in tfda_details.items():
                feature_details[key] = value

        base_img_hat = self.base_net(x_for_base, sigma, class_labels=class_labels, st_film=st_film, **kwargs)
        base_ts_hat = self.img_to_ts(base_img_hat)
        base_ts_clean = base_ts_hat.detach()
        base_ts_for_st = base_ts_clean
        if self.training and self.st_input_noise > 0:
            base_ts_for_st = base_ts_clean + self.st_input_noise * torch.randn_like(base_ts_clean)
        residual_context = torch.nan_to_num(noisy_ts.detach() - base_ts_for_st)
        context_gate = base_ts_for_st.new_tensor(1.0)
        if self.use_context_sigma_gate:
            context_gate = self._sigma_gate(sigma, base_img_hat).reshape(base_ts_for_st.shape[0], 1, 1).to(
                device=base_ts_for_st.device,
                dtype=base_ts_for_st.dtype,
            )
        residual_context = residual_context * context_gate
        delta_ts, st_details = self.st_denoiser(
            base_ts_for_st, sigma, residual_context=residual_context,
            return_details=True
        )
        delta_img = self.ts_to_img(delta_ts)
        delta_img = self._match_img_shape(delta_img, base_img_hat).to(base_img_hat.device, base_img_hat.dtype)

        gate = (
            self._warmup_gate(base_img_hat)
            * self._sigma_gate(sigma, base_img_hat)
            * late_decay_gate.to(device=base_img_hat.device, dtype=base_img_hat.dtype)
        )
        alpha = self._effective_alpha(base_img_hat)
        trust_gate = self._trust_gate(base_img_hat)
        final_img_hat = base_img_hat + gate * alpha * trust_gate * torch.nan_to_num(delta_img)

        self.last_st_state = {
            "base_img_hat": base_img_hat.detach(),
            "delta_img": delta_img.detach(),
            "final_img_hat": final_img_hat.detach(),
            "base_ts_hat": base_ts_clean,
            "base_ts_hat_live": base_ts_hat,
            "noisy_ts": noisy_ts,
            "residual_context": residual_context,
            "context_gate": context_gate,
            "delta_ts": delta_ts,
            "delta_raw_ts": st_details["delta_raw_ts"],
            "trend_delta_ts": st_details.get("trend_delta_ts"),
            "season_delta_ts": st_details.get("season_delta_ts"),
            "confidence": st_details["confidence"],
            "relation_beta": st_details.get("relation_beta"),
            "relation_norm": st_details.get("relation_norm"),
            "period_details": st_details.get("period_details", {}),
            "mstc_token_details": st_details.get("mstc_token_details", {}),
            "tfda_token_details": st_details.get("tfda_token_details", {}),
            "period_input_details": period_input_details,
            "feature_details": feature_details,
            "period_input_gate": period_input_gate,
            "period_input_alpha": period_input_alpha,
            "gate": gate,
            "alpha": alpha,
            "trust_gate": trust_gate,
            "late_decay_gate": late_decay_gate,
        }
        return torch.nan_to_num(final_img_hat)
