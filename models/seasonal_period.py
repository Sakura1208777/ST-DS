import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_candidate_periods(length, periods=None, min_period=2, max_period=None, device=None):
    length = int(length)
    max_valid = max(min(length - 1, int(max_period or length - 1)), 1)
    min_valid = max(1, int(min_period))
    if periods is None:
        periods = list(range(min_valid, max_valid + 1))
    periods = sorted({int(p) for p in periods if min_valid <= int(p) <= max_valid})
    if not periods and length > 1:
        periods = [1]
    if not periods:
        periods = [0]
    return torch.tensor(periods, device=device, dtype=torch.long)


def _standardize(x, eps=1e-6):
    x = torch.nan_to_num(x)
    x = x - x.mean(dim=1, keepdim=True)
    scale = x.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    return torch.nan_to_num(x / scale)


def autocorr_repr(x_ts, max_lag=None, eps=1e-6):
    x_ts = _standardize(x_ts, eps=eps)
    length = x_ts.shape[1]
    if length <= 1:
        return x_ts.new_zeros(x_ts.shape[0], 1, x_ts.shape[2])
    fft = torch.fft.rfft(x_ts.float(), n=length, dim=1)
    power = fft.real.square() + fft.imag.square()
    corr = torch.fft.irfft(power, n=length, dim=1).to(x_ts.dtype)
    corr = corr / corr[:, :1, :].abs().clamp_min(eps)
    max_lag = min(int(max_lag or length - 1), length - 1)
    return torch.nan_to_num(corr[:, 1 : max_lag + 1, :])


class SoftPeriodAttention(nn.Module):
    def __init__(
        self,
        candidate_periods=None,
        min_period=2,
        max_period=None,
        temperature=0.35,
        eps=1e-6,
    ):
        super().__init__()
        self.candidate_periods = None if candidate_periods is None else [int(p) for p in candidate_periods]
        self.min_period = int(min_period)
        self.max_period = None if max_period is None else int(max_period)
        self.temperature = max(float(temperature), eps)
        self.eps = eps

    def forward(self, season_ts):
        season_ts = torch.nan_to_num(season_ts)
        batch, length, _ = season_ts.shape
        periods = _safe_candidate_periods(
            length,
            periods=self.candidate_periods,
            min_period=self.min_period,
            max_period=self.max_period,
            device=season_ts.device,
        )
        if length <= 1 or periods.numel() == 1 and periods[0].item() == 0:
            weights = season_ts.new_ones(batch, 1)
            return periods, weights, weights.new_zeros(())

        corr = autocorr_repr(season_ts, max_lag=int(periods.max().item()), eps=self.eps)
        lag_index = (periods - 1).clamp(0, corr.shape[1] - 1)
        scores = corr.index_select(dim=1, index=lag_index).mean(dim=-1)
        weights = torch.softmax(scores / self.temperature, dim=-1).to(dtype=season_ts.dtype)
        entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=-1).mean()
        entropy = entropy / math.log(max(weights.shape[-1], 2))
        return periods, torch.nan_to_num(weights), torch.nan_to_num(entropy)


class SeasonalBasisBank(nn.Module):
    def __init__(self, channels, max_scale=0.20, init_scale=0.02, zero_init=True):
        super().__init__()
        self.channels = int(channels)
        self.max_scale = max(float(max_scale), 1e-6)
        self.basis_proj = nn.Sequential(
            nn.Linear(2, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.basis_proj[-1].weight)
            nn.init.zeros_(self.basis_proj[-1].bias)
            nn.init.zeros_(self.gate_proj[-1].weight)
            nn.init.zeros_(self.gate_proj[-1].bias)

    def forward(self, tokens, periods, weights):
        batch, length, _ = tokens.shape
        if periods.numel() == 1 and periods[0].item() == 0:
            zero = tokens.new_tensor(0.0)
            return tokens, {"period_gate": zero, "period_strength": zero, "period_scale": zero}

        pos = torch.arange(length, device=tokens.device, dtype=tokens.dtype).view(1, length, 1)
        per = periods.to(device=tokens.device, dtype=tokens.dtype).view(1, 1, -1).clamp_min(1.0)
        phase = 2.0 * math.pi * pos / per
        sin = torch.sin(phase)
        cos = torch.cos(phase)
        weights = weights.to(device=tokens.device, dtype=tokens.dtype).view(batch, 1, -1)
        basis = torch.stack([(sin * weights).sum(dim=-1), (cos * weights).sum(dim=-1)], dim=-1)
        basis = self.basis_proj(basis)
        gate = torch.sigmoid(self.gate_proj(torch.cat([tokens, basis], dim=-1)))
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=tokens.device, dtype=tokens.dtype))
        out = tokens + scale * gate * basis
        details = {
            "period_gate": torch.nan_to_num(gate).mean(),
            "period_strength": torch.nan_to_num(basis).square().mean().sqrt(),
            "period_scale": torch.nan_to_num(scale).abs(),
        }
        return torch.nan_to_num(out), details


class SeasonalPeriodModulator(nn.Module):
    def __init__(
        self,
        channels,
        candidate_periods=None,
        min_period=2,
        max_period=None,
        temperature=0.35,
        max_scale=0.20,
        init_scale=0.02,
        zero_init=True,
    ):
        super().__init__()
        self.attention = SoftPeriodAttention(
            candidate_periods=candidate_periods,
            min_period=min_period,
            max_period=max_period,
            temperature=temperature,
        )
        self.basis_bank = SeasonalBasisBank(
            channels, max_scale=max_scale, init_scale=init_scale, zero_init=zero_init
        )

    def forward(self, season_ts, season_tokens):
        periods, weights, entropy = self.attention(season_ts)
        out, details = self.basis_bank(season_tokens, periods, weights)
        details["period_entropy"] = entropy
        details["period_mean"] = (weights * periods.to(weights.device, weights.dtype).view(1, -1)).sum(dim=-1).mean()
        details["period_weight_max"] = weights.max(dim=-1).values.mean()
        return out, details


class PeriodicInputConditioner(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        candidate_periods=None,
        min_period=2,
        max_period=None,
        temperature=0.45,
        max_scale=0.20,
        init_scale=0.02,
        zero_init=True,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.max_scale = max(float(max_scale), 1e-6)
        self.attention = SoftPeriodAttention(
            candidate_periods=candidate_periods,
            min_period=min_period,
            max_period=max_period,
            temperature=temperature,
        )
        self.in_proj = nn.Linear(self.input_channels + 2, hidden_channels)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(hidden_channels, self.input_channels)
        init = max(min(float(init_scale) / self.max_scale, 0.999), -0.999)
        self.scale_raw = nn.Parameter(torch.tensor(math.atanh(init), dtype=torch.float32))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _weighted_basis(length, periods, weights, dtype, device):
        if periods.numel() == 1 and periods[0].item() == 0:
            return torch.zeros(weights.shape[0], length, 2, device=device, dtype=dtype)
        pos = torch.arange(length, device=device, dtype=dtype).view(1, length, 1)
        per = periods.to(device=device, dtype=dtype).view(1, 1, -1).clamp_min(1.0)
        phase = 2.0 * math.pi * pos / per
        weights = weights.to(device=device, dtype=dtype).view(weights.shape[0], 1, -1)
        sin = (torch.sin(phase) * weights).sum(dim=-1)
        cos = (torch.cos(phase) * weights).sum(dim=-1)
        return torch.stack([sin, cos], dim=-1)

    def forward(self, season_ts):
        season_ts = torch.nan_to_num(season_ts)
        periods, weights, entropy = self.attention(season_ts)
        basis = self._weighted_basis(
            season_ts.shape[1],
            periods,
            weights,
            dtype=season_ts.dtype,
            device=season_ts.device,
        )
        h = self.in_proj(torch.cat([season_ts, basis], dim=-1))
        h = h + self.block(h)
        scale = self.max_scale * torch.tanh(self.scale_raw.to(device=season_ts.device, dtype=season_ts.dtype))
        cond_ts = scale * self.out_proj(h)
        details = {
            "period_input_entropy": entropy,
            "period_input_mean": (weights * periods.to(weights.device, weights.dtype).view(1, -1)).sum(dim=-1).mean(),
            "period_input_weight_max": weights.max(dim=-1).values.mean(),
            "period_input_scale": torch.nan_to_num(scale).abs(),
            "period_input_norm": torch.nan_to_num(cond_ts).square().mean().sqrt(),
        }
        return torch.nan_to_num(cond_ts), details


class PeriodConsistencyLoss(nn.Module):
    def __init__(self, args=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.kernels = getattr(args, "period_lma_kernels", None) or getattr(args, "ds_lma_kernels", [1, 2, 4, 6, 12])
        self.max_lag = getattr(args, "period_max_lag", None)
        self.lambda_autocorr = float(getattr(args, "lambda_period_autocorr", 0.010))
        self.lambda_amp = float(getattr(args, "lambda_period_amp", 0.006))
        self.lambda_phase = float(getattr(args, "lambda_period_phase", 0.002))
        self.use_sigma_weight = bool(getattr(args, "period_sigma_weight", True))
        self.sigma_mid = float(getattr(args, "period_sigma_mid", 0.0))
        self.sigma_scale = max(float(getattr(args, "period_sigma_scale", 2.0)), eps)

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

    def _season(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        valid = [int(k) for k in self.kernels if int(k) <= max(1, x_ts.shape[1])]
        if not valid:
            valid = [1]
        trend = torch.stack([self._moving_average(x_ts, k) for k in valid], dim=0).mean(dim=0)
        return torch.nan_to_num(x_ts - trend)

    def _spectrum(self, season):
        if season.shape[1] <= 1:
            z = season.new_zeros(season.shape[0], 1, season.shape[2])
            return z, z, z
        fft = torch.fft.rfft(season.float(), dim=1)
        amp = fft.abs().to(season.dtype)
        amp = amp / amp.mean(dim=1, keepdim=True).clamp_min(self.eps)
        unit_real = (fft.real / fft.abs().clamp_min(self.eps)).to(season.dtype)
        unit_imag = (fft.imag / fft.abs().clamp_min(self.eps)).to(season.dtype)
        return torch.nan_to_num(amp), torch.nan_to_num(unit_real), torch.nan_to_num(unit_imag)

    def _sigma_weight(self, sigma, ref):
        if sigma is None or not self.use_sigma_weight:
            return ref.new_tensor(1.0)
        sigma = torch.as_tensor(sigma, device=ref.device, dtype=ref.dtype).clamp_min(self.eps)
        log_sigma = sigma.log()
        weight = torch.sigmoid((self.sigma_mid - log_sigma) / self.sigma_scale).mean()
        return torch.nan_to_num(weight, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(self, pred_ts, real_ts, sigma=None):
        pred_ts = torch.nan_to_num(pred_ts)
        real_ts = torch.nan_to_num(real_ts.to(device=pred_ts.device, dtype=pred_ts.dtype))
        pred_season = self._season(pred_ts)
        real_season = self._season(real_ts)

        max_lag = self.max_lag or max(1, pred_ts.shape[1] // 2)
        autocorr_loss = F.l1_loss(
            autocorr_repr(pred_season, max_lag=max_lag, eps=self.eps),
            autocorr_repr(real_season, max_lag=max_lag, eps=self.eps),
        )

        pred_amp, pred_real, pred_imag = self._spectrum(pred_season)
        real_amp, real_real, real_imag = self._spectrum(real_season)
        amp_loss = F.l1_loss(pred_amp, real_amp)
        phase_weight = torch.minimum(pred_amp.detach(), real_amp.detach()).clamp(0.0, 5.0)
        phase_loss = (
            (phase_weight * (pred_real - real_real).abs()).mean()
            + (phase_weight * (pred_imag - real_imag).abs()).mean()
        )

        weight = self._sigma_weight(sigma, pred_ts)
        total = weight * (
            self.lambda_autocorr * autocorr_loss
            + self.lambda_amp * amp_loss
            + self.lambda_phase * phase_loss
        )
        total = torch.nan_to_num(total)
        logs = {
            "period/autocorr_loss": torch.nan_to_num(autocorr_loss).detach().item(),
            "period/amp_loss": torch.nan_to_num(amp_loss).detach().item(),
            "period/phase_loss": torch.nan_to_num(phase_loss).detach().item(),
            "period/weight": torch.nan_to_num(weight).detach().item(),
            "period/total_loss": total.detach().item(),
        }
        return total, logs
