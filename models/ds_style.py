import torch
import torch.nn as nn
import torch.nn.functional as F


class DSStyleExtractor(nn.Module):
    def __init__(self, args=None, eps=1e-6):
        super().__init__()
        kernels = getattr(args, "ds_lma_kernels", [1, 2, 4, 6, 12]) if args is not None else [1, 2, 4, 6, 12]
        if isinstance(kernels, str):
            kernels = kernels.strip("[]").split(",")
        self.kernels = [max(1, int(k)) for k in kernels]
        self.eps = eps

    def _moving_average(self, x_ts, kernel):
        batch, length, channels = x_ts.shape
        kernel = min(max(1, int(kernel)), max(1, length))
        if kernel <= 1 or length <= 1:
            return x_ts

        x = x_ts.permute(0, 2, 1)
        left = (kernel - 1) // 2
        right = kernel - 1 - left
        x = F.pad(x, (left, right), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=kernel, stride=1)
        return x.permute(0, 2, 1)

    def decompose(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        valid_kernels = [k for k in self.kernels if k <= max(1, x_ts.shape[1])]
        if not valid_kernels:
            valid_kernels = [1]

        trends = [self._moving_average(x_ts, k) for k in valid_kernels]
        trend = torch.stack(trends, dim=0).mean(dim=0)
        season = x_ts - trend
        return torch.nan_to_num(trend), torch.nan_to_num(season)

    def freq_repr(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        return torch.nan_to_num(torch.fft.rfft(x_ts, dim=1).abs())

    def corr_repr(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        batch, length, channels = x_ts.shape
        if channels <= 1:
            return None

        centered = x_ts - x_ts.mean(dim=1, keepdim=True)
        denom_len = max(length - 1, 1)
        cov = centered.transpose(1, 2).matmul(centered) / denom_len
        var = centered.pow(2).sum(dim=1) / denom_len
        std = var.add(self.eps).sqrt()
        denom = std.unsqueeze(2) * std.unsqueeze(1) + self.eps
        corr = cov / denom
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    def ar_residual_repr(self, x_ts, order=1):
        """Compute AR(order) residual statistics: mean, std, acf1 of residual."""
        x_ts = torch.nan_to_num(x_ts)
        batch, length, channels = x_ts.shape
        if length <= order + 1:
            return None

        # Build design matrix for AR(order): X = [x[t-1], x[t-2], ..., x[t-order]]
        # Target: y = x[t]
        max_order = min(order, length - 2)
        if max_order < 1:
            return None

        # OLS: phi = (X^T X)^{-1} X^T y, per channel
        y = x_ts[:, max_order:, :]  # (B, T-order, C)
        X_cols = []
        for lag in range(1, max_order + 1):
            X_cols.append(x_ts[:, max_order - lag:-lag, :])  # (B, T-order, C)
        X = torch.stack(X_cols, dim=-1)  # (B, T-order, C, order)

        # Solve per-channel OLS over the temporal axis:
        # X_bc: (B, C, T-order, order), y_bc: (B, C, T-order)
        X_bc = X.permute(0, 2, 1, 3)
        y_bc = y.permute(0, 2, 1)
        XtX = X_bc.transpose(-1, -2).matmul(X_bc)  # (B, C, order, order)
        Xty = X_bc.transpose(-1, -2).matmul(y_bc.unsqueeze(-1))  # (B, C, order, 1)
        try:
            phi = torch.linalg.solve(XtX + self.eps * torch.eye(max_order, device=XtX.device, dtype=XtX.dtype), Xty)
        except Exception:
            return None
        phi = phi.squeeze(-1)  # (B, C, order)

        # Residual: e[t] = y[t] - X[t] @ phi
        residual = y_bc - (X_bc * phi.unsqueeze(2)).sum(dim=-1)  # (B, C, T-order)

        # Return residual statistics
        res_mean = residual.mean(dim=2)  # (B, C)
        res_std = residual.var(dim=2, unbiased=False).add(self.eps).sqrt()  # (B, C)

        # Lag-1 autocorrelation of residual (should be ~0 for white noise)
        if residual.shape[2] > 2:
            res_centered = residual - residual.mean(dim=2, keepdim=True)
            res_var = res_centered.pow(2).sum(dim=2).clamp_min(self.eps)
            res_acf1 = (res_centered[:, :, 1:] * res_centered[:, :, :-1]).sum(dim=2) / res_var
        else:
            res_acf1 = torch.zeros_like(res_mean)

        return {
            "mean": torch.nan_to_num(res_mean),
            "std": torch.nan_to_num(res_std),
            "acf1": torch.nan_to_num(res_acf1).clamp(-1.0, 1.0),
        }

    def multi_lag_autocorr_repr(self, x_ts, lags=None):
        """Compute per-channel autocorrelation at multiple lags."""
        x_ts = torch.nan_to_num(x_ts)
        batch, length, channels = x_ts.shape
        if lags is None:
            lags = [1, 5, 10, 20, 50]
        valid_lags = [l for l in lags if l < length]
        if not valid_lags:
            return None

        centered = x_ts - x_ts.mean(dim=1, keepdim=True)
        var = centered.pow(2).sum(dim=1).clamp_min(self.eps)  # (B, C)

        result = {}
        for lag in valid_lags:
            acf = (centered[:, lag:, :] * centered[:, :-lag, :]).sum(dim=1) / var  # (B, C)
            result[f"lag{lag}"] = torch.nan_to_num(acf).clamp(-1.0, 1.0)
        return result

    def spectral_coherence_repr(self, x_ts, max_channels=64):
        """Compute spectral coherence between variable pairs (subsampled if needed)."""
        x_ts = torch.nan_to_num(x_ts)
        batch, length, channels = x_ts.shape
        if channels <= 1 or length <= 4:
            return None

        # Subsample channels if too many
        if channels > max_channels:
            idx = torch.linspace(0, channels - 1, max_channels, dtype=torch.long, device=x_ts.device)
            x_sub = x_ts[:, :, idx]
            C_sub = max_channels
        else:
            x_sub = x_ts
            C_sub = channels

        # Compute cross-spectral and power spectral densities
        X_fft = torch.fft.rfft(x_sub.float(), dim=1)  # (B, F, C_sub)
        power = X_fft.abs().square().clamp_min(self.eps)  # (B, F, C_sub)

        # Compute coherence for a subset of pairs (diagonal +/- 1 and random pairs)
        n_pairs = min(C_sub * 2, C_sub * (C_sub - 1) // 2)
        pairs = []
        for i in range(min(C_sub, 8)):
            for j in range(i + 1, min(C_sub, i + 4)):
                pairs.append((i, j))
        if len(pairs) > n_pairs:
            import random
            random.seed(42)
            pairs = random.sample(pairs, n_pairs)
        if not pairs:
            return None

        coherence_values = []
        for i, j in pairs:
            cross_spectrum = (X_fft[:, :, i] * X_fft[:, :, j].conj()).abs().square()
            denom = (power[:, :, i] * power[:, :, j]).clamp_min(self.eps)
            coh = cross_spectrum / denom  # (B, F)
            # Average over frequency
            coh_mean = coh.mean(dim=1)  # (B,)
            coherence_values.append(coh_mean)

        # Stack: (B, n_pairs)
        coherence = torch.stack(coherence_values, dim=-1)
        return torch.nan_to_num(coherence).clamp(0.0, 1.0)

    def stat_repr(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        mean = x_ts.mean(dim=1)
        std = x_ts.var(dim=1, unbiased=False).add(self.eps).sqrt()
        return {
            "mean": torch.nan_to_num(mean),
            "std": torch.nan_to_num(std),
        }
