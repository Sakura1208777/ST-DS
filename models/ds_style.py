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

    def stat_repr(self, x_ts):
        x_ts = torch.nan_to_num(x_ts)
        mean = x_ts.mean(dim=1)
        std = x_ts.var(dim=1, unbiased=False).add(self.eps).sqrt()
        return {
            "mean": torch.nan_to_num(mean),
            "std": torch.nan_to_num(std),
        }
