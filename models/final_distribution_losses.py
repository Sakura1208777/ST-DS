import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalDistributionLoss(nn.Module):
    def __init__(self, args, eps=1e-6):
        super().__init__()
        self.eps = eps
        quantiles = getattr(args, "final_dist_quantiles", None)
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.lambda_mean = float(getattr(args, "lambda_final_mean", 0.005))
        self.lambda_std = float(getattr(args, "lambda_final_std", 0.010))
        self.lambda_diff_std = float(getattr(args, "lambda_final_diff_std", 0.020))
        self.lambda_quantile = float(getattr(args, "lambda_final_quantile", 0.020))
        self.lambda_highfreq = float(getattr(args, "lambda_final_highfreq", 0.010))
        self.use_sigma_weight = bool(getattr(args, "final_dist_sigma_weight", True))
        self.sigma_mid = float(getattr(args, "final_dist_sigma_mid", 0.0))
        self.sigma_scale = max(float(getattr(args, "final_dist_sigma_scale", 2.0)), eps)
        self.highfreq_start_ratio = min(
            0.95, max(0.0, float(getattr(args, "final_dist_highfreq_start_ratio", 0.50)))
        )

    def _sigma_weights(self, sigma, ref):
        if sigma is None or not self.use_sigma_weight:
            return ref.new_tensor(1.0), ref.new_tensor(1.0)
        sigma = torch.as_tensor(sigma, device=ref.device, dtype=ref.dtype).clamp_min(self.eps)
        low_noise = torch.sigmoid((self.sigma_mid - sigma.log()) / self.sigma_scale).mean()
        high_noise = 1.0 - low_noise
        return high_noise, low_noise

    @staticmethod
    def _diff_std(x, eps):
        if x.shape[1] <= 1:
            return x.new_tensor(0.0)
        return (x[:, 1:] - x[:, :-1]).std(dim=(0, 1), unbiased=False).clamp_min(eps)

    def _highfreq_amp(self, x):
        if x.shape[1] <= 1:
            return x.new_zeros(x.shape[0], 1, x.shape[2])
        freq = torch.fft.rfft(torch.nan_to_num(x), dim=1).abs()
        start = int(freq.shape[1] * self.highfreq_start_ratio)
        start = min(max(start, 1), freq.shape[1] - 1)
        return freq[:, start:, :]

    def forward(self, pred_ts, real_ts, sigma=None):
        pred_ts = torch.nan_to_num(pred_ts)
        real_ts = torch.nan_to_num(real_ts.to(device=pred_ts.device, dtype=pred_ts.dtype))
        trend_w, detail_w = self._sigma_weights(sigma, pred_ts)

        pred_mean = pred_ts.mean(dim=(0, 1))
        real_mean = real_ts.mean(dim=(0, 1))
        mean_loss = F.l1_loss(pred_mean, real_mean)

        pred_std = pred_ts.std(dim=(0, 1), unbiased=False).clamp_min(self.eps)
        real_std = real_ts.std(dim=(0, 1), unbiased=False).clamp_min(self.eps)
        std_loss = F.l1_loss(pred_std, real_std)

        diff_std_loss = F.l1_loss(self._diff_std(pred_ts, self.eps), self._diff_std(real_ts, self.eps))

        q = self.quantiles.to(device=pred_ts.device, dtype=pred_ts.dtype)
        pred_q = torch.quantile(pred_ts.flatten(0, 1), q, dim=0)
        real_q = torch.quantile(real_ts.flatten(0, 1), q, dim=0)
        quantile_loss = F.l1_loss(pred_q, real_q)

        highfreq_loss = F.l1_loss(self._highfreq_amp(pred_ts), self._highfreq_amp(real_ts))

        total = (
            self.lambda_mean * trend_w * mean_loss
            + self.lambda_std * trend_w * std_loss
            + self.lambda_diff_std * detail_w * diff_std_loss
            + self.lambda_quantile * quantile_loss
            + self.lambda_highfreq * detail_w * highfreq_loss
        )
        total = torch.nan_to_num(total)
        logs = {
            "final/mean_loss": torch.nan_to_num(mean_loss).detach().item(),
            "final/std_loss": torch.nan_to_num(std_loss).detach().item(),
            "final/diff_std_loss": torch.nan_to_num(diff_std_loss).detach().item(),
            "final/quantile_loss": torch.nan_to_num(quantile_loss).detach().item(),
            "final/highfreq_loss": torch.nan_to_num(highfreq_loss).detach().item(),
            "final/total_loss": total.detach().item(),
            "final/trend_weight": torch.nan_to_num(trend_w).detach().item(),
            "final/detail_weight": torch.nan_to_num(detail_w).detach().item(),
        }
        return total, logs
