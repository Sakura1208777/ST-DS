import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ds_style import DSStyleExtractor


class DSStyleLoss(nn.Module):
    def __init__(self, args=None, eps=1e-6):
        super().__init__()
        self.extractor = DSStyleExtractor(args, eps=eps)
        self.lambda_ds_trend = float(getattr(args, "lambda_ds_trend", 0.05))
        self.lambda_ds_season = float(getattr(args, "lambda_ds_season", 0.05))
        self.lambda_ds_freq = float(getattr(args, "lambda_ds_freq", 0.02))
        self.lambda_ds_corr = float(getattr(args, "lambda_ds_corr", 0.01))
        self.lambda_ds_dist = float(getattr(args, "lambda_ds_dist", 0.01))
        self.use_sigma_weight = bool(getattr(args, "ds_sigma_weight", True))
        self.sigma_mid = float(getattr(args, "ds_sigma_mid", 0.0))
        self.sigma_scale = max(float(getattr(args, "ds_sigma_scale", 2.0)), eps)
        self.eps = eps

    def _sigma_weights(self, sigma, ref):
        if not self.use_sigma_weight or sigma is None:
            one = ref.new_tensor(1.0)
            return one, one

        sigma = torch.as_tensor(sigma, device=ref.device, dtype=ref.dtype).clamp_min(self.eps)
        log_sigma = sigma.log()
        w_trend = torch.sigmoid((log_sigma - self.sigma_mid) / self.sigma_scale).mean()
        w_trend = torch.nan_to_num(w_trend, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        w_season = 1.0 - w_trend
        return w_trend, w_season

    def component_losses(self, pred_ts, real_ts):
        pred_ts = torch.nan_to_num(pred_ts)
        real_ts = torch.nan_to_num(real_ts.to(device=pred_ts.device, dtype=pred_ts.dtype))

        trend_pred, season_pred = self.extractor.decompose(pred_ts)
        trend_real, season_real = self.extractor.decompose(real_ts)

        trend_loss = F.mse_loss(trend_pred, trend_real)
        season_loss = F.mse_loss(season_pred, season_real)
        freq_loss = F.l1_loss(self.extractor.freq_repr(pred_ts), self.extractor.freq_repr(real_ts))

        corr_pred = self.extractor.corr_repr(pred_ts)
        corr_real = self.extractor.corr_repr(real_ts)
        if corr_pred is None or corr_real is None:
            corr_loss = pred_ts.new_zeros(())
        else:
            corr_loss = F.l1_loss(corr_pred, corr_real)

        stat_pred = self.extractor.stat_repr(pred_ts)
        stat_real = self.extractor.stat_repr(real_ts)
        dist_loss = F.l1_loss(stat_pred["mean"], stat_real["mean"]) + F.l1_loss(stat_pred["std"], stat_real["std"])

        return {
            "trend": torch.nan_to_num(trend_loss),
            "season": torch.nan_to_num(season_loss),
            "freq": torch.nan_to_num(freq_loss),
            "corr": torch.nan_to_num(corr_loss),
            "dist": torch.nan_to_num(dist_loss),
        }

    def forward(self, pred_ts, real_ts, sigma=None):
        pred_ts = torch.nan_to_num(pred_ts)
        real_ts = torch.nan_to_num(real_ts.to(device=pred_ts.device, dtype=pred_ts.dtype))

        losses = self.component_losses(pred_ts, real_ts)
        trend_loss = losses["trend"]
        season_loss = losses["season"]
        freq_loss = losses["freq"]
        corr_loss = losses["corr"]
        dist_loss = losses["dist"]

        w_trend, w_season = self._sigma_weights(sigma, pred_ts)
        total = (
            self.lambda_ds_trend * w_trend * trend_loss
            + self.lambda_ds_season * w_season * season_loss
            + self.lambda_ds_freq * freq_loss
            + self.lambda_ds_corr * corr_loss
            + self.lambda_ds_dist * dist_loss
        )
        total = torch.nan_to_num(total)

        logs = {
            "ds/trend_loss": torch.nan_to_num(trend_loss).detach().item(),
            "ds/season_loss": torch.nan_to_num(season_loss).detach().item(),
            "ds/freq_loss": torch.nan_to_num(freq_loss).detach().item(),
            "ds/corr_loss": torch.nan_to_num(corr_loss).detach().item(),
            "ds/dist_loss": torch.nan_to_num(dist_loss).detach().item(),
            "ds/total_loss": total.detach().item(),
            "ds/w_trend": w_trend.detach().item(),
            "ds/w_season": w_season.detach().item(),
        }
        return total, logs
