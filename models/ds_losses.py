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
        self.lambda_ds_ar_residual = float(getattr(args, "lambda_ds_ar_residual", 0.0) or 0.0)
        self.lambda_ds_multi_lag = float(getattr(args, "lambda_ds_multi_lag", 0.0) or 0.0)
        self.lambda_ds_ar_order = int(getattr(args, "ds_ar_order", 1))
        self.lambda_ds_coherence = float(getattr(args, "lambda_ds_coherence", 0.0) or 0.0)
        self.ds_multi_lag_lags = getattr(args, "ds_multi_lag_lags", None) or [1, 5, 10, 20, 50]
        self.ds_coherence_max_channels = int(getattr(args, "ds_coherence_max_channels", 64))
        self.ds_coherence_min_channels = int(getattr(args, "ds_coherence_min_channels", 0) or 0)
        self.use_long_loss_gate = bool(getattr(args, "use_ds_long_loss_gate", False))
        self.long_loss_length_mid = float(getattr(args, "ds_long_loss_length_mid", 96.0) or 96.0)
        self.long_loss_length_tau = max(float(getattr(args, "ds_long_loss_length_tau", 16.0) or 16.0), eps)
        self.long_loss_gate_floor = min(1.0, max(0.0, float(getattr(args, "ds_long_loss_gate_floor", 0.0) or 0.0)))
        self.use_sigma_weight = bool(getattr(args, "ds_sigma_weight", True))
        self.sigma_mid = float(getattr(args, "ds_sigma_mid", 0.0))
        self.sigma_scale = max(float(getattr(args, "ds_sigma_scale", 2.0)), eps)
        self.eps = eps

    def _long_loss_gate(self, ref):
        if not self.use_long_loss_gate:
            return ref.new_tensor(1.0)
        seq_len = int(ref.shape[1])
        raw = torch.sigmoid(ref.new_tensor((float(seq_len) - self.long_loss_length_mid) / self.long_loss_length_tau))
        return self.long_loss_gate_floor + (1.0 - self.long_loss_gate_floor) * raw

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

        ar_pred = self.extractor.ar_residual_repr(pred_ts, order=self.lambda_ds_ar_order)
        ar_real = self.extractor.ar_residual_repr(real_ts, order=self.lambda_ds_ar_order)
        if ar_pred is None or ar_real is None:
            ar_residual_loss = pred_ts.new_zeros(())
        else:
            ar_residual_loss = (
                F.l1_loss(ar_pred["mean"], ar_real["mean"])
                + F.l1_loss(ar_pred["std"], ar_real["std"])
                + F.l1_loss(ar_pred["acf1"], ar_real["acf1"])
            )

        # Multi-lag autocorrelation loss
        ml_pred = self.extractor.multi_lag_autocorr_repr(pred_ts, lags=self.ds_multi_lag_lags)
        ml_real = self.extractor.multi_lag_autocorr_repr(real_ts, lags=self.ds_multi_lag_lags)
        if ml_pred is None or ml_real is None:
            multi_lag_loss = pred_ts.new_zeros(())
        else:
            multi_lag_loss = sum(
                F.l1_loss(ml_pred[k], ml_real[k]) for k in ml_pred if k in ml_real
            ) / max(len(ml_pred), 1)

        # Spectral coherence loss (skip for low-dimensional data)
        n_channels = pred_ts.shape[2]
        if n_channels <= self.ds_coherence_min_channels:
            coherence_loss = pred_ts.new_zeros(())
        else:
            coh_pred = self.extractor.spectral_coherence_repr(pred_ts, max_channels=self.ds_coherence_max_channels)
            coh_real = self.extractor.spectral_coherence_repr(real_ts, max_channels=self.ds_coherence_max_channels)
            if coh_pred is None or coh_real is None:
                coherence_loss = pred_ts.new_zeros(())
            else:
                coherence_loss = F.l1_loss(coh_pred, coh_real)

        return {
            "trend": torch.nan_to_num(trend_loss),
            "season": torch.nan_to_num(season_loss),
            "freq": torch.nan_to_num(freq_loss),
            "corr": torch.nan_to_num(corr_loss),
            "dist": torch.nan_to_num(dist_loss),
            "ar_residual": torch.nan_to_num(ar_residual_loss),
            "multi_lag": torch.nan_to_num(multi_lag_loss),
            "coherence": torch.nan_to_num(coherence_loss),
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
        ar_residual_loss = losses["ar_residual"]
        multi_lag_loss = losses["multi_lag"]
        coherence_loss = losses["coherence"]

        w_trend, w_season = self._sigma_weights(sigma, pred_ts)
        long_loss_gate = self._long_loss_gate(pred_ts)
        total = (
            self.lambda_ds_trend * w_trend * trend_loss
            + self.lambda_ds_season * w_season * season_loss
            + self.lambda_ds_freq * freq_loss
            + self.lambda_ds_corr * corr_loss
            + self.lambda_ds_dist * dist_loss
            + long_loss_gate * (
                self.lambda_ds_ar_residual * ar_residual_loss
                + self.lambda_ds_multi_lag * multi_lag_loss
                + self.lambda_ds_coherence * coherence_loss
            )
        )
        total = torch.nan_to_num(total)

        logs = {
            "ds/trend_loss": torch.nan_to_num(trend_loss).detach().item(),
            "ds/season_loss": torch.nan_to_num(season_loss).detach().item(),
            "ds/freq_loss": torch.nan_to_num(freq_loss).detach().item(),
            "ds/corr_loss": torch.nan_to_num(corr_loss).detach().item(),
            "ds/dist_loss": torch.nan_to_num(dist_loss).detach().item(),
            "ds/ar_residual_loss": torch.nan_to_num(ar_residual_loss).detach().item(),
            "ds/multi_lag_loss": torch.nan_to_num(multi_lag_loss).detach().item(),
            "ds/coherence_loss": torch.nan_to_num(coherence_loss).detach().item(),
            "ds/long_loss_gate": torch.nan_to_num(long_loss_gate).detach().item(),
            "ds/total_loss": total.detach().item(),
            "ds/w_trend": w_trend.detach().item(),
            "ds/w_season": w_season.detach().item(),
        }
        return total, logs
