import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from models.networks import EDMPrecond
from models.ema import LitEma
from models.img_transformations import STFTEmbedder, DelayEmbedder
from models.st_adapter import STEDMPrecondWrapper
from models.ds_losses import DSStyleLoss
from models.final_distribution_losses import FinalDistributionLoss
from models.seasonal_period import PeriodConsistencyLoss


class ImagenTime(nn.Module):
    def __init__(self, args, device):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.args = args
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.epoch = 0

        # delay embedding is used
        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len

            # NOTE: added this
            self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding)
        else:
            self.ts_img = STFTEmbedder(self.device, args.seq_len, args.n_fft, args.hop_length)

        base_net = EDMPrecond(args.img_resolution, args.input_channels, channel_mult=args.ch_mult,
                              model_channels=args.unet_channels, attn_resolutions=args.attn_resolution)
        if getattr(args, "use_st_adapter", False):
            self.net = STEDMPrecondWrapper(
                base_net=base_net,
                ts_to_img=self.ts_to_img,
                img_to_ts=self.img_to_ts,
                args=args,
            )
        else:
            self.net = base_net

        if getattr(args, "use_ds_train", False):
            self.ds_loss_fn = DSStyleLoss(args).to(device)
        else:
            self.ds_loss_fn = None
        if getattr(args, "use_final_dist_train", False):
            self.final_dist_loss_fn = FinalDistributionLoss(args).to(device)
        else:
            self.final_dist_loss_fn = None
        if getattr(args, "use_period_train", False):
            self.period_loss_fn = PeriodConsistencyLoss(args).to(device)
        else:
            self.period_loss_fn = None

        if args.ema:
            self.use_ema = True
            ema_d = float(getattr(args, 'ema_decay', 0.9999) or 0.9999)
            self.model_ema = LitEma(self.net, decay=ema_d, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

        ts_channels = args.input_channels if not args.use_stft else args.input_channels // 2
        self.temporal_transform = None
        # This block is intentionally opt-in. When it is enabled, it is used only
        # inside training losses; it is not part of the sampler's generation path.
        # Keeping it disabled by default prevents a loss-only module from absorbing
        # supervision without changing the generated samples.
        if bool(getattr(args, "use_loss_temporal_transform", False)) and int(getattr(args, "seq_len", 24)) > 48:
            d_model = min(64, max(8, ts_channels * 2))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=min(4, max(2, d_model // 16)),
                dim_feedforward=d_model * 2, dropout=0.0, batch_first=True,
            )
            self.temporal_transform = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.temporal_proj_in = nn.Linear(ts_channels, d_model)
            self.temporal_proj_out = nn.Linear(d_model, ts_channels)

    def _apply_temporal_transform(self, ts):
        if self.temporal_transform is None:
            return ts
        B, L, C = ts.shape
        x = self.temporal_proj_in(ts)
        x = self.temporal_transform(x)
        return ts + self.temporal_proj_out(x)

    def ts_to_img(self, signal, pad_val=None):
        """
        Args:
            signal: signal to convert to image
            pad_val: value to pad the image with, if delay embedding is used. Do not use for STFT embedding

        """
        # pad_val is used only for delay embedding, as the value to pad the image with
        # when creating the mask, we need to use 1 as padding value
        # if pad_val is given, it is used to overwrite the default value of 0
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val is not None else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    @staticmethod
    def _expand_to_batch_3d(value, batch_size, device, dtype):
        """Return value as shape [B, 1, 1] for safe ST gate/scale broadcasting."""
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(batch_size)
        elif value.numel() != batch_size:
            value = value[:1].expand(batch_size)
        return value.reshape(batch_size, 1, 1)

    def _late_decay_scale(self):
        if not bool(getattr(self.args, "use_late_decay", False)):
            return 1.0
        total_epochs = max(int(getattr(self.args, "epochs", 0) or 0), 0)
        if total_epochs <= 0:
            return 1.0
        start_epoch = int(getattr(self.args, "late_decay_start_epoch", 0) or 0)
        if start_epoch <= 0:
            start_ratio = float(getattr(self.args, "late_decay_start_ratio", 0.70))
            start_epoch = int(max(1.0, float(total_epochs) * start_ratio))
        current_epoch = float(getattr(self, "epoch", total_epochs))
        if current_epoch <= float(start_epoch):
            return 1.0
        denom = max(float(total_epochs - start_epoch), 1.0)
        progress = min(1.0, max(0.0, (current_epoch - float(start_epoch)) / denom))
        power = max(float(getattr(self.args, "late_decay_power", 1.0)), 1e-6)
        progress = progress ** power
        min_scale = min(1.0, max(0.0, float(getattr(self.args, "late_decay_min_scale", 1.0))))
        return 1.0 - progress * (1.0 - min_scale)

    @staticmethod
    def _moving_average_ts(x_ts, kernel):
        length = int(x_ts.shape[1])
        kernel = int(max(1, min(int(kernel), length)))
        if kernel <= 1 or length <= 1:
            return x_ts
        left = (kernel - 1) // 2
        right = kernel - 1 - left
        x_ch = x_ts.permute(0, 2, 1)
        x_pad = F.pad(x_ch, (left, right), mode="replicate")
        return F.avg_pool1d(x_pad, kernel_size=kernel, stride=1).permute(0, 2, 1)

    def _multi_scale_average_ts(self, x_ts, kernels):
        length = int(x_ts.shape[1])
        valid = []
        for kernel in kernels or [1]:
            kernel = int(kernel)
            if 1 <= kernel <= length:
                valid.append(kernel)
        if not valid:
            valid = [1]
        parts = [self._moving_average_ts(x_ts, kernel) for kernel in valid]
        result = torch.stack(parts, dim=0).mean(dim=0)
        if length > 48 and max(valid) / float(length) < 0.05:
            win = 48
            chunks = []
            for start in range(0, length, win):
                end = min(start + win, length)
                chunk = x_ts[:, start:end, :]
                chunk_parts = [self._moving_average_ts(chunk, min(k, end - start))
                               for k in valid]
                chunk_avg = torch.stack(chunk_parts, dim=0).mean(dim=0)
                chunks.append(chunk_avg)
            local_result = torch.cat(chunks, dim=1)
            result = (result + local_result) * 0.5
        return result

    def _structured_confidence(self, real_ts, real_trend, real_season, kernels):
        centered = real_ts - real_ts.mean(dim=1, keepdim=True)
        total_energy = centered.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        trend_energy = (
            real_trend - real_trend.mean(dim=1, keepdim=True)
        ).square().mean(dim=(1, 2), keepdim=True)
        short_kernels = [int(k) for k in (kernels or []) if int(k) > 1 and int(k) <= int(real_ts.shape[1])]
        short_kernel = min(short_kernels) if short_kernels else 2
        smooth_ts = self._moving_average_ts(real_ts, short_kernel)
        highfreq_energy = (real_ts - smooth_ts).square().mean(dim=(1, 2), keepdim=True)
        season_energy = real_season.square().mean(dim=(1, 2), keepdim=True)
        structured_energy = trend_energy + (season_energy - highfreq_energy).clamp_min(0.0)
        confidence = structured_energy / (structured_energy + highfreq_energy + 1e-8)
        power = max(float(getattr(self.args, "structured_confidence_power", 1.0)), 1e-6)
        confidence = confidence.clamp(0.0, 1.0).pow(power)
        min_conf = min(1.0, max(0.0, float(getattr(self.args, "structured_confidence_min", 0.0))))
        confidence = min_conf + (1.0 - min_conf) * confidence
        info = {
            "confidence": confidence.detach(),
            "trend_ratio": (trend_energy / total_energy).detach(),
            "highfreq_ratio": (highfreq_energy / total_energy).detach(),
        }
        return confidence.detach(), info

    def _structured_st_target(self, real_ts, base_ts):
        kernels = getattr(self.args, "structured_target_kernels", None)
        if kernels is None:
            kernels = getattr(self.args, "ds_lma_kernels", [1, 2, 4, 6, 12])

        real_trend = self._multi_scale_average_ts(real_ts, kernels)
        base_trend = self._multi_scale_average_ts(base_ts, kernels)
        trend_delta = real_trend - base_trend

        real_season = real_ts - real_trend
        base_season = base_ts - base_trend
        season_delta = self._multi_scale_average_ts(real_season - base_season, kernels)
        confidence, confidence_info = self._structured_confidence(real_ts, real_trend, real_season, kernels)

        trend_weight = float(getattr(self.args, "structured_trend_weight", 1.0))
        season_weight = float(getattr(self.args, "structured_season_weight", 0.45))
        total_epochs = max(int(getattr(self.args, "epochs", 0) or 0), 0)
        if total_epochs > 0:
            start_ratio = float(getattr(self.args, "structured_late_start_ratio", 0.50))
            start = max(1.0, float(total_epochs) * start_ratio)
            current_epoch = float(getattr(self, "epoch", total_epochs))
            if current_epoch > start:
                denom = max(float(total_epochs) - start, 1.0)
                progress = min(1.0, max(0.0, (current_epoch - start) / denom))
                min_season = min(season_weight, max(0.0, float(getattr(self.args, "structured_late_season_min", 0.20))))
                season_weight = season_weight - progress * (season_weight - min_season)

        target_delta = trend_weight * trend_delta + season_weight * season_delta
        norm_ratio = float(getattr(self.args, "structured_target_norm_ratio", 0.35))
        if bool(getattr(self.args, "structured_target_norm_switch", False)):
            final_ratio = float(getattr(self.args, "structured_target_norm_ratio_final", norm_ratio))
            max_blend = max(float(getattr(self.args, "structured_target_max_blend", 1.0)), 1e-6)
            blend_frac = min(1.0, max(0.0, self._structured_target_blend() / max_blend))
            norm_ratio = norm_ratio + (final_ratio - norm_ratio) * blend_frac
        if norm_ratio > 0:
            target_norm = target_delta.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            real_scale = real_ts.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            max_norm = norm_ratio * real_scale
            target_delta = target_delta * (max_norm / target_norm).clamp(max=1.0)

        info = {
            "trend_weight": trend_weight,
            "season_weight": season_weight,
            "target_norm": torch.nan_to_num(target_delta).square().mean().sqrt(),
            "confidence": confidence,
            "trend_ratio": confidence_info["trend_ratio"],
            "highfreq_ratio": confidence_info["highfreq_ratio"],
        }
        return torch.nan_to_num(target_delta), info

    def _residual_reliability(self, residual_ts):
        residual_ts = torch.nan_to_num(residual_ts)
        length = int(residual_ts.shape[1])
        centered = residual_ts - residual_ts.mean(dim=1, keepdim=True)
        total_energy = centered.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)

        kernels = getattr(self.args, "residual_reliability_kernels", None)
        if kernels is None:
            kernels = [3, 5, 7, 11]
        valid_kernels = [int(k) for k in kernels if 1 < int(k) <= length]
        if not valid_kernels:
            valid_kernels = [min(max(length, 1), 3)]

        trend = self._multi_scale_average_ts(centered, valid_kernels)
        season = centered - trend
        highfreq = centered - self._moving_average_ts(centered, min(valid_kernels))



        trend_ratio = trend.square().mean(dim=(1, 2), keepdim=True) / total_energy
        season_ratio = season.square().mean(dim=(1, 2), keepdim=True) / total_energy
        highfreq_ratio = highfreq.square().mean(dim=(1, 2), keepdim=True) / total_energy

        if length > 2:
            spectrum = torch.fft.rfft(centered.float(), dim=1)
            power = spectrum.abs().square().mean(dim=2)
            if power.shape[1] > 1:
                power = power[:, 1:]
            freq_total = power.sum(dim=1, keepdim=True).clamp_min(1e-8)
            topk_arg = getattr(self.args, "residual_reliability_freq_topk", 3)
            topk = int(topk_arg) if topk_arg is not None else 3
            if topk <= 0:
                topk = max(2, min(length // 15, 10))
            topk = max(1, topk)
            topk = min(topk, power.shape[1])
            freq_peak_ratio = power.topk(topk, dim=1).values.sum(dim=1, keepdim=True) / freq_total
            freq_peak_ratio = freq_peak_ratio.reshape(-1, 1, 1).to(dtype=residual_ts.dtype)
        else:
            freq_peak_ratio = torch.zeros_like(total_energy)

        acf_arg = getattr(self.args, "residual_reliability_acf_max_lag", 12)
        acf_max_lag = int(acf_arg) if acf_arg is not None else 12
        if acf_max_lag <= 0:
            acf_max_lag = max(6, min(length // 3, 48))
        max_lag = min(acf_max_lag, max(length - 1, 0))
        if max_lag >= 2:
            var = centered.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            acf_values = []
            for lag in range(2, max_lag + 1):
                corr = (centered[:, :-lag, :] * centered[:, lag:, :]).mean(dim=(1, 2), keepdim=True) / var
                acf_values.append(corr.abs())
            acf_peak_ratio = torch.stack(acf_values, dim=0).amax(dim=0)
        else:
            acf_peak_ratio = torch.zeros_like(total_energy)

        structure_score = (
            0.35 * trend_ratio.clamp_min(0.0)
            + 0.20 * season_ratio.clamp_min(0.0)
            + 0.30 * freq_peak_ratio.clamp_min(0.0)
            + 0.15 * acf_peak_ratio.clamp_min(0.0)
        )
        confidence = structure_score / (structure_score + highfreq_ratio.clamp_min(0.0) + 1e-8)
        power = max(float(getattr(self.args, "residual_reliability_power", 1.0)), 1e-6)
        confidence = confidence.clamp(0.0, 1.0).pow(power)

        info = {
            "confidence": confidence.detach(),
            "trend_ratio": trend_ratio.detach(),
            "season_ratio": season_ratio.detach(),
            "freq_peak_ratio": freq_peak_ratio.detach(),
            "acf_peak_ratio": acf_peak_ratio.detach(),
            "highfreq_ratio": highfreq_ratio.detach(),
        }
        return confidence.detach(), info

    def _predictive_structure_scale(self):
        if not bool(getattr(self.args, "use_pred_structure_loss", False)):
            return 0.0
        total_epochs = max(int(getattr(self.args, "epochs", 0) or 0), 1)
        start_ratio = max(0.0, float(getattr(self.args, "pred_structure_warmup_ratio", 0.25) or 0.0))
        window_ratio = max(0.0, float(getattr(self.args, "pred_structure_warmup_window", 0.05) or 0.0))
        start = float(total_epochs) * start_ratio
        window = max(1.0, float(total_epochs) * window_ratio)
        current_epoch = float(getattr(self, "epoch", total_epochs))
        if current_epoch <= start:
            return 0.0
        return min(1.0, max(0.0, (current_epoch - start) / window))

    @staticmethod
    def _lag_relation_features(x_ts, max_lag, include_cross=True, no_self=True, max_channels=64):
        x_ts = torch.nan_to_num(x_ts)
        batch, length, channels = x_ts.shape
        if length <= 1:
            return x_ts.new_zeros(batch, min(max(int(max_channels), 1), channels))
        if int(max_lag) <= 0:
            max_lag = max(6, min(length // 8, 24))
        max_lag = min(max(int(max_lag), 1), max(length - 1, 1))
        used_channels = min(max(int(max_channels), 1), channels)
        x_ts = x_ts[:, :, :used_channels]
        centered = x_ts - x_ts.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        normalized = centered / scale

        features = []
        for lag in range(1, max_lag + 1):
            left = normalized[:, :-lag, :]
            right = normalized[:, lag:, :]
            features.append((left * right).mean(dim=1))
            if include_cross and used_channels > 1:
                cross = torch.bmm(left.transpose(1, 2), right) / float(left.shape[1])
                if no_self:
                    eye = torch.eye(used_channels, device=cross.device, dtype=torch.bool).unsqueeze(0)
                    cross = cross.masked_fill(eye, 0.0)
                features.append(cross.reshape(batch, -1))
        return torch.cat(features, dim=1)

    def _predictive_structure_loss(self, pred_ts, target_ts):
        seq_len = int(pred_ts.shape[1])
        max_lag_arg = getattr(self.args, "pred_structure_max_lag", 6)
        max_lag = int(max_lag_arg) if max_lag_arg is not None else 6
        # --- [MAX1] Sequence-adaptive lag: cover ~25% of time steps, capped at 24 ---
        if bool(getattr(self.args, "pred_structure_auto_lag", False)):
            max_lag = max(6, min(seq_len // 4, 24))
            auto_lag = False
        elif max_lag <= 0:
            max_lag = max(4, min(seq_len // 6, 24))
            auto_lag = True
        else:
            auto_lag = False
        include_cross = bool(getattr(self.args, "pred_structure_include_cross", True))
        no_self = bool(getattr(self.args, "pred_structure_no_self", True))
        max_channels = int(getattr(self.args, "pred_structure_max_channels", 64) or 64)
        pred_features = self._lag_relation_features(
            pred_ts,
            max_lag=max_lag,
            include_cross=include_cross,
            no_self=no_self,
            max_channels=max_channels,
        )
        with torch.no_grad():
            target_features = self._lag_relation_features(
                target_ts,
                max_lag=max_lag,
                include_cross=include_cross,
                no_self=no_self,
                max_channels=max_channels,
            )
            strength = target_features.abs().mean(dim=1, keepdim=True)
            if bool(getattr(self.args, "pred_structure_adaptive", True)):
                floor = max(0.0, float(getattr(self.args, "pred_structure_strength_floor", 0.04) or 0.0))
                scale = max(1e-6, float(getattr(self.args, "pred_structure_strength_scale", 0.20) or 0.20))
                confidence = ((strength - floor) / scale).clamp(0.0, 1.0)
                min_conf = min(1.0, max(0.0, float(getattr(self.args, "pred_structure_confidence_min", 0.0) or 0.0)))
                power = max(1e-6, float(getattr(self.args, "pred_structure_confidence_power", 1.0) or 1.0))
                confidence = min_conf + (1.0 - min_conf) * confidence.pow(power)
            else:
                confidence = torch.ones_like(strength)
            if auto_lag:
                length_factor = min(1.0, 24.0 / max(float(seq_len), 24.0))
                confidence = confidence * length_factor
        beta = max(float(getattr(self.args, "pred_structure_huber_beta", 0.03) or 0.03), 1e-6)
        raw = F.smooth_l1_loss(pred_features, target_features, beta=beta, reduction="none").mean(dim=1, keepdim=True)
        loss = (raw * confidence).mean()
        info = {
            "confidence": confidence.detach(),
            "strength": strength.detach(),
            "raw_loss": raw.detach(),
            "feature_dim": pred_features.new_tensor(float(pred_features.shape[1])),
        }
        return torch.nan_to_num(loss), info

    def _transition_teacher_scale(self):
        if not bool(getattr(self.args, "use_transition_teacher", False)):
            return 0.0
        total_epochs = max(int(getattr(self.args, "epochs", 0) or 0), 1)
        start_ratio = max(0.0, float(getattr(self.args, "transition_teacher_warmup_ratio", 0.35) or 0.0))
        window_ratio = max(0.0, float(getattr(self.args, "transition_teacher_warmup_window", 0.10) or 0.0))
        start = float(total_epochs) * start_ratio
        window = max(1.0, float(total_epochs) * window_ratio)
        current_epoch = float(getattr(self, "epoch", total_epochs))
        if current_epoch <= start:
            return 0.0
        return min(1.0, max(0.0, (current_epoch - start) / window))

    def _transition_teacher_loss(self, pred_ts, target_ts):
        pred_ts = torch.nan_to_num(pred_ts)
        target_ts = torch.nan_to_num(target_ts).to(device=pred_ts.device, dtype=pred_ts.dtype)
        batch, seq_len, channels = pred_ts.shape
        horizons = getattr(self.args, "transition_teacher_horizons", None) or [1, 2, 4]
        horizons = sorted({int(h) for h in horizons if int(h) > 0 and int(h) < seq_len})
        if seq_len <= 1 or channels <= 0 or not horizons:
            zero = pred_ts.new_zeros(())
            return zero, {"horizons": 0, "channels": 0}

        max_channels = max(1, int(getattr(self.args, "transition_teacher_max_channels", 64) or 64))
        used_channels = min(max_channels, channels)
        pred_ts = pred_ts[:, :, :used_channels]
        target_ts = target_ts[:, :, :used_channels]

        with torch.no_grad():
            mean = target_ts.mean(dim=(0, 1), keepdim=True)
            scale = target_ts.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
            target_norm = (target_ts - mean) / scale
        pred_norm = (pred_ts - mean) / scale

        ridge = max(float(getattr(self.args, "transition_teacher_ridge", 0.01) or 0.01), 1e-8)
        losses = []
        used_horizons = []
        for horizon in horizons:
            real_past = target_norm[:, :-horizon, :].reshape(-1, used_channels)
            real_future = target_norm[:, horizon:, :].reshape(-1, used_channels)
            pred_past = pred_norm[:, :-horizon, :].reshape(-1, used_channels)
            pred_future = pred_norm[:, horizon:, :].reshape(-1, used_channels)
            if real_past.shape[0] <= used_channels + 1:
                continue

            ones_real = torch.ones(real_past.shape[0], 1, device=real_past.device, dtype=real_past.dtype)
            ones_pred = torch.ones(pred_past.shape[0], 1, device=pred_past.device, dtype=pred_past.dtype)
            real_design = torch.cat([real_past, ones_real], dim=1)
            pred_design = torch.cat([pred_past, ones_pred], dim=1)

            design = real_design.float()
            future = real_future.float()
            gram = design.transpose(0, 1).matmul(design) / max(float(design.shape[0]), 1.0)
            rhs = design.transpose(0, 1).matmul(future) / max(float(design.shape[0]), 1.0)
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            eye[-1, -1] = 0.0
            gram = gram + ridge * eye
            try:
                teacher = torch.linalg.solve(gram, rhs)
            except RuntimeError:
                teacher = torch.linalg.pinv(gram).matmul(rhs)
            teacher = teacher.to(device=pred_design.device, dtype=pred_design.dtype)
            pred_future_hat = pred_design.matmul(teacher)
            losses.append(F.smooth_l1_loss(pred_future_hat, pred_future, reduction="mean"))
            used_horizons.append(horizon)

        if not losses:
            zero = pred_ts.new_zeros(())
            return zero, {"horizons": 0, "channels": used_channels}
        loss = torch.stack(losses).mean()
        return torch.nan_to_num(loss), {"horizons": len(used_horizons), "channels": used_channels}

    def _structured_target_blend(self):
        if not bool(getattr(self.args, "use_structured_st_target", False)):
            return 0.0
        max_blend = min(1.0, max(0.0, float(getattr(self.args, "structured_target_max_blend", 1.0))))
        start = int(getattr(self.args, "structured_target_blend_start_epoch", 0) or 0)
        end = int(getattr(self.args, "structured_target_blend_end_epoch", 0) or 0)
        if start <= 0 and end <= 0:
            start_ratio = float(getattr(self.args, "structured_target_blend_start_ratio", 0.0) or 0.0)
            if start_ratio <= 0:
                return max_blend
            total = max(int(getattr(self.args, "epochs", 0) or 0), 1)
            start = max(1, int(float(total) * start_ratio))
            window_ratio = float(getattr(self.args, "structured_target_blend_window", 0.02) or 0.02)
            end = start + max(1, int(float(total) * window_ratio))
        current_epoch = float(getattr(self, "epoch", end if end > 0 else start))
        if current_epoch <= float(start):
            return 0.0
        if end <= start:
            return max_blend
        progress = min(1.0, max(0.0, (current_epoch - float(start)) / float(end - start)))
        return max_blend * progress

    # init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
    def init_stft_embedder(self, train_loader):
        """
        Args:
            train_loader: training data

        caches min and max values for the real and imaginary parts
        of the STFT transformation, which will be used for normalization.
        """
        assert type(self.ts_img) == STFTEmbedder, "You must use the STFTEmbedder to initialize the min and max values"
        data = []
        for i, data_batch in enumerate(train_loader):
            data.append(data_batch[0])
        self.ts_img.cache_min_max_params(torch.cat(data, dim=0))

    def loss_fn(self, x, x_ts=None):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        use_ds_train = getattr(self.args, "use_ds_train", False) and self.ds_loss_fn is not None and x_ts is not None
        use_period_train = (
            getattr(self.args, "use_period_train", False)
            and self.period_loss_fn is not None
            and x_ts is not None
        )
        use_st_residual = (
            getattr(self.args, "use_st_adapter", False)
            and getattr(self.args, "st_residual_calib", False)
            and x_ts is not None
            and hasattr(self.net, "pop_st_state")
        )
        use_final_dist = self.final_dist_loss_fn is not None and x_ts is not None
        use_ts_supervision = use_ds_train or use_st_residual or use_period_train or use_final_dist

        if use_ts_supervision:
            output, weight, sigma = self.forward(x, return_sigma=True)
        else:
            output, weight = self.forward(x)
            sigma = None
        st_state = self.net.pop_st_state() if hasattr(self.net, "pop_st_state") else None

        # denoising matching term
        # loss = weight * ((output - x) ** 2)
        image_loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = image_loss.detach().item()
        loss = image_loss

        if not use_ts_supervision:
            return image_loss, to_log

        warmup_epochs = max(int(getattr(self.args, "ds_warmup_epochs", 200)), 0)
        if warmup_epochs <= 0:
            warmup = 1.0
        else:
            warmup = min(1.0, max(0.0, float(getattr(self, "epoch", warmup_epochs)) / float(warmup_epochs)))
        style_scale = warmup
        late_decay_scale = self._late_decay_scale()
        if bool(getattr(self.args, "late_decay_style_loss", False)):
            style_scale *= late_decay_scale

        out_ts = None
        if use_ds_train:
            to_log['style scale'] = float(style_scale)
            to_log['late/decay_scale'] = float(late_decay_scale)
            out_ts = self.img_to_ts(output)
            out_ts = self._apply_temporal_transform(out_ts)
            style_ts = out_ts
            if (
                bool(getattr(self.args, "st_detach_base_for_style", False))
                and st_state is not None
                and st_state.get("base_ts_hat") is not None
                and st_state.get("delta_ts") is not None
                and st_state.get("gate") is not None
                and st_state.get("alpha") is not None
            ):
                base_ts_for_style = st_state["base_ts_hat"].detach()
                delta_ts_for_style = st_state["delta_ts"].to(
                    device=base_ts_for_style.device,
                    dtype=base_ts_for_style.dtype,
                )
                gate_for_style = self._expand_to_batch_3d(
                    st_state["gate"],
                    base_ts_for_style.shape[0],
                    base_ts_for_style.device,
                    base_ts_for_style.dtype,
                )
                alpha_for_style = self._expand_to_batch_3d(
                    st_state["alpha"],
                    base_ts_for_style.shape[0],
                    base_ts_for_style.device,
                    base_ts_for_style.dtype,
                )
                trust_for_style = self._expand_to_batch_3d(
                    st_state.get("trust_gate", base_ts_for_style.new_tensor(1.0)),
                    base_ts_for_style.shape[0],
                    base_ts_for_style.device,
                    base_ts_for_style.dtype,
                )
                style_ts = torch.nan_to_num(
                    base_ts_for_style + gate_for_style * alpha_for_style * trust_for_style * delta_ts_for_style
                )
                to_log['st/style_detach_base'] = 1.0
            else:
                to_log['st/style_detach_base'] = 0.0
            x_ts_for_loss = x_ts.to(device=style_ts.device, dtype=style_ts.dtype)
            ts_loss = F.mse_loss(torch.nan_to_num(style_ts), torch.nan_to_num(x_ts_for_loss))
            ds_loss, ds_logs = self.ds_loss_fn(style_ts, x_ts_for_loss, sigma)
            lambda_ts = float(getattr(self.args, "lambda_ts", 0.10))
            loss = loss + output.new_tensor(style_scale) * (lambda_ts * ts_loss + ds_loss)

            to_log['ts loss'] = torch.nan_to_num(ts_loss).detach().item()
            to_log.update(ds_logs)

        if (
            use_st_residual
            and bool(getattr(self.args, "st_branch_style_calib", False))
            and self.ds_loss_fn is not None
            and st_state is not None
            and st_state.get("delta_ts") is not None
            and x_ts is not None
        ):
            base_ts_hat_for_branch = st_state["base_ts_hat"]
            delta_ts_for_branch = st_state["delta_ts"].to(
                device=base_ts_hat_for_branch.device,
                dtype=base_ts_hat_for_branch.dtype,
            )
            branch_ts = torch.nan_to_num(base_ts_hat_for_branch + delta_ts_for_branch)
            real_ts_for_branch = x_ts.to(device=branch_ts.device, dtype=branch_ts.dtype)
            branch_losses = self.ds_loss_fn.component_losses(branch_ts, real_ts_for_branch)
            branch_style_loss = (
                float(getattr(self.args, "lambda_st_branch_trend", 0.010)) * branch_losses["trend"]
                + float(getattr(self.args, "lambda_st_branch_season", 0.010)) * branch_losses["season"]
                + float(getattr(self.args, "lambda_st_branch_freq", 0.005)) * branch_losses["freq"]
                + float(getattr(self.args, "lambda_st_branch_corr", 0.002)) * branch_losses["corr"]
                + float(getattr(self.args, "lambda_st_branch_dist", 0.002)) * branch_losses["dist"]
            )
            loss = loss + output.new_tensor(style_scale) * torch.nan_to_num(branch_style_loss)
            to_log['st/branch_style_loss'] = torch.nan_to_num(branch_style_loss).detach().item()
            to_log['st/branch_trend_loss'] = torch.nan_to_num(branch_losses["trend"]).detach().item()
            to_log['st/branch_season_loss'] = torch.nan_to_num(branch_losses["season"]).detach().item()
            to_log['st/branch_freq_loss'] = torch.nan_to_num(branch_losses["freq"]).detach().item()
            to_log['st/branch_corr_loss'] = torch.nan_to_num(branch_losses["corr"]).detach().item()
            to_log['st/branch_dist_loss'] = torch.nan_to_num(branch_losses["dist"]).detach().item()

        if self.final_dist_loss_fn is not None and x_ts is not None:
            if out_ts is None:
                out_ts = self.img_to_ts(output)
                out_ts = self._apply_temporal_transform(out_ts)
            x_ts_for_final = x_ts.to(device=out_ts.device, dtype=out_ts.dtype)
            final_dist_loss, final_dist_logs = self.final_dist_loss_fn(out_ts, x_ts_for_final, sigma)
            loss = loss + output.new_tensor(style_scale) * final_dist_loss
            to_log.update(final_dist_logs)

        if use_period_train:
            if out_ts is None:
                out_ts = self.img_to_ts(output)
                out_ts = self._apply_temporal_transform(out_ts)
            period_warmup_epochs = max(
                int(getattr(self.args, "period_warmup_epochs", getattr(self.args, "ds_warmup_epochs", 200))),
                0,
            )
            if period_warmup_epochs <= 0:
                period_warmup = 1.0
            else:
                period_warmup = min(
                    1.0,
                    max(0.0, float(getattr(self, "epoch", period_warmup_epochs)) / float(period_warmup_epochs)),
                )
            period_late_scale = 1.0
            if bool(getattr(self.args, "period_late_stabilize", False)):
                total_epochs = max(int(getattr(self.args, "epochs", 0) or 0), 0)
                if total_epochs > 0:
                    start_ratio = float(getattr(self.args, "period_late_start_ratio", 0.80))
                    min_scale = min(1.0, max(0.0, float(getattr(self.args, "period_late_min_scale", 0.55))))
                    start = max(1.0, float(total_epochs) * start_ratio)
                    current_epoch = float(getattr(self, "epoch", total_epochs))
                    if current_epoch > start:
                        denom = max(float(total_epochs) - start, 1.0)
                        progress = min(1.0, max(0.0, (current_epoch - start) / denom))
                        period_late_scale = 1.0 - progress * (1.0 - min_scale)
            period_scale = period_warmup * period_late_scale
            x_ts_for_period = x_ts.to(device=out_ts.device, dtype=out_ts.dtype)
            period_loss, period_logs = self.period_loss_fn(out_ts, x_ts_for_period, sigma)
            loss = loss + output.new_tensor(period_scale) * period_loss
            to_log["period/scale"] = float(period_scale)
            to_log.update(period_logs)

        if use_st_residual and st_state is not None and st_state.get("delta_ts") is not None:
            st_warmup_epochs = max(
                int(getattr(self.args, "st_residual_warmup_epochs", getattr(self.args, "st_warmup_epochs", 200))),
                0,
            )
            if st_warmup_epochs <= 0:
                st_warmup = 1.0
            else:
                st_warmup = min(1.0, max(0.0, float(getattr(self, "epoch", st_warmup_epochs)) / float(st_warmup_epochs)))
            st_calib_scale = st_warmup
            if bool(getattr(self.args, "late_decay_style_loss", False)):
                st_calib_scale *= late_decay_scale
            base_ts_hat = st_state["base_ts_hat"]
            delta_ts = st_state["delta_ts"]
            real_ts_for_st = x_ts.to(device=base_ts_hat.device, dtype=base_ts_hat.dtype)
            full_target_delta = torch.nan_to_num(real_ts_for_st - base_ts_hat.detach())
            if bool(getattr(self.args, "use_structured_st_target", False)):
                structured_delta, structured_info = self._structured_st_target(real_ts_for_st, base_ts_hat.detach())
                structured_blend = self._structured_target_blend()
                if bool(getattr(self.args, "structured_adaptive_blend", False)):
                    confidence = structured_info["confidence"].to(device=full_target_delta.device, dtype=full_target_delta.dtype)
                    blend = structured_blend * confidence
                    effective_blend = torch.nan_to_num(blend).detach().mean().item()
                else:
                    blend = structured_blend
                    effective_blend = float(structured_blend)
                target_delta = (1.0 - blend) * full_target_delta + blend * structured_delta
                to_log['st/structured_target'] = 1.0
                to_log['st/structured_blend'] = float(structured_blend)
                to_log['st/structured_effective_blend'] = float(effective_blend)
                to_log['st/structured_confidence'] = torch.nan_to_num(
                    structured_info["confidence"]
                ).detach().mean().item()
                to_log['st/structured_trend_ratio'] = torch.nan_to_num(
                    structured_info["trend_ratio"]
                ).detach().mean().item()
                to_log['st/structured_highfreq_ratio'] = torch.nan_to_num(
                    structured_info["highfreq_ratio"]
                ).detach().mean().item()
                to_log['st/structured_trend_weight'] = float(structured_info["trend_weight"])
                to_log['st/structured_season_weight'] = float(structured_info["season_weight"])
                to_log['st/structured_target_norm'] = torch.nan_to_num(structured_info["target_norm"]).detach().item()
            else:
                target_delta = full_target_delta
                to_log['st/structured_target'] = 0.0
                to_log['st/structured_blend'] = 0.0
                to_log['st/structured_effective_blend'] = 0.0
            target_scale = max(0.0, float(getattr(self.args, "st_residual_target_scale", 0.25)))
            target_delta = target_delta * target_scale
            use_residual_reliability = bool(getattr(self.args, "use_residual_reliability", False))
            if use_residual_reliability:
                reliability_min = min(1.0, max(0.0, float(getattr(self.args, "residual_reliability_min", 0.20))))
                reliability_result = self._residual_reliability(full_target_delta)
                reliability = reliability_result[0].to(device=target_delta.device, dtype=target_delta.dtype)
                reliability_info = reliability_result[1]
                residual_loss_weight = reliability_min + (1.0 - reliability_min) * reliability
                delta_reg_boost_max = max(1.0, float(getattr(self.args, "reliability_delta_reg_boost", 1.0)))
                effective_boost_max = max(1.0, float(getattr(self.args, "reliability_effective_boost", 1.0)))
                delta_reg_weight = 1.0 + (delta_reg_boost_max - 1.0) * (1.0 - reliability)
                effective_reg_weight = 1.0 + (effective_boost_max - 1.0) * (1.0 - reliability)
            else:
                reliability = None
                residual_loss_weight = None
                delta_reg_weight = None
                effective_reg_weight = None
            gate_ts = self._expand_to_batch_3d(
                st_state.get("gate", base_ts_hat.new_tensor(1.0)),
                base_ts_hat.shape[0],
                base_ts_hat.device,
                base_ts_hat.dtype,
            )
            alpha_ts = self._expand_to_batch_3d(
                st_state.get("alpha", base_ts_hat.new_tensor(1.0)),
                base_ts_hat.shape[0],
                base_ts_hat.device,
                base_ts_hat.dtype,
            )
            trust_ts = self._expand_to_batch_3d(
                st_state.get("trust_gate", base_ts_hat.new_tensor(1.0)),
                base_ts_hat.shape[0],
                base_ts_hat.device,
                base_ts_hat.dtype,
            )
            effective_delta_ts = torch.nan_to_num(gate_ts * alpha_ts * trust_ts * delta_ts)
            residual_pred_for_loss = effective_delta_ts
            target_for_residual_loss = target_delta
            huber_beta = max(float(getattr(self.args, "st_residual_huber_beta", 0.05)), 1e-6)
            st_residual_loss_raw = F.smooth_l1_loss(
                residual_pred_for_loss,
                target_for_residual_loss,
                beta=huber_beta,
                reduction="none",
            )
            if residual_loss_weight is not None:
                st_residual_loss = (st_residual_loss_raw * residual_loss_weight).mean()
                st_delta_reg = (effective_delta_ts.square() * delta_reg_weight).mean()
                st_raw_delta_reg = (torch.nan_to_num(delta_ts).square() * delta_reg_weight).mean()
            else:
                st_residual_loss = st_residual_loss_raw.mean()
                st_delta_reg = effective_delta_ts.square().mean()
                st_raw_delta_reg = torch.nan_to_num(delta_ts).square().mean()
            lambda_st_residual = float(getattr(self.args, "lambda_st_residual", 0.05))
            lambda_st_delta_reg = float(getattr(self.args, "lambda_st_delta_reg", 0.001))
            lambda_st_raw_delta_reg = float(getattr(self.args, "lambda_st_raw_delta_reg", 0.0005))
            loss = loss + output.new_tensor(st_calib_scale) * (
                lambda_st_residual * st_residual_loss
                + lambda_st_delta_reg * st_delta_reg
                + lambda_st_raw_delta_reg * st_raw_delta_reg
            )
            # --- [MAX1] Delta smoothness regularization ---
            if bool(getattr(self.args, "use_delta_smooth_reg", False)):
                if effective_delta_ts.shape[1] > 1:
                    delta_diff = effective_delta_ts[:, 1:, :] - effective_delta_ts[:, :-1, :]
                    delta_smooth_loss = delta_diff.square().mean()
                else:
                    delta_smooth_loss = effective_delta_ts.new_tensor(0.0)
                lambda_delta_smooth = float(getattr(self.args, "lambda_delta_smooth", 0.005))
                loss = loss + output.new_tensor(st_calib_scale) * (lambda_delta_smooth * delta_smooth_loss)
                to_log['st/delta_smooth_loss'] = torch.nan_to_num(delta_smooth_loss).detach().item()
            # --- [MAX1] Delta spectral alignment ---
            if bool(getattr(self.args, "use_delta_spectral_reg", False)):
                if effective_delta_ts.shape[1] > 3:
                    delta_fft = torch.fft.rfft(effective_delta_ts, dim=1)
                    target_fft = torch.fft.rfft(target_delta, dim=1)
                    spectral_loss = (delta_fft.abs() - target_fft.abs()).square().mean()
                else:
                    spectral_loss = effective_delta_ts.new_tensor(0.0)
                lambda_delta_spectral = float(getattr(self.args, "lambda_delta_spectral", 0.003))
                loss = loss + output.new_tensor(st_calib_scale) * (lambda_delta_spectral * spectral_loss)
                to_log['st/delta_spectral_loss'] = torch.nan_to_num(spectral_loss).detach().item()
            if bool(getattr(self.args, "use_pred_structure_loss", False)):
                pred_structure_scale = self._predictive_structure_scale()
                pred_structure_loss = output.new_tensor(0.0)
                pred_structure_info = None
                if pred_structure_scale > 0.0:
                    pred_structure_ts = torch.nan_to_num(base_ts_hat.detach() + effective_delta_ts)
                    target_structure_ts = real_ts_for_st.to(
                        device=pred_structure_ts.device,
                        dtype=pred_structure_ts.dtype,
                    )
                    pred_structure_loss, pred_structure_info = self._predictive_structure_loss(
                        pred_structure_ts,
                        target_structure_ts,
                    )
                    lambda_pred_structure = float(getattr(self.args, "lambda_pred_structure", 0.003))
                    seq_len_val = int(target_structure_ts.shape[1])
                    if seq_len_val > 48:
                        lambda_pred_structure *= min(1.0, (48.0 / float(seq_len_val)) ** 0.5)
                    loss = loss + output.new_tensor(st_calib_scale * pred_structure_scale) * (
                        lambda_pred_structure * pred_structure_loss
                    )
                to_log['pred_structure/scale'] = float(pred_structure_scale)
                to_log['pred_structure/loss'] = torch.nan_to_num(pred_structure_loss).detach().item()
                if pred_structure_info is not None:
                    to_log['pred_structure/confidence'] = torch.nan_to_num(
                        pred_structure_info["confidence"]
                    ).detach().mean().item()
                    to_log['pred_structure/strength'] = torch.nan_to_num(
                        pred_structure_info["strength"]
                    ).detach().mean().item()
                    to_log['pred_structure/raw_loss'] = torch.nan_to_num(
                        pred_structure_info["raw_loss"]
                    ).detach().mean().item()
                    to_log['pred_structure/feature_dim'] = torch.nan_to_num(
                        pred_structure_info["feature_dim"]
                    ).detach().item()
            if bool(getattr(self.args, "use_transition_teacher", False)):
                teacher_scale = self._transition_teacher_scale()
                teacher_loss = output.new_tensor(0.0)
                teacher_info = None
                if teacher_scale > 0.0 and st_calib_scale > 0.0:
                    pred_teacher_ts = torch.nan_to_num(base_ts_hat.detach() + effective_delta_ts)
                    target_teacher_ts = real_ts_for_st.to(
                        device=pred_teacher_ts.device,
                        dtype=pred_teacher_ts.dtype,
                    )
                    teacher_loss, teacher_info = self._transition_teacher_loss(
                        pred_teacher_ts,
                        target_teacher_ts,
                    )
                    lambda_transition_teacher = float(getattr(self.args, "lambda_transition_teacher", 0.0005) or 0.0005)
                    loss = loss + output.new_tensor(st_calib_scale * teacher_scale) * (
                        lambda_transition_teacher * teacher_loss
                    )
                to_log['transition_teacher/scale'] = float(teacher_scale)
                to_log['transition_teacher/loss'] = torch.nan_to_num(teacher_loss).detach().item()
                if teacher_info is not None:
                    to_log['transition_teacher/horizons'] = float(teacher_info["horizons"])
                    to_log['transition_teacher/channels'] = float(teacher_info["channels"])
                else:
                    to_log['transition_teacher/horizons'] = 0.0
                    to_log['transition_teacher/channels'] = 0.0
            if bool(getattr(self.args, "st_effective_align", False)):
                if out_ts is None:
                    out_ts = self.img_to_ts(output)
                base_ts_live = st_state.get("base_ts_hat_live", base_ts_hat).to(device=out_ts.device, dtype=out_ts.dtype).detach()
                effective_delta = torch.nan_to_num(out_ts - base_ts_live)
                effective_target = target_delta.to(device=effective_delta.device, dtype=effective_delta.dtype)
                effective_beta = max(float(getattr(self.args, "st_effective_huber_beta", 0.05)), 1e-6)
                effective_loss = F.smooth_l1_loss(effective_delta, effective_target, beta=effective_beta)

                target_norm = effective_target.square().mean(dim=(1, 2)).sqrt().clamp_min(1e-6)
                effective_norm = effective_delta.square().mean(dim=(1, 2)).sqrt()
                effective_ratio_per_sample = effective_norm / target_norm
                max_ratio = max(float(getattr(self.args, "st_effective_max_ratio", 0.35)), 0.0)
                ratio_penalty_per_sample = F.relu(effective_ratio_per_sample - max_ratio).square()
                if effective_reg_weight is not None:
                    effective_reg_per_sample = effective_reg_weight.reshape(-1)
                    ratio_penalty = (ratio_penalty_per_sample * effective_reg_per_sample).mean()
                else:
                    ratio_penalty = ratio_penalty_per_sample.mean()
                lambda_st_effective = float(getattr(self.args, "lambda_st_effective", 0.05))
                lambda_st_effective_ratio = float(getattr(self.args, "lambda_st_effective_ratio", 0.01))
                loss = loss + output.new_tensor(st_calib_scale) * (
                    lambda_st_effective * effective_loss + lambda_st_effective_ratio * ratio_penalty
                )

                base_ts_mse = F.mse_loss(torch.nan_to_num(base_ts_hat.detach()), real_ts_for_st)
                final_ts_for_log = out_ts.to(device=real_ts_for_st.device, dtype=real_ts_for_st.dtype)
                final_ts_mse = F.mse_loss(torch.nan_to_num(final_ts_for_log), real_ts_for_st)
                to_log['st/effective_residual_loss'] = torch.nan_to_num(effective_loss).detach().item()
                to_log['st/effective_ratio'] = torch.nan_to_num(effective_ratio_per_sample).detach().mean().item()
                to_log['st/effective_ratio_penalty'] = torch.nan_to_num(ratio_penalty).detach().item()
                to_log['st/base_ts_mse'] = torch.nan_to_num(base_ts_mse).detach().item()
                to_log['st/final_ts_mse'] = torch.nan_to_num(final_ts_mse).detach().item()

            to_log['st/calib_scale'] = float(st_calib_scale)
            to_log['st/residual_target_scale'] = float(target_scale)
            to_log['st/residual_loss'] = torch.nan_to_num(st_residual_loss).detach().item()
            to_log['st/delta_reg'] = torch.nan_to_num(st_delta_reg).detach().item()
            to_log['st/raw_delta_reg'] = torch.nan_to_num(st_raw_delta_reg).detach().item()
            to_log['st/effective_delta_norm'] = torch.nan_to_num(effective_delta_ts).detach().square().mean().sqrt().item()
            if use_residual_reliability:
                to_log['st/reliability'] = torch.nan_to_num(reliability).detach().mean().item()
                to_log['st/reliability_loss_weight'] = torch.nan_to_num(residual_loss_weight).detach().mean().item()
                to_log['st/reliability_reg_weight'] = torch.nan_to_num(delta_reg_weight).detach().mean().item()
                to_log['st/reliability_trend_ratio'] = torch.nan_to_num(
                    reliability_info["trend_ratio"]
                ).detach().mean().item()
                to_log['st/reliability_season_ratio'] = torch.nan_to_num(
                    reliability_info["season_ratio"]
                ).detach().mean().item()
                to_log['st/reliability_freq_peak_ratio'] = torch.nan_to_num(
                    reliability_info["freq_peak_ratio"]
                ).detach().mean().item()
                to_log['st/reliability_acf_peak_ratio'] = torch.nan_to_num(
                    reliability_info["acf_peak_ratio"]
                ).detach().mean().item()
                to_log['st/reliability_highfreq_ratio'] = torch.nan_to_num(
                    reliability_info["highfreq_ratio"]
                ).detach().mean().item()
            trend_delta = st_state.get("trend_delta_ts")
            if trend_delta is not None:
                to_log['st/trend_delta_norm'] = torch.nan_to_num(trend_delta).detach().square().mean().sqrt().item()
            season_delta = st_state.get("season_delta_ts")
            if season_delta is not None:
                to_log['st/season_delta_norm'] = torch.nan_to_num(season_delta).detach().square().mean().sqrt().item()
            trend_scale = st_state.get("trend_scale")
            if trend_scale is not None:
                to_log['st/trend_scale'] = torch.nan_to_num(trend_scale).detach().mean().item()
            season_scale = st_state.get("season_scale")
            if season_scale is not None:
                to_log['st/season_scale'] = torch.nan_to_num(season_scale).detach().mean().item()
            confidence = st_state.get("confidence")
            if confidence is not None:
                to_log['st/confidence'] = torch.nan_to_num(confidence).detach().mean().item()
            context_gate = st_state.get("context_gate")
            if context_gate is not None:
                to_log['st/context_gate'] = torch.nan_to_num(context_gate).detach().mean().item()
            late_decay_gate = st_state.get("late_decay_gate")
            if late_decay_gate is not None:
                to_log['st/late_decay_gate'] = torch.nan_to_num(late_decay_gate).detach().mean().item()
            relation_beta = st_state.get("relation_beta")
            if relation_beta is not None:
                relation_reg = torch.nan_to_num(relation_beta).square()
                lambda_st_relation_reg = float(getattr(self.args, "lambda_st_relation_reg", 0.001))
                loss = loss + output.new_tensor(st_calib_scale) * lambda_st_relation_reg * relation_reg
                to_log['st/relation_reg'] = torch.nan_to_num(relation_reg).detach().item()
                to_log['st/relation_beta'] = torch.nan_to_num(relation_beta).detach().item()
            relation_norm = st_state.get("relation_norm")
            if relation_norm is not None:
                to_log['st/relation_norm'] = torch.nan_to_num(relation_norm).detach().item()
            period_details = st_state.get("period_details") or {}
            for key, value in period_details.items():
                if torch.is_tensor(value):
                    to_log[f"st/{key}"] = torch.nan_to_num(value).detach().item()
                else:
                    to_log[f"st/{key}"] = float(value)
            period_input_details = st_state.get("period_input_details") or {}
            for key, value in period_input_details.items():
                if torch.is_tensor(value):
                    to_log[f"st/{key}"] = torch.nan_to_num(value).detach().item()
                else:
                    to_log[f"st/{key}"] = float(value)
            feature_details = st_state.get("feature_details") or {}
            for key, value in feature_details.items():
                if torch.is_tensor(value):
                    to_log[f"st/{key}"] = torch.nan_to_num(value).detach().item()
                else:
                    to_log[f"st/{key}"] = float(value)
            period_input_gate = st_state.get("period_input_gate")
            if period_input_gate is not None:
                to_log["st/period_input_gate"] = torch.nan_to_num(period_input_gate).detach().mean().item()
            period_input_alpha = st_state.get("period_input_alpha")
            if period_input_alpha is not None:
                to_log["st/period_input_alpha"] = torch.nan_to_num(period_input_alpha).detach().abs().item()
            gate = st_state.get("gate")
            if gate is not None:
                to_log['st/gate'] = torch.nan_to_num(gate).detach().mean().item()
            alpha = st_state.get("alpha")
            if alpha is not None:
                to_log['st/alpha'] = torch.nan_to_num(alpha).detach().abs().item()
            trust_gate = st_state.get("trust_gate")
            if trust_gate is not None:
                to_log['st/trust_gate'] = torch.nan_to_num(trust_gate).detach().mean().item()
        elif use_st_residual:
            to_log['st/residual_loss'] = 0.0
            to_log['st/delta_reg'] = 0.0

        to_log['total loss'] = torch.nan_to_num(loss).detach().item()

        return loss, to_log

    def loss_fn_impute(self, x, mask):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''

        to_log = {}
        output, weight = self.forward_impute(x, mask)
        x = self.unpad(x * (1 - mask), x.shape)
        output = self.unpad(output * (1 - mask), x.shape)
        loss = (weight * (output - x).square()).mean()
        to_log['karras loss'] = loss.detach().item()

        return loss, to_log


    def forward(self, x, labels=None, augment_pipe=None, return_sigma=False):

        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)
        n = torch.randn_like(y) * sigma
        D_yn = self.net(y + n, sigma, labels, augment_labels=augment_labels)
        if return_sigma:
            return D_yn, weight, sigma
        return D_yn, weight

    def forward_impute(self, x, mask, labels=None, augment_pipe=None):

        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # noisy impute part
        n = torch.randn_like(x) * sigma
        noise_impute = n * (1 - mask)
        x_to_impute = x * (1 - mask) + noise_impute

        # clear image
        x = x * mask
        y, augment_labels = augment_pipe(x) if augment_pipe is not None else (x, None)

        D_yn = self.net(y + x_to_impute, sigma, labels, augment_labels=augment_labels)
        return D_yn, weight

    def forward_forecast(self, past, future, labels=None, augment_pipe=None):
        s, e = past.shape[-1], future.shape[-1]
        rnd_normal = torch.randn([past.shape[0], 1, 1, 1], device=past.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(past) if augment_pipe is not None else (past, None)
        n = torch.randn_like(future) * sigma
        full_seq = self.pad_f(torch.cat([past, future + n], dim=-1))
        D_yn = self.net(full_seq, sigma, labels, augment_labels=augment_labels)[..., s:(s + e)]
        return D_yn, weight

    def pad_f(self, x):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(32, rows)
        padding = (
            0, max_side - rows, 0, 0)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to temporarily switch to EMA weights during inference.
        Args:
            context: some string to print when switching to EMA weights

        Returns:

        """
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args):
        """
        this function updates the EMA model, if it is used
        Args:
            *args:

        Returns:

        """
        if self.use_ema:
            self.model_ema(self.net)
