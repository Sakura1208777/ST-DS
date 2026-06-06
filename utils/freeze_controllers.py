import logging
import os

import numpy as np
import torch

from utils.utils import restore_checkpoint, save_checkpoint


class STInternalFreezeController:
    def __init__(self, args, model, state):
        self.args = args
        self.model = model
        self.state = state
        self.st_freeze_enabled = bool(getattr(args, 'st_freeze', False))
        self.st_internal_freeze_enabled = bool(getattr(args, 'st_internal_freeze', False))
        self.any_freeze_enabled = self.st_freeze_enabled or self.st_internal_freeze_enabled

    def apply_freeze(self):
        self.model.net.freeze_st_loss = True
        for module in [
            self.model.net.st_denoiser,
            getattr(self.model.net, 'feature_conditioner', None),
            getattr(self.model.net, 'period_input_conditioner', None),
        ]:
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        if hasattr(self.model.net, 'st_alpha_raw') and self.model.net.st_alpha_raw is not None:
            self.model.net.st_alpha_raw.requires_grad = False
        if hasattr(self.model.net, 'period_input_alpha_raw') and self.model.net.period_input_alpha_raw is not None:
            self.model.net.period_input_alpha_raw.requires_grad = False

    def apply_existing_freeze(self):
        if self.any_freeze_enabled and self.state.get('st_frozen', False):
            self.apply_freeze()

    def restore_external_best_and_freeze(self, freeze_reason, optimizer, best_score,
                                         freeze_metrics=None, freeze_epoch=None):
        external_best_checkpoint = self.state.get('best_checkpoint', self.args.log_dir)
        if not (
            np.isfinite(float(best_score))
            and external_best_checkpoint is not None
            and os.path.exists(external_best_checkpoint)
        ):
            return optimizer, best_score, False

        freeze_metrics = freeze_metrics or {}
        external_best_score = best_score
        external_best_metric = self.state.get('best_score_metric', 'external_score')
        external_best_epoch = self.state.get('best_score_epoch', 0)
        logging.info(
            "ST internal freeze: restoring external-best checkpoint "
            "from epoch %s (%s=%.4f), then freeze ST "
            "(reason=%s, health=%.4f, health_best=%.4f, final/base=%.4f, "
            "current_final/base=%.4f, delta_ratio=%.4f, hf_leak=%.4f, "
            "delta_growth=%.4f, hf_growth=%.4f)",
            external_best_epoch,
            external_best_metric,
            external_best_score,
            freeze_reason,
            float(freeze_metrics.get('health_ema', float('nan'))),
            float(freeze_metrics.get('health_best', float('nan'))),
            float(freeze_metrics.get('final_base_mse_ratio_ema', float('nan'))),
            float(freeze_metrics.get('curr_final_base_mse_ratio', float('nan'))),
            float(freeze_metrics.get('delta_ratio_ema', float('nan'))),
            float(freeze_metrics.get('highfreq_leak_ema', float('nan'))),
            float(freeze_metrics.get('best_delta_growth_ratio', float('nan'))),
            float(freeze_metrics.get('best_highfreq_growth_ratio', float('nan'))),
        )
        ema_model_restore = self.model.model_ema if self.args.ema else None
        restore_checkpoint(
            external_best_checkpoint,
            self.state,
            device=self.args.device,
            ema_model=ema_model_restore,
            optimizer=optimizer,
        )
        best_score = external_best_score
        self.state['best_score'] = external_best_score
        self.state['best_score_metric'] = external_best_metric
        self.state['best_score_epoch'] = external_best_epoch
        self.state['best_checkpoint'] = external_best_checkpoint
        self.state['st_freeze_reason'] = freeze_reason
        if self.args.ema and ema_model_restore is not None:
            ema_model_restore.copy_to(self.model.net)
        self.apply_freeze()
        health_freeze_lr_ratio = float(getattr(self.args, 'st_internal_freeze_lr_ratio', 0.22))
        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate * health_freeze_lr_ratio,
            weight_decay=self.args.weight_decay,
        )
        self.state['optimizer'] = optimizer
        self.state['st_frozen'] = True
        self.state['st_frozen_epoch'] = freeze_epoch if freeze_epoch is not None else 0
        self.state['st_health_degrade_count'] = 0
        return optimizer, best_score, True

    def record_external_best_internal_baseline(self, epoch_value):
        if 'st_health_delta_ratio_ema' in self.state:
            self.state['st_external_best_delta_ratio_ema'] = float(self.state['st_health_delta_ratio_ema'])
        if 'st_health_highfreq_leak_ema' in self.state:
            self.state['st_external_best_highfreq_leak_ema'] = float(self.state['st_health_highfreq_leak_ema'])
        if 'st_health_final_base_mse_ratio_ema' in self.state:
            self.state['st_external_best_final_base_mse_ratio_ema'] = float(
                self.state['st_health_final_base_mse_ratio_ema']
            )
        self.state['st_external_best_epoch'] = int(epoch_value)

    def _ensure_external_best_internal_baseline(self):
        if (
            'st_external_best_delta_ratio_ema' in self.state
            and 'st_external_best_highfreq_leak_ema' in self.state
        ):
            return
        if self.state.get('_st_external_best_baseline_load_attempted', False):
            return
        self.state['_st_external_best_baseline_load_attempted'] = True
        external_best_checkpoint = self.state.get('best_checkpoint', None)
        if external_best_checkpoint is None or not os.path.exists(external_best_checkpoint):
            return
        try:
            loaded_state = torch.load(
                external_best_checkpoint,
                map_location='cpu',
                weights_only=False,
            )
        except Exception as exc:
            logging.warning(
                "Could not load external-best internal baseline from %s: %s",
                external_best_checkpoint,
                exc,
            )
            return
        mapping = {
            'st_health_delta_ratio_ema': 'st_external_best_delta_ratio_ema',
            'st_health_highfreq_leak_ema': 'st_external_best_highfreq_leak_ema',
            'st_health_final_base_mse_ratio_ema': 'st_external_best_final_base_mse_ratio_ema',
        }
        for src_key, dst_key in mapping.items():
            if src_key in loaded_state and loaded_state[src_key] is not None:
                self.state[dst_key] = float(loaded_state[src_key])
        self.state['st_external_best_epoch'] = int(
            loaded_state.get('best_score_epoch', loaded_state.get('display_epoch', 0)) or 0
        )

    def update_after_epoch(self, epoch_logs, train_loader_len, human_epoch,
                           optimizer, best_score, internal_eval_due):
        st_internal_freeze = bool(getattr(self.args, 'st_internal_freeze', False))
        if not (
            st_internal_freeze
            and train_loader_len > 0
            and 'st/internal_health' in epoch_logs
            and not self.state.get('st_frozen', False)
        ):
            return optimizer, best_score

        def _avg_log(name, default=0.0):
            return float(epoch_logs.get(name, default)) / float(train_loader_len)

        curr_health = _avg_log('st/internal_health')
        curr_reliability = _avg_log('st/reliability', curr_health)
        curr_alignment = _avg_log('st/internal_alignment')
        curr_delta_ratio = _avg_log('st/internal_delta_ratio')
        curr_highfreq_leak = _avg_log('st/internal_highfreq_leak')
        curr_saturation = _avg_log('st/internal_saturation')
        curr_base_ts_mse = _avg_log('st/base_ts_mse')
        curr_final_ts_mse = _avg_log('st/final_ts_mse')
        curr_final_base_mse_ratio = _avg_log('st/final_base_mse_ratio')
        curr_effective_delta_norm = _avg_log('st/effective_delta_norm')

        health_ema_decay = min(0.999, max(0.0, float(getattr(self.args, 'st_internal_health_ema', 0.95))))

        def _update_ema(key, curr):
            prev = self.state.get(key, None)
            value = curr if prev is None else health_ema_decay * float(prev) + (1.0 - health_ema_decay) * curr
            self.state[key] = value
            return value

        health_ema = _update_ema('st_health_ema_value', curr_health)
        reliability_ema = _update_ema('st_health_reliability_ema', curr_reliability)
        alignment_ema = _update_ema('st_health_alignment_ema', curr_alignment)
        delta_ratio_ema = _update_ema('st_health_delta_ratio_ema', curr_delta_ratio)
        highfreq_leak_ema = _update_ema('st_health_highfreq_leak_ema', curr_highfreq_leak)
        saturation_ema = _update_ema('st_health_saturation_ema', curr_saturation)
        base_ts_mse_ema = _update_ema('st_health_base_ts_mse_ema', curr_base_ts_mse)
        final_ts_mse_ema = _update_ema('st_health_final_ts_mse_ema', curr_final_ts_mse)
        final_base_mse_ratio_ema = _update_ema('st_health_final_base_mse_ratio_ema', curr_final_base_mse_ratio)
        effective_delta_norm_ema = _update_ema('st_health_effective_delta_norm_ema', curr_effective_delta_norm)

        total_epochs = max(int(getattr(self.args, 'epochs', 0) or 0), 1)
        soft_warmup_ratio = float(getattr(self.args, 'st_internal_freeze_warmup_ratio', 0.25) or 0.0)
        legacy_health_trigger = bool(getattr(self.args, 'st_internal_legacy_health_trigger', True))
        monitor_warmup_ratio = float(
            getattr(self.args, 'st_internal_monitor_warmup_ratio', soft_warmup_ratio) or 0.0
        )
        hard_warmup_ratio = float(
            getattr(self.args, 'st_internal_hard_freeze_warmup_ratio', monitor_warmup_ratio) or 0.0
        )
        monitor_epoch = max(1, int(float(total_epochs) * monitor_warmup_ratio))
        hard_epoch = max(monitor_epoch, int(float(total_epochs) * hard_warmup_ratio))
        soft_epoch = max(hard_epoch, int(float(total_epochs) * soft_warmup_ratio))
        patience = max(int(getattr(self.args, 'st_internal_freeze_patience', 3) or 1), 1)
        health_drop = max(0.0, float(getattr(self.args, 'st_internal_health_drop', 0.20) or 0.0))
        reliability_floor = float(getattr(self.args, 'st_internal_reliability_floor', 0.30) or 0.0)
        alignment_floor = float(getattr(self.args, 'st_internal_alignment_floor', 0.05) or 0.0)
        delta_ratio_max = float(getattr(self.args, 'st_internal_delta_ratio_max', 0.65) or 0.0)
        highfreq_leak_max = float(getattr(self.args, 'st_internal_highfreq_leak_max', 1.25) or 0.0)
        structural_soft_ratio = max(
            1.0,
            float(getattr(self.args, 'st_internal_structural_soft_ratio', 1.0) or 1.0),
        )
        structural_hard_ratio = max(
            structural_soft_ratio,
            float(getattr(self.args, 'st_internal_structural_hard_ratio', 1.15) or 1.15),
        )
        saturation_ratio = float(getattr(self.args, 'st_internal_saturation_ratio', 0.90) or 0.0)
        final_mse_ratio_max = float(getattr(self.args, 'st_internal_final_mse_ratio_max', 1.05) or 0.0)
        final_mse_hard_ratio = float(getattr(self.args, 'st_internal_final_mse_hard_ratio', 1.10) or 0.0)
        final_mse_min_abs = max(0.0, float(getattr(self.args, 'st_internal_final_mse_min_abs', 0.0) or 0.0))
        best_delta_growth = max(0.0, float(getattr(self.args, 'st_internal_best_delta_growth', 0.0) or 0.0))
        best_highfreq_growth = max(0.0, float(getattr(self.args, 'st_internal_best_highfreq_growth', 0.0) or 0.0))
        delta_ratio_min = max(0.0, float(getattr(self.args, 'st_internal_delta_ratio_min', 0.0) or 0.0))

        self.state.setdefault('st_health_best', float('-inf'))
        self.state.setdefault('st_health_best_epoch', 0)
        self.state.setdefault('st_health_degrade_count', 0)

        if human_epoch < monitor_epoch:
            return optimizer, best_score

        best_health = float(self.state['st_health_best'])
        if health_ema > float(self.state['st_health_best']):
            self.state['st_health_best'] = health_ema
            self.state['st_health_best_epoch'] = human_epoch
            health_degrade = False
        else:
            health_degrade = best_health > float('-inf') and health_ema < best_health * (1.0 - health_drop)

        residual_unreliable = (
            reliability_floor > 0
            and alignment_floor > 0
            and reliability_ema < reliability_floor
            and alignment_ema < alignment_floor
        )
        st_active = delta_ratio_min <= 0 or delta_ratio_ema > delta_ratio_min or curr_delta_ratio > delta_ratio_min
        unstable_delta = (
            delta_ratio_max > 0
            and highfreq_leak_max > 0
            and delta_ratio_ema > delta_ratio_max
            and highfreq_leak_ema > highfreq_leak_max
        )
        structural_soft_delta = (
            delta_ratio_max > 0
            and delta_ratio_ema > delta_ratio_max * structural_soft_ratio
            and curr_delta_ratio > delta_ratio_max
        )
        structural_soft_highfreq = (
            highfreq_leak_max > 0
            and highfreq_leak_ema > highfreq_leak_max * structural_soft_ratio
            and curr_highfreq_leak > highfreq_leak_max
        )
        structural_hard_delta = (
            delta_ratio_max > 0
            and delta_ratio_ema > delta_ratio_max
            and curr_delta_ratio > delta_ratio_max * structural_hard_ratio
        )
        structural_hard_highfreq = (
            highfreq_leak_max > 0
            and highfreq_leak_ema > highfreq_leak_max
            and curr_highfreq_leak > highfreq_leak_max * structural_hard_ratio
        )
        alignment_degraded = alignment_floor > 0 and alignment_ema < alignment_floor
        structural_degrade = (
            residual_unreliable
            or unstable_delta
            or alignment_degraded
            or (highfreq_leak_max > 0 and highfreq_leak_ema > highfreq_leak_max)
            or (delta_ratio_max > 0 and delta_ratio_ema > delta_ratio_max)
        )
        structure_confirmed = (
            (legacy_health_trigger and health_degrade)
            or final_base_mse_ratio_ema > 1.0
            or curr_final_base_mse_ratio > 1.0
        )

        self._ensure_external_best_internal_baseline()
        best_delta_ratio_ema = self.state.get('st_external_best_delta_ratio_ema', None)
        best_highfreq_leak_ema = self.state.get('st_external_best_highfreq_leak_ema', None)
        best_delta_growth_ratio = float('nan')
        best_highfreq_growth_ratio = float('nan')
        external_best_drift_degrade = False
        if (
            best_delta_growth > 0.0
            and best_highfreq_growth > 0.0
            and best_delta_ratio_ema is not None
            and best_highfreq_leak_ema is not None
        ):
            best_delta_ratio_ema = max(float(best_delta_ratio_ema), 1e-8)
            best_highfreq_leak_ema = max(float(best_highfreq_leak_ema), 1e-8)
            best_delta_growth_ratio = delta_ratio_ema / best_delta_ratio_ema - 1.0
            best_highfreq_growth_ratio = highfreq_leak_ema / best_highfreq_leak_ema - 1.0
            self.state['st_external_best_delta_growth_ratio'] = best_delta_growth_ratio
            self.state['st_external_best_highfreq_growth_ratio'] = best_highfreq_growth_ratio
            external_best_drift_degrade = (
                st_active
                and human_epoch > int(self.state.get('best_score_epoch', 0) or 0)
                and best_delta_growth_ratio > best_delta_growth
                and best_highfreq_growth_ratio > best_highfreq_growth
                and curr_delta_ratio > best_delta_ratio_ema
                and curr_highfreq_leak > best_highfreq_leak_ema
            )
        self.state['st_external_best_drift_active'] = bool(external_best_drift_degrade)
        if external_best_drift_degrade:
            structure_confirmed = True
        external_best_hard_drift_degrade = (
            external_best_drift_degrade
            and best_delta_ratio_ema is not None
            and best_highfreq_leak_ema is not None
            and best_delta_growth_ratio > 1.5 * best_delta_growth
            and best_highfreq_growth_ratio > 1.5 * best_highfreq_growth
        )

        structural_soft_degrade = (
            st_active
            and structure_confirmed
            and (unstable_delta or structural_soft_delta or structural_soft_highfreq or alignment_degraded)
        )
        structural_hard_degrade = (
            st_active
            and structure_confirmed
            and (
                (structural_hard_delta and highfreq_leak_max > 0 and curr_highfreq_leak > highfreq_leak_max)
                or (structural_hard_highfreq and delta_ratio_max > 0 and curr_delta_ratio > delta_ratio_max)
                or (structural_hard_delta and structural_hard_highfreq)
            )
        )
        saturated_and_degrading = saturation_ratio > 0 and saturation_ema > saturation_ratio and health_degrade
        final_mse_soft_degrade = (
            final_mse_ratio_max > 0
            and base_ts_mse_ema > 0
            and final_ts_mse_ema > base_ts_mse_ema * final_mse_ratio_max + final_mse_min_abs
            and curr_final_base_mse_ratio > final_mse_ratio_max
        )
        final_mse_hard_degrade = (
            final_mse_hard_ratio > 0
            and curr_final_base_mse_ratio > final_mse_hard_ratio
            and final_base_mse_ratio_ema > final_mse_ratio_max
        )
        st_harmful_soft_degrade = st_active and final_mse_soft_degrade and effective_delta_norm_ema > 0
        st_harmful_hard_degrade = st_active and final_mse_hard_degrade and effective_delta_norm_ema > 0

        freeze_reason = None
        if human_epoch >= hard_epoch and st_harmful_hard_degrade:
            self.state['st_health_degrade_count'] = patience
            freeze_reason = 'hard_final_mse'
        elif human_epoch >= hard_epoch and structural_hard_degrade:
            self.state['st_health_degrade_count'] = patience
            freeze_reason = 'hard_structure'
        elif human_epoch >= hard_epoch and external_best_hard_drift_degrade:
            self.state['st_health_degrade_count'] = patience
            freeze_reason = 'hard_external_best_drift'
        elif human_epoch >= soft_epoch and st_harmful_soft_degrade:
            self.state['st_health_degrade_count'] += 1
            freeze_reason = 'soft_final_mse'
        elif human_epoch >= soft_epoch and structural_soft_degrade:
            self.state['st_health_degrade_count'] += 1
            freeze_reason = 'soft_structure'
        elif human_epoch >= soft_epoch and external_best_drift_degrade:
            self.state['st_health_degrade_count'] += 1
            freeze_reason = 'external_best_drift'
        elif legacy_health_trigger and human_epoch >= soft_epoch and health_degrade and structural_degrade:
            self.state['st_health_degrade_count'] += 1
            freeze_reason = 'health_structural'
        elif legacy_health_trigger and human_epoch >= soft_epoch and saturated_and_degrading:
            self.state['st_health_degrade_count'] += 1
            freeze_reason = 'alpha_saturation'
        else:
            self.state['st_health_degrade_count'] = 0
            freeze_reason = None
        self.state['st_health_degrade_reason'] = freeze_reason

        freeze_metrics = {
            'health_ema': health_ema,
            'health_best': self.state.get('st_health_best', float('nan')),
            'final_base_mse_ratio_ema': final_base_mse_ratio_ema,
            'curr_final_base_mse_ratio': curr_final_base_mse_ratio,
            'delta_ratio_ema': delta_ratio_ema,
            'highfreq_leak_ema': highfreq_leak_ema,
            'best_delta_ratio_ema': self.state.get('st_external_best_delta_ratio_ema', float('nan')),
            'best_highfreq_leak_ema': self.state.get('st_external_best_highfreq_leak_ema', float('nan')),
            'best_delta_growth_ratio': best_delta_growth_ratio,
            'best_highfreq_growth_ratio': best_highfreq_growth_ratio,
        }

        if self.state.get('st_health_degrade_count', 0) >= patience:
            external_freeze_reason = self.state.get('st_health_degrade_reason', 'unknown')
            defer_until_external_eval = bool(
                getattr(self.args, 'st_internal_cancel_on_external_improve', False)
            )
            if internal_eval_due or defer_until_external_eval:
                self.state['_st_internal_freeze_pending'] = {
                    'reason': external_freeze_reason,
                    'metrics': freeze_metrics,
                    'epoch': human_epoch,
                }
                logging.info(
                    "ST internal freeze: degradation detected at epoch %s; "
                    "deferring ST freeze until after external-best checkpoint update",
                    human_epoch,
                )
            else:
                optimizer, best_score, _ = self.restore_external_best_and_freeze(
                    external_freeze_reason,
                    optimizer,
                    best_score,
                    freeze_metrics=freeze_metrics,
                    freeze_epoch=human_epoch,
                )
        return optimizer, best_score

    def apply_pending_after_eval(self, human_epoch, curr_score_metric, curr_score,
                                 best_score, external_best_improved, optimizer):
        pending_internal_freeze = self.state.pop('_st_internal_freeze_pending', None)
        if not pending_internal_freeze or self.state.get('st_frozen', False):
            return optimizer, best_score

        pending_freeze_reason = pending_internal_freeze.get('reason', 'unknown')
        cancel_on_external_improve = bool(getattr(self.args, 'st_internal_cancel_on_external_improve', False))
        external_confirm_evals = max(int(getattr(self.args, 'st_internal_external_confirm_evals', 1) or 1), 1)
        hard_external_confirm_evals = max(
            int(
                getattr(
                    self.args,
                    'st_internal_hard_external_confirm_evals',
                    external_confirm_evals,
                ) or external_confirm_evals
            ),
            1,
        )
        required_external_confirm_evals = (
            hard_external_confirm_evals
            if str(pending_freeze_reason).startswith('hard_')
            else external_confirm_evals
        )
        external_nonimprove_count = int(self.state.get('st_internal_external_nonimprove_count', 0) or 0)

        if cancel_on_external_improve and external_best_improved:
            logging.info(
                "ST internal freeze: canceled at epoch %s because external best improved (%s=%.4f)",
                human_epoch,
                curr_score_metric,
                curr_score,
            )
            self.state['st_health_degrade_count'] = 0
            return optimizer, best_score

        require_mature_best = bool(getattr(self.args, 'st_internal_external_require_mature_best', False))
        external_best_mature = bool(self.state.get('st_internal_external_best_mature', False))
        if cancel_on_external_improve and require_mature_best and not external_best_mature:
            logging.info(
                "ST internal freeze: delayed at epoch %s; external best is not mature "
                "(reason=%s, %s=%.4f, best=%.4f, improvement=%.4f, required=%.4f)",
                human_epoch,
                pending_freeze_reason,
                curr_score_metric,
                curr_score,
                best_score,
                float(self.state.get('st_internal_external_best_improvement', float('nan'))),
                float(self.state.get('st_internal_external_best_required_improvement', float('nan'))),
            )
            self.state['st_health_degrade_count'] = max(
                int(self.state.get('st_health_degrade_count', 0) or 0) - 1,
                0,
            )
            return optimizer, best_score

        require_external_degrade = bool(getattr(self.args, 'st_internal_external_require_degrade', False))
        external_degraded = bool(self.state.get('st_internal_external_degraded', False))
        if cancel_on_external_improve and require_external_degrade and not external_degraded:
            logging.info(
                "ST internal freeze: delayed at epoch %s; external metric has not degraded enough "
                "(reason=%s, %s=%.4f, best=%.4f, degrade=%.4f, required=%.4f)",
                human_epoch,
                pending_freeze_reason,
                curr_score_metric,
                curr_score,
                best_score,
                float(self.state.get('st_internal_external_degrade', float('nan'))),
                float(self.state.get('st_internal_external_required_degrade', float('nan'))),
            )
            self.state['st_health_degrade_count'] = max(
                int(self.state.get('st_health_degrade_count', 0) or 0) - 1,
                0,
            )
            return optimizer, best_score

        if cancel_on_external_improve and external_nonimprove_count < required_external_confirm_evals:
            logging.info(
                "ST internal freeze: delayed at epoch %s; external non-improve evals "
                "%s/%s (reason=%s, %s=%.4f, best=%.4f)",
                human_epoch,
                external_nonimprove_count,
                required_external_confirm_evals,
                pending_freeze_reason,
                curr_score_metric,
                curr_score,
                best_score,
            )
            self.state['st_health_degrade_count'] = max(
                int(self.state.get('st_health_degrade_count', 0) or 0) - 1,
                0,
            )
            return optimizer, best_score

        optimizer, best_score, _ = self.restore_external_best_and_freeze(
            pending_freeze_reason,
            optimizer,
            best_score,
            freeze_metrics=pending_internal_freeze.get('metrics', {}),
            freeze_epoch=pending_internal_freeze.get('epoch', human_epoch),
        )
        return optimizer, best_score


class F3FreezeController:
    def __init__(self, args, model, state, apply_freeze):
        self.args = args
        self.model = model
        self.state = state
        self.apply_freeze = apply_freeze

    @staticmethod
    def _pct_degraded(curr, best, pct_threshold, min_abs):
        if not np.isfinite(curr) or not np.isfinite(best) or best <= 0:
            return False
        abs_delta = curr - best
        return (
            pct_threshold > 0
            and abs_delta >= min_abs
            and abs_delta / max(best, 1e-12) >= pct_threshold
        )

    @staticmethod
    def _std_degraded(curr_std, best_std, abs_threshold, ratio, min_abs=0.0):
        if not np.isfinite(curr_std) or not np.isfinite(best_std):
            return False
        abs_delta = curr_std - best_std
        abs_degrade = abs_threshold > 0 and abs_delta > abs_threshold
        ratio_degrade = ratio > 0 and best_std > 0 and abs_delta >= min_abs and curr_std > best_std * ratio
        return abs_degrade or ratio_degrade

    def update_after_eval(self, scores, human_epoch, epoch, optimizer, best_score):
        if not getattr(self.args, 'st_freeze', False):
            return optimizer, best_score, False

        st_freeze_warmup = max(int(getattr(self.args, 'st_freeze_warmup', 200)), 0)
        st_freeze_patience = max(int(getattr(self.args, 'st_freeze_patience', 2)), 1)
        st_freeze_threshold = float(getattr(self.args, 'st_freeze_threshold', 0.005))
        st_freeze_hard_threshold = float(getattr(self.args, 'st_freeze_hard_threshold', 0.0) or 0.0)
        st_freeze_pct_threshold = float(getattr(self.args, 'st_freeze_pct_threshold', 0.0) or 0.0)
        st_freeze_min_abs = float(getattr(self.args, 'st_freeze_min_abs', 0.0) or 0.0)
        st_freeze_hard_pct_threshold = float(getattr(self.args, 'st_freeze_hard_pct_threshold', 0.0) or 0.0)
        st_freeze_hard_min_abs = float(getattr(self.args, 'st_freeze_hard_min_abs', 0.0) or 0.0)
        st_freeze_std_threshold = float(getattr(self.args, 'st_freeze_std_threshold', 0.0) or 0.0)
        st_freeze_std_ratio = float(getattr(self.args, 'st_freeze_std_ratio', 0.0) or 0.0)
        st_freeze_std_min_abs = float(getattr(self.args, 'st_freeze_std_min_abs', 0.0) or 0.0)
        st_freeze_watch = bool(getattr(self.args, 'st_freeze_watch', False))
        st_freeze_watch_interval = max(int(getattr(self.args, 'st_freeze_watch_interval', 50) or 50), 1)
        st_freeze_watch_patience = max(int(getattr(self.args, 'st_freeze_watch_patience', 1) or 1), 1)
        st_post_freeze_patience = max(int(getattr(self.args, 'st_post_freeze_patience', 2)), 1)
        st_post_freeze_threshold = float(getattr(self.args, 'st_post_freeze_threshold', 0.01))
        st_post_freeze_hard_threshold = float(getattr(self.args, 'st_post_freeze_hard_threshold', 0.0) or 0.0)
        st_post_freeze_std_threshold = float(getattr(self.args, 'st_post_freeze_std_threshold', 0.0) or 0.0)
        st_post_freeze_std_ratio = float(getattr(self.args, 'st_post_freeze_std_ratio', 0.0) or 0.0)

        self.state.setdefault('st_frozen', False)
        self.state.setdefault('st_degrade_count', 0)
        self.state.setdefault('st_best_disc', float('inf'))
        self.state.setdefault('st_best_disc_std', float('inf'))
        self.state.setdefault('st_best_epoch', 0)
        self.state.setdefault('st_best_checkpoint', None)
        self.state.setdefault('st_post_freeze_degrade', 0)
        self.state.setdefault('st_freeze_watch_active', False)
        self.state.setdefault('st_freeze_watch_epoch', 0)
        self.state.setdefault('st_freeze_watch_count', 0)
        curr_disc = scores.get('disc_mean', float('inf'))
        curr_disc_std = scores.get('disc_std', 0.0)

        if not self.state['st_frozen']:
            if human_epoch >= st_freeze_warmup:
                if curr_disc < self.state['st_best_disc']:
                    self.state['st_best_disc'] = curr_disc
                    self.state['st_best_disc_std'] = curr_disc_std
                    self.state['st_best_epoch'] = human_epoch
                    self.state['st_degrade_count'] = 0
                    self.state['st_freeze_watch_active'] = False
                    self.state['st_freeze_watch_epoch'] = 0
                    self.state['st_freeze_watch_count'] = 0
                    f3_best_path = os.path.join(os.path.dirname(self.args.log_dir), 'checkpoint.f3_disc_best')
                    ema_model_save = self.model.model_ema if self.args.ema else None
                    save_checkpoint(
                        f3_best_path,
                        self.state,
                        epoch,
                        ema_model_save,
                        optimizer=optimizer,
                        best_score=curr_disc,
                    )
                    self.state['st_best_checkpoint'] = f3_best_path
                else:
                    soft_abs_degrade = curr_disc > self.state['st_best_disc'] + st_freeze_threshold
                    soft_pct_degrade = self._pct_degraded(
                        curr_disc,
                        self.state['st_best_disc'],
                        st_freeze_pct_threshold,
                        st_freeze_min_abs,
                    )
                    soft_degrade = soft_abs_degrade or soft_pct_degrade
                    hard_abs_degrade = (
                        st_freeze_hard_threshold > 0
                        and curr_disc > self.state['st_best_disc'] + st_freeze_hard_threshold
                    )
                    hard_pct_degrade = self._pct_degraded(
                        curr_disc,
                        self.state['st_best_disc'],
                        st_freeze_hard_pct_threshold,
                        st_freeze_hard_min_abs,
                    )
                    hard_degrade = hard_abs_degrade or hard_pct_degrade
                    std_degrade = self._std_degraded(
                        curr_disc_std,
                        self.state['st_best_disc_std'],
                        st_freeze_std_threshold,
                        st_freeze_std_ratio,
                        st_freeze_std_min_abs,
                    )
                    combined_degrade = soft_degrade and std_degrade
                    if hard_degrade:
                        self.state['st_degrade_count'] = st_freeze_patience
                    elif st_freeze_watch and combined_degrade:
                        if self.state.get('st_freeze_watch_active', False):
                            self.state['st_freeze_watch_count'] += 1
                            if self.state['st_freeze_watch_count'] >= st_freeze_watch_patience:
                                self.state['st_degrade_count'] = st_freeze_patience
                        else:
                            self.state['st_freeze_watch_active'] = True
                            self.state['st_freeze_watch_epoch'] = human_epoch + st_freeze_watch_interval
                            self.state['st_freeze_watch_count'] = 0
                            self.state['st_degrade_count'] = 0
                            logging.info(
                                "F3 watch: suspicious degradation at epoch %s; next check at epoch %s",
                                human_epoch,
                                self.state['st_freeze_watch_epoch'],
                            )
                    elif combined_degrade:
                        self.state['st_degrade_count'] = st_freeze_patience
                    elif soft_degrade:
                        self.state['st_degrade_count'] += 1
                        if st_freeze_watch and self.state.get('st_freeze_watch_active', False):
                            self.state['st_freeze_watch_active'] = False
                            self.state['st_freeze_watch_epoch'] = 0
                            self.state['st_freeze_watch_count'] = 0
                    else:
                        self.state['st_degrade_count'] = 0
                        self.state['st_freeze_watch_active'] = False
                        self.state['st_freeze_watch_epoch'] = 0
                        self.state['st_freeze_watch_count'] = 0

            if (
                self.state['st_degrade_count'] >= st_freeze_patience
                and self.state['st_best_checkpoint'] is not None
                and os.path.exists(self.state['st_best_checkpoint'])
            ):
                logging.info(f"F3: restoring best checkpoint from epoch {self.state['st_best_epoch']}, then freeze")
                ema_model_restore = self.model.model_ema if self.args.ema else None
                external_best_score = best_score
                restore_checkpoint(
                    self.state['st_best_checkpoint'],
                    self.state,
                    device=self.args.device,
                    ema_model=ema_model_restore,
                    optimizer=optimizer,
                )
                best_score = external_best_score
                self.state['best_score'] = external_best_score
                if self.args.ema and ema_model_restore is not None:
                    ema_model_restore.copy_to(self.model.net)
                self.apply_freeze()
                st_freeze_lr_ratio = float(getattr(self.args, 'st_freeze_lr_ratio', 0.3))
                optimizer = torch.optim.AdamW(
                    filter(lambda param: param.requires_grad, self.model.parameters()),
                    lr=self.args.learning_rate * st_freeze_lr_ratio,
                    weight_decay=self.args.weight_decay,
                )
                self.state['optimizer'] = optimizer
                self.state['st_frozen'] = True
                self.state['st_frozen_epoch'] = human_epoch
                self.state['st_degrade_count'] = 0
                self.state['st_freeze_watch_active'] = False
                self.state['st_freeze_watch_epoch'] = 0
                self.state['st_freeze_watch_count'] = 0
            return optimizer, best_score, False

        post_soft_degrade = curr_disc > self.state['st_best_disc'] + st_post_freeze_threshold
        post_hard_degrade = (
            st_post_freeze_hard_threshold > 0
            and curr_disc > self.state['st_best_disc'] + st_post_freeze_hard_threshold
        )
        post_std_degrade = self._std_degraded(
            curr_disc_std,
            self.state['st_best_disc_std'],
            st_post_freeze_std_threshold,
            st_post_freeze_std_ratio,
        )
        if post_hard_degrade or (post_soft_degrade and post_std_degrade):
            self.state['st_post_freeze_degrade'] = st_post_freeze_patience
        elif post_soft_degrade:
            self.state['st_post_freeze_degrade'] += 1
        else:
            self.state['st_post_freeze_degrade'] = 0
        should_early_stop = self.state['st_post_freeze_degrade'] >= st_post_freeze_patience
        if should_early_stop:
            logging.info(f"F3: UNet degrading after freeze (disc={curr_disc:.4f}), early stop")
        return optimizer, best_score, should_early_stop
