import os
import sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from tqdm import tqdm
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, restore_checkpoint, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, latest_checkpoint_path, save_eval_results
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_uncond

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args, new_run=not args.resume)

    # log args
    logging.info(args)

    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # set-up data and device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        state = dict(model=model, optimizer=optimizer, epoch=0, best_score=float('inf'),
                     run_id=getattr(args, 'run_id', None), run_name=getattr(args, 'run_name', None),
                     dataset=getattr(args, 'dataset', None))
        init_epoch = 0

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model, optimizer=optimizer)

        # print model parameters
        print_model_params(logger, model)

        # --- train model ---
        logging.info(f"Continuing training loop from epoch {init_epoch}.")
        best_score = state.get('best_score', float('inf'))
        if np.isfinite(float(best_score)):
            state.setdefault('best_checkpoint', args.log_dir)
        should_early_stop = False
        st_freeze_enabled = getattr(args, 'st_freeze', False)
        st_internal_freeze_enabled = getattr(args, 'st_internal_freeze', False)
        st_any_freeze_enabled = st_freeze_enabled or st_internal_freeze_enabled

        def _apply_freeze():
            model.net.freeze_st_loss = True
            for m in [model.net.st_denoiser,
                      getattr(model.net, 'feature_conditioner', None),
                      getattr(model.net, 'period_input_conditioner', None)]:
                if m is not None:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
            if hasattr(model.net, 'st_alpha_raw') and model.net.st_alpha_raw is not None:
                model.net.st_alpha_raw.requires_grad = False
            if hasattr(model.net, 'period_input_alpha_raw') and model.net.period_input_alpha_raw is not None:
                model.net.period_input_alpha_raw.requires_grad = False

        def _restore_external_best_and_freeze(freeze_reason, freeze_metrics=None, freeze_epoch=None):
            nonlocal optimizer, best_score
            external_best_checkpoint = state.get('best_checkpoint', args.log_dir)
            if not (
                np.isfinite(float(best_score))
                and external_best_checkpoint is not None
                and os.path.exists(external_best_checkpoint)
            ):
                return False

            freeze_metrics = freeze_metrics or {}
            external_best_score = best_score
            external_best_metric = state.get('best_score_metric', 'external_score')
            external_best_epoch = state.get('best_score_epoch', 0)
            logging.info(
                "PRO3: internal health degraded; restoring external-best checkpoint "
                "from epoch %s (%s=%.4f), then freeze ST "
                "(reason=%s, health=%.4f, health_best=%.4f, final/base=%.4f, "
                "current_final/base=%.4f, delta_ratio=%.4f, hf_leak=%.4f)",
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
            )
            ema_model_restore = model.model_ema if args.ema else None
            restore_checkpoint(external_best_checkpoint, state, device=args.device,
                               ema_model=ema_model_restore, optimizer=optimizer)
            best_score = external_best_score
            state['best_score'] = external_best_score
            state['best_score_metric'] = external_best_metric
            state['best_score_epoch'] = external_best_epoch
            state['best_checkpoint'] = external_best_checkpoint
            state['st_freeze_reason'] = freeze_reason
            if args.ema and ema_model_restore is not None:
                ema_model_restore.copy_to(model.net)
            _apply_freeze()
            health_freeze_lr_ratio = float(getattr(args, 'st_internal_freeze_lr_ratio', 0.22))
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate * health_freeze_lr_ratio,
                weight_decay=args.weight_decay,
            )
            state['optimizer'] = optimizer
            state['st_frozen'] = True
            state['st_frozen_epoch'] = freeze_epoch if freeze_epoch is not None else 0
            state['st_health_degrade_count'] = 0
            return True

        if st_any_freeze_enabled and state.get('st_frozen', False):
            _apply_freeze()
        train_bar_format = "{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        eval_bar_format = "{percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        for epoch in range(init_epoch, args.epochs):
            if should_early_stop:
                break
            human_epoch = epoch + 1
            model.train()
            if st_any_freeze_enabled and state.get('st_frozen', False):
                _apply_freeze()
            model.epoch = human_epoch
            if hasattr(model.net, 'current_epoch'):
                model.net.current_epoch = human_epoch
            if args.neptune:
                logger.log_name_params('train/epoch', human_epoch)

            # --- train loop ---
            epoch_loss = 0.0
            epoch_logs = {}
            train_bar = tqdm(train_loader, total=len(train_loader), leave=True,
                             desc=f"epoch {epoch + 1:04d}/{args.epochs}",
                             bar_format=train_bar_format)
            for i, data in enumerate(train_bar, 1):
                x_ts = data[0].to(args.device)
                x_img = model.ts_to_img(x_ts)
                optimizer.zero_grad()
                loss_result = model.loss_fn(x_img, x_ts=x_ts)
                if len(loss_result) == 2:
                    loss, to_log = loss_result
                else:
                    loss, to_log = loss_result, {}

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                model.on_train_batch_end()

                loss_value = loss.detach().item()
                epoch_loss += loss_value
                for key, value in to_log.items():
                    epoch_logs[key] = epoch_logs.get(key, 0.0) + float(value)
                train_bar.set_postfix_str(f"loss={epoch_loss / i:.4e}")

            if args.neptune and len(train_loader) > 0:
                for key, value in epoch_logs.items():
                    logger.log(f'train/{key}', value / len(train_loader), epoch)

            # --- PRO3 internal health freeze ---
            st_internal_freeze = bool(getattr(args, 'st_internal_freeze', False))
            if (
                st_internal_freeze
                and len(train_loader) > 0
                and 'st/internal_health' in epoch_logs
                and not state.get('st_frozen', False)
            ):
                def _avg_log(name, default=0.0):
                    return float(epoch_logs.get(name, default)) / float(len(train_loader))

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

                health_ema_decay = min(0.999, max(0.0, float(getattr(args, 'st_internal_health_ema', 0.95))))

                def _update_ema(key, curr):
                    prev = state.get(key, None)
                    value = curr if prev is None else health_ema_decay * float(prev) + (1.0 - health_ema_decay) * curr
                    state[key] = value
                    return value

                health_ema = _update_ema('st_health_ema_value', curr_health)
                reliability_ema = _update_ema('st_health_reliability_ema', curr_reliability)
                alignment_ema = _update_ema('st_health_alignment_ema', curr_alignment)
                delta_ratio_ema = _update_ema('st_health_delta_ratio_ema', curr_delta_ratio)
                highfreq_leak_ema = _update_ema('st_health_highfreq_leak_ema', curr_highfreq_leak)
                saturation_ema = _update_ema('st_health_saturation_ema', curr_saturation)
                base_ts_mse_ema = _update_ema('st_health_base_ts_mse_ema', curr_base_ts_mse)
                final_ts_mse_ema = _update_ema('st_health_final_ts_mse_ema', curr_final_ts_mse)
                final_base_mse_ratio_ema = _update_ema(
                    'st_health_final_base_mse_ratio_ema',
                    curr_final_base_mse_ratio,
                )
                effective_delta_norm_ema = _update_ema(
                    'st_health_effective_delta_norm_ema',
                    curr_effective_delta_norm,
                )

                total_epochs = max(int(getattr(args, 'epochs', 0) or 0), 1)
                soft_warmup_ratio = float(getattr(args, 'st_internal_freeze_warmup_ratio', 0.25) or 0.0)
                legacy_health_trigger = bool(getattr(args, 'st_internal_legacy_health_trigger', True))
                monitor_warmup_ratio = float(
                    getattr(args, 'st_internal_monitor_warmup_ratio', soft_warmup_ratio) or 0.0
                )
                hard_warmup_ratio = float(
                    getattr(args, 'st_internal_hard_freeze_warmup_ratio', monitor_warmup_ratio) or 0.0
                )
                monitor_epoch = max(1, int(float(total_epochs) * monitor_warmup_ratio))
                hard_epoch = max(monitor_epoch, int(float(total_epochs) * hard_warmup_ratio))
                soft_epoch = max(hard_epoch, int(float(total_epochs) * soft_warmup_ratio))
                patience = max(int(getattr(args, 'st_internal_freeze_patience', 3) or 1), 1)
                health_drop = max(0.0, float(getattr(args, 'st_internal_health_drop', 0.20) or 0.0))
                reliability_floor = float(getattr(args, 'st_internal_reliability_floor', 0.30) or 0.0)
                alignment_floor = float(getattr(args, 'st_internal_alignment_floor', 0.05) or 0.0)
                delta_ratio_max = float(getattr(args, 'st_internal_delta_ratio_max', 0.65) or 0.0)
                highfreq_leak_max = float(getattr(args, 'st_internal_highfreq_leak_max', 1.25) or 0.0)
                structural_soft_ratio = max(
                    1.0,
                    float(getattr(args, 'st_internal_structural_soft_ratio', 1.0) or 1.0),
                )
                structural_hard_ratio = max(
                    structural_soft_ratio,
                    float(getattr(args, 'st_internal_structural_hard_ratio', 1.15) or 1.15),
                )
                saturation_ratio = float(getattr(args, 'st_internal_saturation_ratio', 0.90) or 0.0)
                final_mse_ratio_max = float(getattr(args, 'st_internal_final_mse_ratio_max', 1.05) or 0.0)
                final_mse_hard_ratio = float(getattr(args, 'st_internal_final_mse_hard_ratio', 1.10) or 0.0)
                final_mse_min_abs = max(
                    0.0,
                    float(getattr(args, 'st_internal_final_mse_min_abs', 0.0) or 0.0),
                )
                delta_ratio_min = max(
                    0.0,
                    float(getattr(args, 'st_internal_delta_ratio_min', 0.0) or 0.0),
                )

                state.setdefault('st_health_best', float('-inf'))
                state.setdefault('st_health_best_epoch', 0)
                state.setdefault('st_health_degrade_count', 0)

                if human_epoch >= monitor_epoch:
                    best_health = float(state['st_health_best'])
                    if health_ema > float(state['st_health_best']):
                        state['st_health_best'] = health_ema
                        state['st_health_best_epoch'] = human_epoch
                        health_degrade = False
                    else:
                        health_degrade = (
                            best_health > float('-inf')
                            and health_ema < best_health * (1.0 - health_drop)
                        )
                    residual_unreliable = (
                        reliability_floor > 0
                        and alignment_floor > 0
                        and reliability_ema < reliability_floor
                        and alignment_ema < alignment_floor
                    )
                    st_active = (
                        delta_ratio_min <= 0
                        or delta_ratio_ema > delta_ratio_min
                        or curr_delta_ratio > delta_ratio_min
                    )
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
                    alignment_degraded = (
                        alignment_floor > 0
                        and alignment_ema < alignment_floor
                    )
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
                    structural_soft_degrade = (
                        st_active
                        and structure_confirmed
                        and (
                            unstable_delta
                            or structural_soft_delta
                            or structural_soft_highfreq
                            or alignment_degraded
                        )
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
                    saturated_and_degrading = (
                        saturation_ratio > 0
                        and saturation_ema > saturation_ratio
                        and health_degrade
                    )
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
                    st_harmful_soft_degrade = (
                        st_active
                        and final_mse_soft_degrade
                        and effective_delta_norm_ema > 0
                    )
                    st_harmful_hard_degrade = (
                        st_active
                        and final_mse_hard_degrade
                        and effective_delta_norm_ema > 0
                    )
                    freeze_reason = None
                    if human_epoch >= hard_epoch and st_harmful_hard_degrade:
                        state['st_health_degrade_count'] = patience
                        freeze_reason = 'hard_final_mse'
                    elif human_epoch >= hard_epoch and structural_hard_degrade:
                        state['st_health_degrade_count'] = patience
                        freeze_reason = 'hard_structure'
                    elif human_epoch >= soft_epoch and st_harmful_soft_degrade:
                        state['st_health_degrade_count'] += 1
                        freeze_reason = 'soft_final_mse'
                    elif human_epoch >= soft_epoch and structural_soft_degrade:
                        state['st_health_degrade_count'] += 1
                        freeze_reason = 'soft_structure'
                    elif (
                        legacy_health_trigger
                        and human_epoch >= soft_epoch
                        and health_degrade
                        and structural_degrade
                    ):
                        state['st_health_degrade_count'] += 1
                        freeze_reason = 'health_structural'
                    elif legacy_health_trigger and human_epoch >= soft_epoch and saturated_and_degrading:
                        state['st_health_degrade_count'] += 1
                        freeze_reason = 'alpha_saturation'
                    else:
                        state['st_health_degrade_count'] = 0
                        freeze_reason = None
                    state['st_health_degrade_reason'] = freeze_reason

                    freeze_metrics = {
                        'health_ema': health_ema,
                        'health_best': state.get('st_health_best', float('nan')),
                        'final_base_mse_ratio_ema': final_base_mse_ratio_ema,
                        'curr_final_base_mse_ratio': curr_final_base_mse_ratio,
                        'delta_ratio_ema': delta_ratio_ema,
                        'highfreq_leak_ema': highfreq_leak_ema,
                    }
                    internal_eval_due = (human_epoch % args.logging_iter == 0) or (human_epoch == args.epochs)
                    if (
                        state.get('st_health_degrade_count', 0) >= patience
                    ):
                        external_freeze_reason = state.get('st_health_degrade_reason', 'unknown')
                        if internal_eval_due:
                            state['_st_internal_freeze_pending'] = {
                                'reason': external_freeze_reason,
                                'metrics': freeze_metrics,
                                'epoch': human_epoch,
                            }
                            logging.info(
                                "PRO3: internal degradation detected at eval epoch %s; "
                                "deferring ST freeze until after external-best checkpoint update",
                                human_epoch,
                            )
                        else:
                            _restore_external_best_and_freeze(
                                external_freeze_reason,
                                freeze_metrics=freeze_metrics,
                                freeze_epoch=human_epoch,
                            )

            # --- evaluation loop ---
            st_watch_due = (
                st_freeze_enabled
                and bool(getattr(args, 'st_freeze_watch', False))
                and not state.get('st_frozen', False)
                and state.get('st_freeze_watch_active', False)
                and human_epoch >= int(state.get('st_freeze_watch_epoch', 0) or 0)
            )
            should_evaluate = (human_epoch % args.logging_iter == 0) or st_watch_due or (human_epoch == args.epochs)
            if should_evaluate:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        tqdm.write(f"eval sampling: epoch {human_epoch:04d}/{args.epochs}")
                        for data in tqdm(test_loader, total=len(test_loader), leave=True,
                                         bar_format=eval_bar_format):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            # special case for temperature_rain dataset
                            if args.dataset in ['temperature_rain']:
                                x_ts = torch.clamp(x_ts, 0, 1)

                            gen_sig.append(x_ts.detach().cpu().numpy())
                            real_sig.append(data[0].detach().cpu().numpy())
                            if hasattr(model.net, "pop_st_state"):
                                model.net.pop_st_state()

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)
                scores = evaluate_model_uncond(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, human_epoch)
                save_eval_results(args, human_epoch, scores)

                # --- F3 dynamic freeze ---
                st_freeze = getattr(args, 'st_freeze', False)
                if st_freeze:
                    st_freeze_warmup = max(int(getattr(args, 'st_freeze_warmup', 200)), 0)
                    st_freeze_patience = max(int(getattr(args, 'st_freeze_patience', 2)), 1)
                    st_freeze_threshold = float(getattr(args, 'st_freeze_threshold', 0.005))
                    st_freeze_hard_threshold = float(getattr(args, 'st_freeze_hard_threshold', 0.0) or 0.0)
                    st_freeze_pct_threshold = float(getattr(args, 'st_freeze_pct_threshold', 0.0) or 0.0)
                    st_freeze_min_abs = float(getattr(args, 'st_freeze_min_abs', 0.0) or 0.0)
                    st_freeze_hard_pct_threshold = float(
                        getattr(args, 'st_freeze_hard_pct_threshold', 0.0) or 0.0
                    )
                    st_freeze_hard_min_abs = float(getattr(args, 'st_freeze_hard_min_abs', 0.0) or 0.0)
                    st_freeze_std_threshold = float(getattr(args, 'st_freeze_std_threshold', 0.0) or 0.0)
                    st_freeze_std_ratio = float(getattr(args, 'st_freeze_std_ratio', 0.0) or 0.0)
                    st_freeze_std_min_abs = float(getattr(args, 'st_freeze_std_min_abs', 0.0) or 0.0)
                    st_freeze_watch = bool(getattr(args, 'st_freeze_watch', False))
                    st_freeze_watch_interval = max(int(getattr(args, 'st_freeze_watch_interval', 50) or 50), 1)
                    st_freeze_watch_patience = max(int(getattr(args, 'st_freeze_watch_patience', 1) or 1), 1)
                    st_post_freeze_patience = max(int(getattr(args, 'st_post_freeze_patience', 2)), 1)
                    st_post_freeze_threshold = float(getattr(args, 'st_post_freeze_threshold', 0.01))
                    st_post_freeze_hard_threshold = float(
                        getattr(args, 'st_post_freeze_hard_threshold', 0.0) or 0.0
                    )
                    st_post_freeze_std_threshold = float(
                        getattr(args, 'st_post_freeze_std_threshold', 0.0) or 0.0
                    )
                    st_post_freeze_std_ratio = float(getattr(args, 'st_post_freeze_std_ratio', 0.0) or 0.0)
                if st_freeze:
                    state.setdefault('st_frozen', False)
                    state.setdefault('st_degrade_count', 0)
                    state.setdefault('st_best_disc', float('inf'))
                    state.setdefault('st_best_disc_std', float('inf'))
                    state.setdefault('st_best_epoch', 0)
                    state.setdefault('st_best_checkpoint', None)
                    state.setdefault('st_post_freeze_degrade', 0)
                    state.setdefault('st_freeze_watch_active', False)
                    state.setdefault('st_freeze_watch_epoch', 0)
                    state.setdefault('st_freeze_watch_count', 0)
                    curr_disc = scores.get('disc_mean', float('inf'))
                    curr_disc_std = scores.get('disc_std', 0.0)

                    def _pct_degraded(curr, best, pct_threshold, min_abs):
                        if not np.isfinite(curr) or not np.isfinite(best) or best <= 0:
                            return False
                        abs_delta = curr - best
                        return (
                            pct_threshold > 0
                            and abs_delta >= min_abs
                            and abs_delta / max(best, 1e-12) >= pct_threshold
                        )

                    def _std_degraded(curr_std, best_std, abs_threshold, ratio, min_abs=0.0):
                        if not np.isfinite(curr_std) or not np.isfinite(best_std):
                            return False
                        abs_delta = curr_std - best_std
                        abs_degrade = abs_threshold > 0 and abs_delta > abs_threshold
                        ratio_degrade = (
                            ratio > 0
                            and best_std > 0
                            and abs_delta >= min_abs
                            and curr_std > best_std * ratio
                        )
                        return abs_degrade or ratio_degrade

                    if not state['st_frozen']:
                        if human_epoch >= st_freeze_warmup:
                            if curr_disc < state['st_best_disc']:
                                state['st_best_disc'] = curr_disc
                                state['st_best_disc_std'] = curr_disc_std
                                state['st_best_epoch'] = human_epoch
                                state['st_degrade_count'] = 0
                                state['st_freeze_watch_active'] = False
                                state['st_freeze_watch_epoch'] = 0
                                state['st_freeze_watch_count'] = 0
                                f3_best_path = os.path.join(os.path.dirname(args.log_dir), 'checkpoint.f3_disc_best')
                                ema_model_save = model.model_ema if args.ema else None
                                save_checkpoint(f3_best_path, state, epoch, ema_model_save,
                                                optimizer=optimizer, best_score=curr_disc)
                                state['st_best_checkpoint'] = f3_best_path
                            else:
                                soft_abs_degrade = curr_disc > state['st_best_disc'] + st_freeze_threshold
                                soft_pct_degrade = _pct_degraded(
                                    curr_disc, state['st_best_disc'],
                                    st_freeze_pct_threshold, st_freeze_min_abs,
                                )
                                soft_degrade = soft_abs_degrade or soft_pct_degrade
                                hard_abs_degrade = (
                                    st_freeze_hard_threshold > 0
                                    and curr_disc > state['st_best_disc'] + st_freeze_hard_threshold
                                )
                                hard_pct_degrade = _pct_degraded(
                                    curr_disc, state['st_best_disc'],
                                    st_freeze_hard_pct_threshold, st_freeze_hard_min_abs,
                                )
                                hard_degrade = hard_abs_degrade or hard_pct_degrade
                                std_degrade = _std_degraded(
                                    curr_disc_std, state['st_best_disc_std'],
                                    st_freeze_std_threshold, st_freeze_std_ratio,
                                    st_freeze_std_min_abs,
                                )
                                combined_degrade = soft_degrade and std_degrade
                                if hard_degrade:
                                    state['st_degrade_count'] = st_freeze_patience
                                elif st_freeze_watch and combined_degrade:
                                    if state.get('st_freeze_watch_active', False):
                                        state['st_freeze_watch_count'] += 1
                                        if state['st_freeze_watch_count'] >= st_freeze_watch_patience:
                                            state['st_degrade_count'] = st_freeze_patience
                                    else:
                                        state['st_freeze_watch_active'] = True
                                        state['st_freeze_watch_epoch'] = human_epoch + st_freeze_watch_interval
                                        state['st_freeze_watch_count'] = 0
                                        state['st_degrade_count'] = 0
                                        logging.info(
                                            "F3 watch: suspicious degradation at epoch %s; next check at epoch %s",
                                            human_epoch, state['st_freeze_watch_epoch'],
                                        )
                                elif combined_degrade:
                                    state['st_degrade_count'] = st_freeze_patience
                                elif soft_degrade:
                                    state['st_degrade_count'] += 1
                                    if st_freeze_watch and state.get('st_freeze_watch_active', False):
                                        state['st_freeze_watch_active'] = False
                                        state['st_freeze_watch_epoch'] = 0
                                        state['st_freeze_watch_count'] = 0
                                else:
                                    state['st_degrade_count'] = 0
                                    state['st_freeze_watch_active'] = False
                                    state['st_freeze_watch_epoch'] = 0
                                    state['st_freeze_watch_count'] = 0

                        if state['st_degrade_count'] >= st_freeze_patience \
                           and state['st_best_checkpoint'] is not None \
                           and os.path.exists(state['st_best_checkpoint']):
                            logging.info(f"F3: restoring best checkpoint from epoch {state['st_best_epoch']}, then freeze")
                            ema_model_restore = model.model_ema if args.ema else None
                            external_best_score = best_score
                            restore_checkpoint(state['st_best_checkpoint'], state,
                                               device=args.device,
                                               ema_model=ema_model_restore, optimizer=optimizer)
                            best_score = external_best_score
                            state['best_score'] = external_best_score
                            if args.ema and ema_model_restore is not None:
                                ema_model_restore.copy_to(model.net)
                            _apply_freeze()
                            st_freeze_lr_ratio = float(getattr(args, 'st_freeze_lr_ratio', 0.3))
                            optimizer = torch.optim.AdamW(
                                filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate * st_freeze_lr_ratio,
                                weight_decay=args.weight_decay,
                            )
                            state['optimizer'] = optimizer
                            state['st_frozen'] = True
                            state['st_frozen_epoch'] = human_epoch
                            state['st_degrade_count'] = 0
                            state['st_freeze_watch_active'] = False
                            state['st_freeze_watch_epoch'] = 0
                            state['st_freeze_watch_count'] = 0

                    else:
                        post_soft_degrade = curr_disc > state['st_best_disc'] + st_post_freeze_threshold
                        post_hard_degrade = (
                            st_post_freeze_hard_threshold > 0
                            and curr_disc > state['st_best_disc'] + st_post_freeze_hard_threshold
                        )
                        post_std_degrade = _std_degraded(
                            curr_disc_std, state['st_best_disc_std'],
                            st_post_freeze_std_threshold, st_post_freeze_std_ratio,
                        )
                        if post_hard_degrade or (post_soft_degrade and post_std_degrade):
                            state['st_post_freeze_degrade'] = st_post_freeze_patience
                        elif post_soft_degrade:
                            state['st_post_freeze_degrade'] += 1
                        else:
                            state['st_post_freeze_degrade'] = 0
                        if state['st_post_freeze_degrade'] >= st_post_freeze_patience:
                            logging.info(f"F3: UNet degrading after freeze (disc={curr_disc:.4f}), early stop")
                            should_early_stop = True

                # --- save checkpoint ---
                if 'marginal_score_mean' in scores:
                    curr_score_metric = 'marginal_score_mean'
                else:
                    curr_score_metric = 'disc_mean'
                curr_score = scores[curr_score_metric]
                if curr_score < best_score:
                    best_score = curr_score
                    state['best_score'] = best_score
                    state['best_score_metric'] = curr_score_metric
                    state['best_score_epoch'] = human_epoch
                    state['best_checkpoint'] = args.log_dir
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch, ema_model, optimizer=optimizer, best_score=best_score)
                pending_internal_freeze = state.pop('_st_internal_freeze_pending', None)
                if pending_internal_freeze and not state.get('st_frozen', False):
                    _restore_external_best_and_freeze(
                        pending_internal_freeze.get('reason', 'unknown'),
                        freeze_metrics=pending_internal_freeze.get('metrics', {}),
                        freeze_epoch=pending_internal_freeze.get('epoch', human_epoch),
                    )
                del gen_sig, real_sig, scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if getattr(args, 'save_latest', True):
                state['best_score'] = best_score
                ema_model = model.model_ema if args.ema else None
                save_checkpoint(latest_checkpoint_path(args.log_dir), state, epoch, ema_model,
                                optimizer=optimizer, best_score=best_score)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_uncond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
