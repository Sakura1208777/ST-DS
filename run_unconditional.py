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
from utils.freeze_controllers import F3FreezeController, STInternalFreezeController
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, latest_checkpoint_path, save_eval_results
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_uncond

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')


def sample_unconditional_signals(args, model, test_loader, epoch_label, bar_format):
    gen_sig = []
    real_sig = []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(
                args,
                model.net,
                (args.input_channels, args.img_resolution, args.img_resolution),
            )
            tqdm.write(f"eval sampling: epoch {epoch_label:04d}/{args.epochs}")
            for data in tqdm(test_loader, total=len(test_loader), leave=True, bar_format=bar_format):
                x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                x_ts = model.img_to_ts(x_img_sampled)

                if args.dataset in ['temperature_rain']:
                    x_ts = torch.clamp(x_ts, 0, 1)

                gen_sig.append(x_ts.detach().cpu().numpy())
                real_sig.append(data[0].detach().cpu().numpy())
                if hasattr(model.net, "pop_st_state"):
                    model.net.pop_st_state()

    return np.vstack(gen_sig), np.vstack(real_sig)


def evaluate_unconditional_epoch(args, model, test_loader, logger, epoch_label, bar_format):
    gen_sig, real_sig = sample_unconditional_signals(
        args,
        model,
        test_loader,
        epoch_label,
        bar_format,
    )
    scores = evaluate_model_uncond(real_sig, gen_sig, args)
    for key, value in scores.items():
        logger.log(f'test/{key}', value, epoch_label)
    save_eval_results(args, epoch_label, scores)
    return scores


def select_best_score_metric(scores):
    if 'marginal_score_mean' in scores:
        return 'marginal_score_mean'
    return 'disc_mean'


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
        st_controller = STInternalFreezeController(args, model, state)
        f3_controller = F3FreezeController(args, model, state, st_controller.apply_freeze)
        st_controller.apply_existing_freeze()
        train_bar_format = "{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        eval_bar_format = "{percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        for epoch in range(init_epoch, args.epochs):
            if should_early_stop:
                break
            human_epoch = epoch + 1
            model.train()
            st_controller.apply_existing_freeze()
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

            internal_eval_due = (human_epoch % args.logging_iter == 0) or (human_epoch == args.epochs)
            optimizer, best_score = st_controller.update_after_epoch(
                epoch_logs,
                len(train_loader),
                human_epoch,
                optimizer,
                best_score,
                internal_eval_due,
            )

            # --- evaluation loop ---
            st_watch_due = (
                st_controller.st_freeze_enabled
                and bool(getattr(args, 'st_freeze_watch', False))
                and not state.get('st_frozen', False)
                and state.get('st_freeze_watch_active', False)
                and human_epoch >= int(state.get('st_freeze_watch_epoch', 0) or 0)
            )
            should_evaluate = (human_epoch % args.logging_iter == 0) or st_watch_due or (human_epoch == args.epochs)
            if should_evaluate:
                scores = evaluate_unconditional_epoch(
                    args,
                    model,
                    test_loader,
                    logger,
                    human_epoch,
                    eval_bar_format,
                )

                optimizer, best_score, f3_should_stop = f3_controller.update_after_eval(
                    scores,
                    human_epoch,
                    epoch,
                    optimizer,
                    best_score,
                )
                if f3_should_stop:
                    should_early_stop = True

                # --- save checkpoint ---
                curr_score_metric = select_best_score_metric(scores)
                curr_score = scores[curr_score_metric]
                if bool(getattr(args, 'st_internal_freeze', False)):
                    state.setdefault('st_internal_external_first_epoch', human_epoch)
                    state.setdefault('st_internal_external_first_metric', curr_score_metric)
                    state.setdefault('st_internal_external_first_score', float(curr_score))
                prev_best_score = best_score
                external_best_improved = curr_score < prev_best_score
                state['st_internal_external_metric'] = curr_score_metric
                state['st_internal_external_score'] = float(curr_score)
                state['st_internal_external_best_score'] = float(prev_best_score)
                if external_best_improved:
                    best_score = curr_score
                    state['best_score'] = best_score
                    state['best_score_metric'] = curr_score_metric
                    state['best_score_epoch'] = human_epoch
                    state['best_checkpoint'] = args.log_dir
                    state['st_internal_external_nonimprove_count'] = 0
                    st_controller.record_external_best_internal_baseline(human_epoch)
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch, ema_model, optimizer=optimizer, best_score=best_score)
                elif bool(getattr(args, 'st_internal_freeze', False)):
                    state['st_internal_external_nonimprove_count'] = (
                        int(state.get('st_internal_external_nonimprove_count', 0) or 0) + 1
                    )
                if bool(getattr(args, 'st_internal_freeze', False)):
                    first_score = state.get('st_internal_external_first_score', None)
                    first_epoch = int(state.get('st_internal_external_first_epoch', 0) or 0)
                    best_epoch = int(state.get('best_score_epoch', 0) or 0)
                    best_score_for_maturity = float(state.get('best_score', best_score))
                    min_improve_pct = max(
                        0.0,
                        float(getattr(args, 'st_internal_external_best_min_improve_pct', 0.0) or 0.0),
                    )
                    min_improve_abs = max(
                        0.0,
                        float(getattr(args, 'st_internal_external_best_min_improve_abs', 0.0) or 0.0),
                    )
                    degrade_pct = max(
                        0.0,
                        float(getattr(args, 'st_internal_external_degrade_pct', 0.0) or 0.0),
                    )
                    degrade_abs = max(
                        0.0,
                        float(getattr(args, 'st_internal_external_degrade_abs', 0.0) or 0.0),
                    )
                    mature_best = False
                    external_degraded = False
                    if first_score is not None and np.isfinite(float(first_score)) and np.isfinite(best_score_for_maturity):
                        first_score = float(first_score)
                        improvement = first_score - best_score_for_maturity
                        required_improvement = max(min_improve_abs, abs(first_score) * min_improve_pct)
                        mature_best = best_epoch > first_epoch and improvement >= required_improvement
                        state['st_internal_external_best_improvement'] = float(improvement)
                        state['st_internal_external_best_required_improvement'] = float(required_improvement)
                    if np.isfinite(curr_score) and np.isfinite(best_score_for_maturity):
                        external_degrade = float(curr_score) - best_score_for_maturity
                        required_degrade = max(degrade_abs, abs(best_score_for_maturity) * degrade_pct)
                        external_degraded = external_degrade >= required_degrade
                        state['st_internal_external_degrade'] = float(external_degrade)
                        state['st_internal_external_required_degrade'] = float(required_degrade)
                    state['st_internal_external_best_mature'] = bool(mature_best)
                    state['st_internal_external_degraded'] = bool(external_degraded)
                optimizer, best_score = st_controller.apply_pending_after_eval(
                    human_epoch,
                    curr_score_metric,
                    curr_score,
                    best_score,
                    external_best_improved,
                    optimizer,
                )
                del scores
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
