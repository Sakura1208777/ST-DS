import os
import random
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
        train_bar_format = "{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        eval_bar_format = "{percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        for epoch in range(init_epoch, args.epochs):
            human_epoch = epoch + 1
            model.train()
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

            # --- evaluation loop ---
            should_evaluate = (human_epoch % args.logging_iter == 0) or (human_epoch == args.epochs)
            if should_evaluate:
                scores = evaluate_unconditional_epoch(
                    args,
                    model,
                    test_loader,
                    logger,
                    human_epoch,
                    eval_bar_format,
                )

                # --- save checkpoint ---
                curr_score_metric = select_best_score_metric(scores)
                curr_score = scores[curr_score_metric]
                prev_best_score = best_score
                external_best_improved = curr_score < prev_best_score
                if external_best_improved:
                    best_score = curr_score
                    state['best_score'] = best_score
                    state['best_score_metric'] = curr_score_metric
                    state['best_score_epoch'] = human_epoch
                    state['best_checkpoint'] = args.log_dir
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch, ema_model, optimizer=optimizer, best_score=best_score)
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
    # Set all RNGs used in this project. np.random.default_rng(args.seed) only
    # creates a local generator and does not seed np.random.* calls used by the
    # dataloaders and metrics.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
