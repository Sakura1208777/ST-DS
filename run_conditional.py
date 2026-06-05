import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.multiprocessing
import logging
import torch.nn.functional as F
from tqdm import tqdm

from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import ImagenTime
from models.sampler import DiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, get_x_and_mask, latest_checkpoint_path, save_eval_results
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond

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

        # --- set-up data and device ---
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = ImagenTime(args=args, device=args.device).to(args.device)

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
        best_score = state.get('best_score', float('inf'))  # marginal score for long-range metrics, dice score for short-range metrics
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
                mask_ts, x_ts = get_x_and_mask(args, data)

                # transform to image
                x_ts_img = model.ts_to_img(x_ts)
                # pad mask with 1
                mask_ts_img = model.ts_to_img(mask_ts,pad_val=1)
                optimizer.zero_grad()
                loss_result = model.loss_fn_impute(x_ts_img, mask_ts_img)
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
                mse = 0
                mae = 0
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        tqdm.write(f"eval imputation: epoch {human_epoch:04d}/{args.epochs}")
                        eval_bar = tqdm(test_loader, total=len(test_loader), leave=True,
                                        bar_format=eval_bar_format)
                        for idx, data in enumerate(eval_bar, 1):
                            mask_ts, x_ts = get_x_and_mask(args, data)

                            # transform to image
                            x_ts_img = model.ts_to_img(x_ts)
                            mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)

                            # sample from the model
                            # and impute, both interpolation and extrapolation are similar just the mask is different
                            x_img_sampled = process.interpolate(x_ts_img, mask_ts_img).to(x_ts_img.device)
                            x_ts_sampled = model.img_to_ts(x_img_sampled)

                            # task evaluation
                            mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                            mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                            mse += mse_mean.item()
                            mae += mae_mean.item()

                scores = {'mse': mse / (idx + 1), 'mae': mae / (idx + 1)}
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, human_epoch)
                save_eval_results(args, human_epoch, scores)

                # --- save checkpoint ---
                curr_score = scores['mse']
                if curr_score < best_score:
                    best_score = curr_score
                    state['best_score'] = best_score
                    ema_model = model.model_ema if args.ema else None
                    save_checkpoint(args.log_dir, state, epoch, ema_model,
                                    optimizer=optimizer, best_score=best_score)

            if getattr(args, 'save_latest', True):
                state['best_score'] = best_score
                ema_model = model.model_ema if args.ema else None
                save_checkpoint(latest_checkpoint_path(args.log_dir), state, epoch, ema_model,
                                optimizer=optimizer, best_score=best_score)

        logging.info("Training is complete")


if __name__ == '__main__':
    args = parse_args_cond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
