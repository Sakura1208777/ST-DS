import os
import numpy as np
import torch
from metrics import evaluate_model_uncond
from utils.loggers import CompositeLogger, NeptuneLogger, PrintLogger
from utils.utils_args import parse_args_uncond
from models.model import ImagenTime
from models.sampler import DiffusionProcess
import logging
from utils.utils_data import gen_dataloader
from utils.utils import create_model_name_and_dir, restore_state, log_config_and_tags, save_eval_results, \
    get_run_view_dir, latest_checkpoint_path
from utils.utils_vis import prepare_data, PCA_plot, TSNE_plot, density_plot, jensen_shannon_divergence
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# matplotlib.use('Agg')


def main(args):
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        name = create_model_name_and_dir(args, new_run=False)
        args.view_dir = get_run_view_dir(args, create=False)
        log_config_and_tags(args, logger, name)
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args) # combine train and test loaders?
        model = ImagenTime(args=args, device=args.device).to(args.device)
        if args.use_stft:
            model.init_stft_embedder(train_loader)
        else:
            _ = model.ts_to_img(next(iter(train_loader))[0].to(args.device)) # initialize delay embedder

        # restore checkpoint — prefer best, fall back to latest
        state = dict(model=model, epoch=0)
        ema_model = model.model_ema if args.ema else None
        best_ckpt = args.log_dir
        eval_epoch = 'latest'
        eval_checkpoint_path = None
        if os.path.exists(best_ckpt):
            args.resume_checkpoint = best_ckpt
            eval_epoch = 'best'
            eval_checkpoint_path = best_ckpt
        elif os.path.exists(latest_checkpoint_path(best_ckpt)):
            eval_checkpoint_path = latest_checkpoint_path(best_ckpt)
        restore_state(args, state, ema_model=ema_model)
        if eval_checkpoint_path is None:
            eval_checkpoint_path = getattr(args, 'resume_checkpoint', None)
        loaded_epoch = int(state.get('display_epoch', state.get('epoch', 0) + 1))
        model.epoch = loaded_epoch
        if hasattr(model.net, 'current_epoch'):
            model.net.current_epoch = loaded_epoch
        args.eval_checkpoint_type = eval_epoch
        args.eval_checkpoint_path = eval_checkpoint_path

        gen_sig = []
        real_sig = []
        model.eval()
        with torch.no_grad():
            with model.ema_scope():
                process = DiffusionProcess(args, model.net,
                                           (args.input_channels, args.img_resolution, args.img_resolution))
                eval_bar_format = "{percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                tqdm.write(f"eval sampling: checkpoint {eval_epoch}")
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
        ori_sig = np.vstack(real_sig)
        logging.info("Data generation is complete")
        prep_ori, prep_gen, sample_num = prepare_data(ori_sig, gen_sig)

        # PCA Analysis
        PCA_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Do t-SNE Analysis together
        TSNE_plot(prep_ori, prep_gen, sample_num, logger, args)
        # Density plot
        density_plot(prep_ori, prep_gen, logger, args)
        # jensen shannon divergence
        jensen_shannon_divergence(prep_ori, prep_gen, logger)

        scores = evaluate_model_uncond(ori_sig, gen_sig, args)
        for key, value in scores.items():
            logger.log(f'test/{key}', value, eval_epoch)
        save_eval_results(args, eval_epoch, scores)


if __name__ == '__main__':
    args = parse_args_uncond()  # load unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
