from metrics.metrics_long_range import compute_all_metrics, setup_optimizer
import gc
import numpy as np
import torch
from tqdm import tqdm


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(np.mean(values), 4), np.round(np.std(values), 4)


def _mean_only(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(np.mean(values), 4)


def _eval_progress(total, message):
    tqdm.write(message)
    return tqdm(
        total=total,
        ncols=80,
        leave=True,
        dynamic_ncols=False,
        bar_format="{percentage:3.0f}%| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _cleanup_metric_cache(device=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model_uncond(real_sig,gen_sig,args):
    """
    Args:
        real_sig: real signal
        gen_sig: generated signal
        args: args
    Returns:
        marginal score if long-term dataset, discrimin
    this function evaluates the model based on the dataset used:
    for short-term datasets(eg. sine, stock) it uses discriminative_torch.py and predictive_torch.py
    for long-term datasets(eg. fred_md) it uses metrics_long_range.py


    """

    if args.dataset in ['stock', 'stocks', 'sine', 'mujoco', 'energy']:
        # proceed with short term evaluation
        metric_iteration = 10
        extra_metric_iteration = max(1, int(getattr(args, 'st_eval_extra_metric_iteration', 1)))
        total_steps = metric_iteration * 2
        if getattr(args, 'eval_cross_corr', True):
            total_steps += extra_metric_iteration
        if getattr(args, 'eval_context_fid', True):
            total_steps += extra_metric_iteration

        from metrics.discriminative_torch import discriminative_score_metrics
        ## for deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with _eval_progress(total_steps, "eval metrics: discriminative, predictive, and enabled extras") as pbar:
            disc_res = []
            for _ in range(metric_iteration):
                pbar.set_postfix_str("disc")
                dsc = discriminative_score_metrics(real_sig, gen_sig, args)
                disc_res.append(dsc)
                pbar.update(1)
                _cleanup_metric_cache(getattr(args, 'device', None))
            disc_mean, disc_std = _mean_std(disc_res)
            from metrics.predictive_metrics import predictive_score_metrics
            predictive_score = list()
            for _ in range(metric_iteration):
                pbar.set_postfix_str("pred")
                temp_pred = predictive_score_metrics(real_sig, gen_sig, args)
                predictive_score.append(temp_pred)
                pbar.update(1)
                _cleanup_metric_cache(getattr(args, 'device', None))
            pred_mean, pred_std = _mean_std(predictive_score)

            scores = {'disc_mean': disc_mean, 'disc_std': disc_std,
                      'pred_mean': pred_mean, 'pred_std': pred_std}

            if getattr(args, 'eval_cross_corr', True):
                from metrics.cross_correlation import cross_correlation_score
                device = getattr(args, 'device', 'cuda:0')
                cross_corr = []
                for _ in range(extra_metric_iteration):
                    pbar.set_postfix_str("corr")
                    cross_corr.append(cross_correlation_score(real_sig, gen_sig, device=device))
                    pbar.update(1)
                    _cleanup_metric_cache(device)
                scores['cross_corr_mean'] = _mean_only(cross_corr)

            if getattr(args, 'eval_context_fid', True):
                from metrics.context_fid import context_fid_score
                context_fid = []
                for _ in range(extra_metric_iteration):
                    pbar.set_postfix_str("fid")
                    context_fid.append(context_fid_score(real_sig, gen_sig, args))
                    pbar.update(1)
                    _cleanup_metric_cache(getattr(args, 'device', None))
                scores['context_fid_mean'] = _mean_only(context_fid)

        return scores

    else:
        # proceed with long term evaluation
        # conversion to meet benchmark requirements:
        real_sig,gen_sig = torch.Tensor(real_sig).float(),torch.Tensor(gen_sig).float()
        with _eval_progress(1, "eval metrics: long-range metrics") as pbar:
            pbar.set_postfix_str("metrics")
            scores = compute_all_metrics(
                real_sig,
                gen_sig,
                setup_optimizer,
                torch.nn.Sigmoid() if args.dataset == 'temperature_rain' else torch.nn.Identity(),
                args.device,
            )
            pbar.update(1)
            _cleanup_metric_cache(args.device)
        return scores
