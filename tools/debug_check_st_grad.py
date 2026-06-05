import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ImagenTime
from utils.utils_args import parse_args_uncond
from utils.utils_data import gen_dataloader


def _grad_norm(module):
    total = 0.0
    count = 0
    missing = 0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing += 1
            continue
        total += float(param.grad.detach().square().sum().item())
        count += 1
    return total ** 0.5, count, missing


def _print_grad(name, module):
    if module is None:
        print(f"{name}: None")
        return
    norm, count, missing = _grad_norm(module)
    print(f"{name}: grad_norm={norm:.4e}, grad_tensors={count}, missing_grad_tensors={missing}")


def main():
    args = parse_args_uncond()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = min(int(args.batch_size), 8)
    train_loader, _ = gen_dataloader(args)

    model = ImagenTime(args=args, device=args.device).to(args.device)
    if args.use_stft:
        model.init_stft_embedder(train_loader)

    debug_epoch = max(1, min(int(args.epochs), int(getattr(args, "st_warmup_epochs", 1) or 1)))
    model.epoch = debug_epoch
    if hasattr(model.net, "current_epoch"):
        model.net.current_epoch = debug_epoch

    model.train()
    batch = next(iter(train_loader))[0].to(args.device)[: args.batch_size]
    x_img = model.ts_to_img(batch)
    loss_result = model.loss_fn(x_img, x_ts=batch)
    loss, logs = loss_result if isinstance(loss_result, tuple) else (loss_result, {})
    loss.backward()

    print(f"use_st_adapter={hasattr(model.net, 'base_net')}, debug_epoch={debug_epoch}")
    print(f"loss={torch.nan_to_num(loss.detach()).item():.4e}")
    for key in (
        "karras loss",
        "ts loss",
        "ds/total_loss",
        "st/residual_loss",
        "st/effective_delta_norm",
        "st/st_feature_enabled",
        "st/st_feature_gate",
        "st/st_film_norm",
        "st/st_film_raw_norm",
        "st/st_film_gated_norm",
        "st/st_film_scale",
        "st/raw_delta_reg",
        "total loss",
    ):
        if key in logs:
            print(f"{key}={float(logs[key]):.4e}")

    base_net = getattr(model.net, "base_net", model.net)
    st_denoiser = getattr(model.net, "st_denoiser", None)
    feature_conditioner = getattr(model.net, "feature_conditioner", None)
    _print_grad("base_net", base_net)
    _print_grad("feature_conditioner", feature_conditioner)
    _print_grad("st_denoiser", st_denoiser)
    if st_denoiser is not None:
        _print_grad("trend_branch", getattr(st_denoiser, "trend_blocks", None))
        _print_grad("season_branch", getattr(st_denoiser, "season_blocks", None))
        _print_grad("period_branch", getattr(st_denoiser, "period_modulator", None))
        _print_grad("trend_out", getattr(st_denoiser, "trend_out", None))
        _print_grad("season_out", getattr(st_denoiser, "season_out", None))
        print("note=with zero-init heads, early gradients first appear on output heads; inner branches wake up after heads move.")


if __name__ == "__main__":
    main()
