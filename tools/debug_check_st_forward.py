import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ImagenTime
from utils.utils_args import parse_args_uncond
from utils.utils_data import gen_dataloader


def _stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    tensor = torch.nan_to_num(tensor.detach())
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"mean={tensor.mean().item():.4e}, std={tensor.std(unbiased=False).item():.4e}, "
        f"min={tensor.min().item():.4e}, max={tensor.max().item():.4e}"
    )


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

    batch = next(iter(train_loader))[0].to(args.device)[: args.batch_size]
    x_img = model.ts_to_img(batch)

    model.eval()
    with torch.no_grad():
        output, weight, sigma = model.forward(x_img, return_sigma=True)

    state = getattr(model.net, "last_st_state", None)
    print(f"use_st_adapter={hasattr(model.net, 'base_net')}, debug_epoch={debug_epoch}")
    _stats("x_ts", batch)
    _stats("x_img", x_img)
    _stats("sigma", sigma)
    _stats("weight", weight)
    _stats("output", output)

    if state is None:
        print("st_state: None")
        return

    _stats("base_img_hat", state.get("base_img_hat"))
    _stats("delta_img", state.get("delta_img"))
    _stats("final_img_hat", state.get("final_img_hat"))
    _stats("base_ts_hat", state.get("base_ts_hat"))
    _stats("delta_ts", state.get("delta_ts"))
    _stats("gate", state.get("gate"))
    _stats("context_gate", state.get("context_gate"))
    _stats("alpha", state.get("alpha"))
    feature_details = state.get("feature_details") or {}
    for key in (
        "st_feature_enabled",
        "st_feature_gate",
        "st_film_norm",
        "st_film_raw_norm",
        "st_film_gated_norm",
        "st_film_scale",
    ):
        _stats(key, feature_details.get(key))

    base_img = state.get("base_img_hat")
    final_img = state.get("final_img_hat")
    if base_img is not None and final_img is not None:
        diff = torch.nan_to_num(final_img - base_img)
        print(f"final_minus_base_rms={diff.square().mean().sqrt().item():.4e}")
        if diff.square().mean().item() == 0.0:
            print("note=zero residual is expected for a freshly initialized zero-init adapter.")
    print(f"has_nan={torch.isnan(output).any().item()}, has_inf={torch.isinf(output).any().item()}")


if __name__ == "__main__":
    main()
