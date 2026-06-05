import scipy
import numpy as np
import torch

from models.ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def _get_context_fid_config(args):
    if args is None:
        return {
            "device": 0,
            "batch_size": 8,
            "lr": 0.001,
            "output_dims": 320,
            "hidden_dims": 64,
            "depth": 10,
            "max_train_length": 3000,
            "temporal_unit": 0,
            "n_iters": None,
        }

    device = getattr(args, "device", 0)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    return {
        "device": device,
        "batch_size": int(getattr(args, "context_fid_batch_size", 8) or 8),
        "lr": float(getattr(args, "context_fid_lr", 0.001) or 0.001),
        "output_dims": int(getattr(args, "context_fid_output_dims", 320) or 320),
        "hidden_dims": int(getattr(args, "context_fid_hidden_dims", 64) or 64),
        "depth": int(getattr(args, "context_fid_depth", 10) or 10),
        "max_train_length": getattr(args, "context_fid_max_train_length", 3000),
        "temporal_unit": int(getattr(args, "context_fid_temporal_unit", 0) or 0),
        "n_iters": getattr(args, "context_fid_ts2vec_iters", None),
    }


def Context_FID(ori_data, generated_data, args=None):
    cfg = _get_context_fid_config(args)
    n_iters = cfg.pop("n_iters")
    if n_iters is not None:
        n_iters = int(n_iters)
        if n_iters <= 0:
            n_iters = None

    model = TS2Vec(input_dims=ori_data.shape[-1], **cfg)
    model.fit(ori_data, n_iters=n_iters, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window='full_series')
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results


def context_fid_score(ori_data, generated_data, args=None):
    ori_data = np.asarray(ori_data, dtype=np.float32)
    generated_data = np.asarray(generated_data, dtype=np.float32)
    return Context_FID(ori_data, generated_data, args=args)
