import numpy as np
import os
import sys
import subprocess
import logging
import torch
import csv
import json
from datetime import datetime


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def latest_checkpoint_path(ckpt_path):
    return f"{ckpt_path}.latest"


def _to_serializable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _current_run_command():
    if not sys.argv:
        return ''
    return 'python ' + subprocess.list2cmdline(sys.argv)


def _make_run_id(log_root, dataset, run_name):
    date_str = datetime.now().strftime('%Y%m%d')
    parent = os.path.join(log_root, dataset, run_name)
    os.makedirs(parent, exist_ok=True)
    existing = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)) and d.startswith(date_str)]
    seq = 1
    if existing:
        nums = []
        for d in existing:
            parts = d.split('-')
            if len(parts) == 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
        if nums:
            seq = max(nums) + 1
    return f'{date_str}-{seq:06d}'


def _find_latest_run_id(log_root, dataset, run_name):
    parent = os.path.join(log_root, dataset, run_name)
    if not os.path.isdir(parent):
        return None
    candidates = []
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if not os.path.isdir(path):
            continue
        ckpt_path = os.path.join(path, 'checkpoint')
        latest_path = latest_checkpoint_path(ckpt_path)
        if os.path.exists(latest_path):
            candidates.append((os.path.getmtime(latest_path), name))
        elif os.path.exists(ckpt_path):
            candidates.append((os.path.getmtime(ckpt_path), name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _normalize_eval_epoch(epoch):
    try:
        return int(epoch)
    except (TypeError, ValueError):
        return str(epoch)


def _eval_epoch_sort_key(row):
    epoch = row.get('epoch', 0)
    timestamp = str(row.get('timestamp', ''))
    if isinstance(epoch, str):
        epoch_text = epoch.strip().lower()
        if epoch_text == 'last':
            return (1, 0, '', timestamp)
        try:
            return (0, int(epoch_text), '', timestamp)
        except ValueError:
            return (0, 10 ** 12, epoch_text, timestamp)
    try:
        return (0, int(epoch), '', timestamp)
    except (TypeError, ValueError):
        return (0, 10 ** 12, str(epoch), timestamp)


def _write_aligned_eval_table(path, rows, fieldnames):
    text_rows = []
    for row in rows:
        text_rows.append({
            name: '' if row.get(name, '') is None else str(row.get(name, ''))
            for name in fieldnames
        })
    widths = {
        name: max([len(name)] + [len(row.get(name, '')) for row in text_rows])
        for name in fieldnames
    }
    lines = ['  '.join(name.ljust(widths[name]) for name in fieldnames)]
    for row in text_rows:
        lines.append('  '.join(row.get(name, '').ljust(widths[name]) for name in fieldnames))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _load_eval_jsonl_rows(path, initial_fieldnames):
    rows = []
    fieldnames = list(initial_fieldnames)
    if not os.path.exists(path):
        return rows, fieldnames
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row.pop('run_name', None)
            rows.append(row)
            for name in row.keys():
                if name not in fieldnames:
                    fieldnames.append(name)
    rows.sort(key=_eval_epoch_sort_key)
    return rows, fieldnames


def save_checkpoint(ckpt_dir, state, epoch, ema_model=None, optimizer=None, best_score=None):
    saved_state = {
        'epoch': epoch,
        'display_epoch': epoch + 1,
        'model': state['model'].state_dict(),
    }
    for meta_key in ('run_id', 'run_name', 'dataset',
                     'best_score_metric', 'best_score_epoch', 'best_checkpoint'):
        if meta_key in state:
            saved_state[meta_key] = state[meta_key]
    for f3_key in ('st_frozen', 'st_degrade_count', 'st_best_disc', 'st_best_disc_std', 'st_best_epoch',
                   'st_best_checkpoint', 'st_frozen_epoch', 'st_post_freeze_degrade',
                   'st_health_ema_value', 'st_health_reliability_ema', 'st_health_alignment_ema',
                   'st_health_delta_ratio_ema', 'st_health_highfreq_leak_ema',
                   'st_health_saturation_ema', 'st_health_base_ts_mse_ema',
                   'st_health_final_ts_mse_ema', 'st_health_final_base_mse_ratio_ema',
                   'st_health_effective_delta_norm_ema', 'st_health_best', 'st_health_best_epoch',
                   'st_external_best_delta_ratio_ema', 'st_external_best_highfreq_leak_ema',
                   'st_external_best_final_base_mse_ratio_ema', 'st_external_best_epoch',
                   'st_external_best_delta_growth_ratio', 'st_external_best_highfreq_growth_ratio',
                   'st_external_best_drift_active', 'st_internal_external_nonimprove_count',
                   'st_internal_external_metric', 'st_internal_external_score',
                   'st_internal_external_best_score',
                   'st_health_best_checkpoint', 'st_health_degrade_count',
                   'st_health_degrade_reason', 'st_freeze_reason'):
        if f3_key in state:
            saved_state[f3_key] = state[f3_key]
    if ema_model is not None:
        saved_state['ema_model'] = ema_model.state_dict()
    optimizer = optimizer if optimizer is not None else state.get('optimizer', None)
    if optimizer is not None:
        saved_state['optimizer'] = optimizer.state_dict()
    if best_score is not None:
        saved_state['best_score'] = float(best_score)
    elif 'best_score' in state:
        saved_state['best_score'] = float(state['best_score'])
    tmp_path = ckpt_dir + '.tmp'
    torch.save(saved_state, tmp_path)
    os.replace(tmp_path, ckpt_dir)


def _load_state_dict_checked(module, state_dict, name, allow_partial=False):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        msg = (
            f"{name} checkpoint keys do not match current model. "
            f"missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
        if len(missing) > 12 or len(unexpected) > 12:
            msg += " ..."
        if not allow_partial:
            raise RuntimeError(msg + " Use --allow_partial_resume true only when this mismatch is intentional.")
        logging.warning(msg)


def restore_checkpoint(ckpt_dir, state, device='cuda:0', ema_model=None, optimizer=None, allow_partial=False):
    state['_restored'] = False
    if not os.path.exists(ckpt_dir):
        ckpt_parent = os.path.dirname(ckpt_dir)
        if ckpt_parent:
            os.makedirs(ckpt_parent, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        state['epoch'] = loaded_state['epoch']
        _load_state_dict_checked(state['model'], loaded_state['model'], 'model', allow_partial=allow_partial)
        if 'ema_model' in loaded_state and ema_model is not None:
            _load_state_dict_checked(ema_model, loaded_state['ema_model'], 'ema_model',
                                     allow_partial=allow_partial)
        optimizer = optimizer if optimizer is not None else state.get('optimizer', None)
        if 'optimizer' in loaded_state and optimizer is not None:
            try:
                optimizer.load_state_dict(loaded_state['optimizer'])
            except ValueError as exc:
                logging.warning(f"Skipped optimizer state because checkpoint parameters changed: {exc}")
        if 'best_score' in loaded_state:
            state['best_score'] = loaded_state['best_score']
        if 'display_epoch' in loaded_state:
            state['display_epoch'] = loaded_state['display_epoch']
        for meta_key in ('run_id', 'run_name', 'dataset',
                         'best_score_metric', 'best_score_epoch', 'best_checkpoint'):
            if meta_key in loaded_state:
                state[meta_key] = loaded_state[meta_key]
        for f3_key in ('st_frozen', 'st_degrade_count', 'st_best_disc', 'st_best_disc_std', 'st_best_epoch',
                       'st_best_checkpoint', 'st_frozen_epoch', 'st_post_freeze_degrade',
                       'st_health_ema_value', 'st_health_reliability_ema', 'st_health_alignment_ema',
                       'st_health_delta_ratio_ema', 'st_health_highfreq_leak_ema',
                       'st_health_saturation_ema', 'st_health_base_ts_mse_ema',
                       'st_health_final_ts_mse_ema', 'st_health_final_base_mse_ratio_ema',
                       'st_health_effective_delta_norm_ema', 'st_health_best', 'st_health_best_epoch',
                       'st_external_best_delta_ratio_ema', 'st_external_best_highfreq_leak_ema',
                       'st_external_best_final_base_mse_ratio_ema', 'st_external_best_epoch',
                       'st_health_best_checkpoint', 'st_health_degrade_count',
                       'st_health_degrade_reason', 'st_freeze_reason'):
            if f3_key in loaded_state:
                state[f3_key] = loaded_state[f3_key]
        state['_restored'] = True
        logging.info(f'Successfully loaded previous state from {ckpt_dir}')
        return state


def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    logger.add_tags([args.dataset])


def create_model_name_and_dir(args, new_run=False):
    dataset = getattr(args, 'dataset', 'energy')
    train_budget = getattr(args, 'train_budget', None)
    name = dataset
    if train_budget:
        name += f'-{train_budget}'
    log_root = getattr(args, 'log_root', getattr(args, 'log_dir', './logs'))
    args.log_root = log_root
    args.run_name = name
    run_id = getattr(args, 'run_id', None)
    if run_id in ('', 'none', 'None'):
        run_id = None
    if run_id is None:
        if new_run:
            run_id = _make_run_id(log_root, args.dataset, name)
        else:
            run_id = _find_latest_run_id(log_root, args.dataset, name)
    args.run_id = run_id
    if run_id:
        args.log_dir = os.path.join(log_root, args.dataset, name, run_id, 'checkpoint')
    else:
        args.log_dir = os.path.join(log_root, args.dataset, name)
    os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
    args.view_root = getattr(args, 'view_root', getattr(args, 'view_dir', './view'))
    args.view_dir = get_run_view_dir(args, create=False)
    return name


def get_run_view_dir(args, create=False):
    view_root = getattr(args, 'view_root', None)
    if view_root is None:
        view_dir = getattr(args, 'view_dir', './view')
        run_name = getattr(args, 'run_name', '')
        dataset = getattr(args, 'dataset', '')
        run_id = getattr(args, 'run_id', None)
        if run_id:
            suffix = os.path.join(dataset, run_name, run_id) if dataset and run_name else ''
        else:
            suffix = os.path.join(dataset, run_name) if dataset and run_name else ''
        view_root = view_dir
        norm_view_dir = os.path.normpath(view_dir)
        if suffix and norm_view_dir.endswith(os.path.normpath(suffix)):
            levels = 3 if run_id else 2
            view_root = view_dir
            for _ in range(levels):
                view_root = os.path.dirname(view_root)
        elif run_id and os.path.basename(norm_view_dir) == run_id:
            run_parent = os.path.dirname(view_dir)
            dataset_parent = os.path.dirname(run_parent)
            if run_name and os.path.basename(os.path.normpath(run_parent)) == run_name:
                if dataset and os.path.basename(os.path.normpath(dataset_parent)) == dataset:
                    view_root = os.path.dirname(dataset_parent)
                else:
                    view_root = dataset_parent
        elif run_name and os.path.basename(norm_view_dir) == run_name:
            parent_dir = os.path.dirname(view_dir)
            if dataset and os.path.basename(os.path.normpath(parent_dir)) == dataset:
                view_root = os.path.dirname(parent_dir)
            else:
                view_root = parent_dir
    dataset = getattr(args, 'dataset', '')
    run_name = getattr(args, 'run_name', '')
    run_id = getattr(args, 'run_id', None)
    if run_id:
        view_dir = os.path.join(view_root, dataset, run_name, run_id)
    else:
        view_dir = os.path.join(view_root, dataset, run_name)
    if create:
        os.makedirs(view_dir, exist_ok=True)
    args.view_root = view_root
    args.view_dir = view_dir
    return view_dir


def restore_state(args, state, ema_model=None, optimizer=None):
    resume_path = getattr(args, 'resume_checkpoint', None)
    if resume_path is None and getattr(args, 'resume_latest', True):
        latest_path = latest_checkpoint_path(args.log_dir)
        resume_path = latest_path if os.path.exists(latest_path) else args.log_dir
    elif resume_path is None:
        resume_path = args.log_dir
    if os.path.isdir(resume_path) or not os.path.exists(resume_path):
        logging.info("No checkpoint found at {} — starting fresh".format(resume_path))
        return 0
    logging.info("restoring checkpoint from: {}".format(resume_path))
    restore_checkpoint(resume_path, state, device=getattr(args, 'device', 'cuda:0'),
                       ema_model=ema_model, optimizer=optimizer,
                       allow_partial=getattr(args, 'allow_partial_resume', False))
    if state.get('run_id') and not getattr(args, 'run_id', None):
        args.run_id = state['run_id']
        log_root = getattr(args, 'log_root', getattr(args, 'log_dir', './logs'))
        args.log_dir = os.path.join(log_root, getattr(args, 'dataset', ''), getattr(args, 'run_name', ''),
                                    args.run_id, 'checkpoint')
        os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
        args.view_dir = get_run_view_dir(args, create=False)
    init_epoch = state['epoch'] + 1 if state.get('_restored', False) else state['epoch']
    return init_epoch


def save_eval_results(args, epoch, scores):
    record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset': getattr(args, 'dataset', ''),
        'epoch': _normalize_eval_epoch(epoch),
    }
    record.update({
        key: _to_serializable(value)
        for key, value in scores.items()
    })
    record['eval_command'] = getattr(args, 'eval_command', None) or _current_run_command()
    checkpoint_type = getattr(args, 'eval_checkpoint_type', None)
    if checkpoint_type is not None:
        record['checkpoint_type'] = _to_serializable(checkpoint_type)
    checkpoint_path = getattr(args, 'eval_checkpoint_path', None)
    if checkpoint_path is not None:
        record['checkpoint_path'] = _to_serializable(checkpoint_path)

    view_dir = get_run_view_dir(args, create=True)

    jsonl_path = os.path.join(view_dir, 'evaluation.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=True) + '\n')

    csv_path = os.path.join(view_dir, 'evaluation.csv')
    fieldnames = list(record.keys())
    rows, fieldnames = _load_eval_jsonl_rows(jsonl_path, fieldnames)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_aligned_eval_table(
        os.path.join(view_dir, 'evaluation_aligned.txt'),
        rows,
        fieldnames,
    )
    logging.info(f"Saved evaluation results to {view_dir}")


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)


# --- extrapolation and interpolation --- #
# get the mask and x for the time series
def get_x_and_mask(args, data):
    if args.dataset in ['climate', 'physionet']:
        # in the case of these datasets, the 'data_to_predict' is the same as 'observed_data
        if args.task == 'extrapolation':
            # concat the observed and predicted data
            x_ts = torch.cat([data['observed_data'], data['data_to_predict']], dim=1).to(args.device)
            # the predicted mask is opposite. the 1s are observed in the mask so it needed to be flipped in our case
            mask_ts = torch.cat([data['observed_mask'],  1 - data['mask_predicted_data']], dim=1).to(args.device)
        else:
            x_ts = data['observed_data'].to(args.device)
            mask_ts = data['mask_predicted_data'].to(args.device)
    else:
        if args.task == 'extrapolation':
            x_ts = data[0].float().to(args.device)
            # half ones and half zeros
            mask_ts = torch.zeros_like(x_ts)
            mask_ts[:, :x_ts.shape[1] // 2] = 1
        else:
            x_ts = data[0].float().to(args.device)
            # --- generate random mask and mask x as it time series --- #
            B, T, N = x_ts.shape
            mask_ts = torch.rand((B, T, N)).to(args.device)
            mask_ts[mask_ts <= args.mask_rate] = 0  # masked
            mask_ts[mask_ts > args.mask_rate] = 1  # remained

    return mask_ts, x_ts
