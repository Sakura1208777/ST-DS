from omegaconf import OmegaConf
import argparse
import os
import sys


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _cli_has_arg(name):
    variants = {f"--{name}", f"--{name.replace('_', '-')}"}
    for token in sys.argv[1:]:
        if token in variants:
            return True
        if any(token.startswith(f"{variant}=") for variant in variants):
            return True
    return False


def _add_st_ds_args(parser):
    # Generic ST-DS-ImagenTime options. Defaults are filled after the YAML is loaded
    # so config files can still override them without per-dataset duplication.
    parser.add_argument('--use_st_adapter', type=_str2bool, default=None)
    parser.add_argument('--st_channels', type=int, default=None)
    parser.add_argument('--st_res_layers', type=int, default=None)
    parser.add_argument('--st_nheads', type=int, default=None)
    parser.add_argument('--st_freq_tier', type=int, default=None)
    parser.add_argument('--st_alpha', type=float, default=None)
    parser.add_argument('--st_alpha_max', type=float, default=None)
    parser.add_argument('--st_alpha_learnable', type=_str2bool, default=None)
    parser.add_argument('--st_zero_init', type=_str2bool, default=None)
    parser.add_argument('--st_sigma_gate', type=_str2bool, default=None)
    parser.add_argument('--st_context_sigma_gate', type=_str2bool, default=None)
    parser.add_argument('--st_feature_fusion', type=_str2bool, default=None)
    parser.add_argument('--st_feature_channels', type=int, default=None)
    parser.add_argument('--st_feature_scale_max', type=float, default=None)
    parser.add_argument('--st_feature_init_scale', type=float, default=None)
    parser.add_argument('--st_feature_zero_init', type=_str2bool, default=None)
    parser.add_argument('--st_feature_sigma_gate', type=_str2bool, default=None)
    parser.add_argument('--st_feature_sigma_mid', type=float, default=None)
    parser.add_argument('--st_feature_sigma_scale', type=float, default=None)
    parser.add_argument('--st_feature_warmup_epochs', type=int, default=None)
    parser.add_argument('--st_feature_norm_max', type=float, default=None)
    parser.add_argument('--st_feature_input_clip', type=float, default=None)
    parser.add_argument('--st_trust_gate', type=_str2bool, default=None)
    parser.add_argument('--st_trust_init', type=float, default=None)
    parser.add_argument('--st_trust_max', type=float, default=None)
    parser.add_argument('--st_trust_learnable', type=_str2bool, default=None)
    parser.add_argument('--st_sigma_mid', type=float, default=None)
    parser.add_argument('--st_sigma_scale', type=float, default=None)
    parser.add_argument('--st_warmup_epochs', type=int, default=None)
    parser.add_argument('--st_residual_calib', type=_str2bool, default=None)
    parser.add_argument('--st_residual_warmup_epochs', type=int, default=None)
    parser.add_argument('--st_residual_huber_beta', type=float, default=None)
    parser.add_argument('--st_residual_target_scale', type=float, default=None)
    parser.add_argument('--st_effective_align', type=_str2bool, default=None)
    parser.add_argument('--st_effective_huber_beta', type=float, default=None)
    parser.add_argument('--st_effective_max_ratio', type=float, default=None)
    parser.add_argument('--st_lma_affine', type=_str2bool, default=None)
    parser.add_argument('--st_period_branch', type=_str2bool, default=None)
    parser.add_argument('--st_period_candidates', type=int, nargs='+', default=None)
    parser.add_argument('--st_period_min', type=int, default=None)
    parser.add_argument('--st_period_max', type=int, default=None)
    parser.add_argument('--st_period_temperature', type=float, default=None)
    parser.add_argument('--st_period_max_scale', type=float, default=None)
    parser.add_argument('--st_period_input_condition', type=_str2bool, default=None)
    parser.add_argument('--st_period_input_channels', type=int, default=None)
    parser.add_argument('--st_period_input_alpha', type=float, default=None)
    parser.add_argument('--st_period_input_alpha_max', type=float, default=None)
    parser.add_argument('--st_period_input_alpha_learnable', type=_str2bool, default=None)
    parser.add_argument('--st_period_input_warmup_epochs', type=int, default=None)
    parser.add_argument('--st_period_input_max_scale', type=float, default=None)
    parser.add_argument('--st_period_input_init_scale', type=float, default=None)
    parser.add_argument('--st_detach_base_for_style', type=_str2bool, default=None)
    parser.add_argument('--st_branch_style_calib', type=_str2bool, default=None)
    parser.add_argument('--st_var_relation', type=_str2bool, default=None)
    parser.add_argument('--st_var_relation_rank', type=int, default=None)
    parser.add_argument('--st_var_relation_beta', type=float, default=None)
    parser.add_argument('--st_var_relation_init_beta', type=float, default=None)
    parser.add_argument('--st_var_relation_no_self', type=_str2bool, default=None)
    parser.add_argument('--st_dropout', type=float, default=None)
    parser.add_argument('--st_input_noise', type=float, default=None)
    parser.add_argument('--st_freeze', type=_str2bool, default=None)
    parser.add_argument('--st_freeze_warmup', type=int, default=None)
    parser.add_argument('--st_freeze_patience', type=int, default=None)
    parser.add_argument('--st_freeze_threshold', type=float, default=None)
    parser.add_argument('--st_freeze_hard_threshold', type=float, default=None)
    parser.add_argument('--st_freeze_pct_threshold', type=float, default=None)
    parser.add_argument('--st_freeze_min_abs', type=float, default=None)
    parser.add_argument('--st_freeze_hard_pct_threshold', type=float, default=None)
    parser.add_argument('--st_freeze_hard_min_abs', type=float, default=None)
    parser.add_argument('--st_freeze_std_threshold', type=float, default=None)
    parser.add_argument('--st_freeze_std_ratio', type=float, default=None)
    parser.add_argument('--st_freeze_std_min_abs', type=float, default=None)
    parser.add_argument('--st_freeze_lr_ratio', type=float, default=None)
    parser.add_argument('--st_freeze_watch', type=_str2bool, default=None)
    parser.add_argument('--st_freeze_watch_interval', type=int, default=None)
    parser.add_argument('--st_freeze_watch_patience', type=int, default=None)
    parser.add_argument('--st_post_freeze_patience', type=int, default=None)
    parser.add_argument('--st_post_freeze_threshold', type=float, default=None)
    parser.add_argument('--st_post_freeze_hard_threshold', type=float, default=None)
    parser.add_argument('--st_post_freeze_std_threshold', type=float, default=None)
    parser.add_argument('--st_post_freeze_std_ratio', type=float, default=None)
    parser.add_argument('--use_late_decay', type=_str2bool, default=None)
    parser.add_argument('--late_decay_start_ratio', type=float, default=None)
    parser.add_argument('--late_decay_start_epoch', type=int, default=None)
    parser.add_argument('--late_decay_min_scale', type=float, default=None)
    parser.add_argument('--late_decay_power', type=float, default=None)
    parser.add_argument('--late_decay_st_strength', type=_str2bool, default=None)
    parser.add_argument('--late_decay_style_loss', type=_str2bool, default=None)
    parser.add_argument('--use_structured_st_target', type=_str2bool, default=None)
    parser.add_argument('--structured_target_kernels', type=int, nargs='+', default=None)
    parser.add_argument('--structured_target_blend_start_epoch', type=int, default=None)
    parser.add_argument('--structured_target_blend_end_epoch', type=int, default=None)
    parser.add_argument('--structured_target_blend_start_ratio', type=float, default=None)
    parser.add_argument('--structured_target_blend_window', type=float, default=None)
    parser.add_argument('--structured_target_max_blend', type=float, default=None)
    parser.add_argument('--structured_adaptive_blend', type=_str2bool, default=None)
    parser.add_argument('--structured_confidence_min', type=float, default=None)
    parser.add_argument('--structured_confidence_power', type=float, default=None)
    parser.add_argument('--structured_trend_weight', type=float, default=None)
    parser.add_argument('--structured_season_weight', type=float, default=None)
    parser.add_argument('--structured_late_season_min', type=float, default=None)
    parser.add_argument('--structured_late_start_ratio', type=float, default=None)
    parser.add_argument('--structured_target_norm_ratio', type=float, default=None)
    parser.add_argument('--structured_target_norm_ratio_final', type=float, default=None)
    parser.add_argument('--structured_target_norm_switch', type=_str2bool, default=None)
    parser.add_argument('--use_residual_reliability', type=_str2bool, default=None)
    parser.add_argument('--residual_reliability_min', type=float, default=None)
    parser.add_argument('--residual_reliability_power', type=float, default=None)
    parser.add_argument('--residual_reliability_kernels', type=int, nargs='+', default=None)
    parser.add_argument('--residual_reliability_freq_topk', type=int, default=None)
    parser.add_argument('--residual_reliability_acf_max_lag', type=int, default=None)
    parser.add_argument('--reliability_delta_reg_boost', type=float, default=None)
    parser.add_argument('--reliability_effective_boost', type=float, default=None)
    parser.add_argument('--st_internal_health', type=_str2bool, default=None)
    parser.add_argument('--st_internal_health_kernel', type=int, default=None)
    parser.add_argument('--st_internal_freeze', type=_str2bool, default=None)
    parser.add_argument('--st_internal_freeze_warmup_ratio', type=float, default=None)
    parser.add_argument('--st_internal_health_ema', type=float, default=None)
    parser.add_argument('--st_internal_freeze_patience', type=int, default=None)
    parser.add_argument('--st_internal_health_drop', type=float, default=None)
    parser.add_argument('--st_internal_reliability_floor', type=float, default=None)
    parser.add_argument('--st_internal_alignment_floor', type=float, default=None)
    parser.add_argument('--st_internal_delta_ratio_max', type=float, default=None)
    parser.add_argument('--st_internal_highfreq_leak_max', type=float, default=None)
    parser.add_argument('--st_internal_saturation_ratio', type=float, default=None)
    parser.add_argument('--st_internal_freeze_lr_ratio', type=float, default=None)
    parser.add_argument('--use_pred_structure_loss', type=_str2bool, default=None)
    parser.add_argument('--pred_structure_max_lag', type=int, default=None)
    parser.add_argument('--pred_structure_max_channels', type=int, default=None)
    parser.add_argument('--pred_structure_include_cross', type=_str2bool, default=None)
    parser.add_argument('--pred_structure_no_self', type=_str2bool, default=None)
    parser.add_argument('--pred_structure_adaptive', type=_str2bool, default=None)
    parser.add_argument('--pred_structure_warmup_ratio', type=float, default=None)
    parser.add_argument('--pred_structure_warmup_window', type=float, default=None)
    parser.add_argument('--pred_structure_strength_floor', type=float, default=None)
    parser.add_argument('--pred_structure_strength_scale', type=float, default=None)
    parser.add_argument('--pred_structure_confidence_min', type=float, default=None)
    parser.add_argument('--pred_structure_confidence_power', type=float, default=None)
    parser.add_argument('--pred_structure_huber_beta', type=float, default=None)
    parser.add_argument('--lambda_pred_structure', type=float, default=None)
    parser.add_argument('--lambda_st_residual', type=float, default=None)
    parser.add_argument('--lambda_st_delta_reg', type=float, default=None)
    parser.add_argument('--lambda_st_raw_delta_reg', type=float, default=None)
    parser.add_argument('--lambda_st_effective', type=float, default=None)
    parser.add_argument('--lambda_st_effective_ratio', type=float, default=None)
    parser.add_argument('--lambda_st_relation_reg', type=float, default=None)
    parser.add_argument('--lambda_st_branch_trend', type=float, default=None)
    parser.add_argument('--lambda_st_branch_season', type=float, default=None)
    parser.add_argument('--lambda_st_branch_freq', type=float, default=None)
    parser.add_argument('--lambda_st_branch_corr', type=float, default=None)
    parser.add_argument('--lambda_st_branch_dist', type=float, default=None)
    parser.add_argument('--use_ds_train', type=_str2bool, default=None)
    parser.add_argument('--ds_train_mode', type=str, default=None)
    parser.add_argument('--ds_style_extractor', type=str, default=None)
    parser.add_argument('--ds_lma_kernels', type=int, nargs='+', default=None)
    parser.add_argument('--ds_hierarchical', type=_str2bool, default=None)
    parser.add_argument('--ds_sigma_weight', type=_str2bool, default=None)
    parser.add_argument('--ds_warmup_epochs', type=int, default=None)
    parser.add_argument('--lambda_ts', type=float, default=None)
    parser.add_argument('--lambda_ds_trend', type=float, default=None)
    parser.add_argument('--lambda_ds_season', type=float, default=None)
    parser.add_argument('--lambda_ds_freq', type=float, default=None)
    parser.add_argument('--lambda_ds_corr', type=float, default=None)
    parser.add_argument('--lambda_ds_dist', type=float, default=None)
    parser.add_argument('--ds_inference_kernel', type=_str2bool, default=None)
    parser.add_argument('--use_final_dist_train', type=_str2bool, default=None)
    parser.add_argument('--final_dist_quantiles', type=float, nargs='+', default=None)
    parser.add_argument('--final_dist_sigma_weight', type=_str2bool, default=None)
    parser.add_argument('--final_dist_sigma_mid', type=float, default=None)
    parser.add_argument('--final_dist_sigma_scale', type=float, default=None)
    parser.add_argument('--final_dist_highfreq_start_ratio', type=float, default=None)
    parser.add_argument('--lambda_final_mean', type=float, default=None)
    parser.add_argument('--lambda_final_std', type=float, default=None)
    parser.add_argument('--lambda_final_diff_std', type=float, default=None)
    parser.add_argument('--lambda_final_quantile', type=float, default=None)
    parser.add_argument('--lambda_final_highfreq', type=float, default=None)
    parser.add_argument('--use_period_train', type=_str2bool, default=None)
    parser.add_argument('--period_lma_kernels', type=int, nargs='+', default=None)
    parser.add_argument('--period_max_lag', type=int, default=None)
    parser.add_argument('--period_sigma_weight', type=_str2bool, default=None)
    parser.add_argument('--period_sigma_mid', type=float, default=None)
    parser.add_argument('--period_sigma_scale', type=float, default=None)
    parser.add_argument('--period_warmup_epochs', type=int, default=None)
    parser.add_argument('--period_late_stabilize', type=_str2bool, default=None)
    parser.add_argument('--period_late_start_ratio', type=float, default=None)
    parser.add_argument('--period_late_min_scale', type=float, default=None)
    parser.add_argument('--lambda_period_autocorr', type=float, default=None)
    parser.add_argument('--lambda_period_amp', type=float, default=None)
    parser.add_argument('--lambda_period_phase', type=float, default=None)

    parser.add_argument('--eval_cross_corr', type=_str2bool, default=None)
    parser.add_argument('--eval_context_fid', type=_str2bool, default=None)
    parser.add_argument('--st_eval_extra_metric_iteration', type=int, default=None)
    parser.add_argument('--context_fid_ts2vec_iters', type=int, default=None)
    parser.add_argument('--context_fid_batch_size', type=int, default=None)
    parser.add_argument('--context_fid_lr', type=float, default=None)
    parser.add_argument('--context_fid_output_dims', type=int, default=None)
    parser.add_argument('--context_fid_hidden_dims', type=int, default=None)
    parser.add_argument('--context_fid_depth', type=int, default=None)
    parser.add_argument('--context_fid_max_train_length', type=int, default=None)
    parser.add_argument('--context_fid_temporal_unit', type=int, default=None)


_TRAIN_BUDGET_PRESETS = {
    # === [A2] a2 ===
    # Method: structured residual target that blends raw residual with trend/season-smoothed residual.
    # Isolation: uses use_structured_st_target only; does not depend on F3 freeze or A5 disentangled loss.
    "a2": {
        "epochs": 1000,
        "logging_iter": 100,
        "st_freeze": False,
        "use_late_decay": False,
        "use_structured_st_target": True,
        "structured_target_kernels": [1, 2, 4, 6, 12],
        "structured_target_blend_start_epoch": 0,
        "structured_target_blend_end_epoch": 0,
        "structured_target_blend_start_ratio": 0.58,
        "structured_target_blend_window": 0.02,
        "structured_target_max_blend": 0.70,
        "structured_adaptive_blend": False,
        "structured_confidence_min": 0.20,
        "structured_confidence_power": 1.0,
        "structured_trend_weight": 1.0,
        "structured_season_weight": 0.60,
        "structured_late_season_min": 0.35,
        "structured_late_start_ratio": 0.95,
        "structured_target_norm_ratio": 0.60,
        "st_alpha": 0.06,
        "st_alpha_max": 0.10,
        "st_warmup_epochs": 100,
        "st_residual_calib": True,
        "st_residual_warmup_epochs": 100,
        "st_residual_target_scale": 0.40,
        "st_feature_fusion": True,
        "st_feature_channels": 64,
        "st_feature_scale_max": 0.025,
        "st_feature_init_scale": 0.004,
        "st_feature_zero_init": True,
        "st_feature_sigma_gate": True,
        "st_feature_sigma_mid": -1.0,
        "st_feature_sigma_scale": 0.75,
        "st_feature_warmup_epochs": 250,
        "st_feature_norm_max": 0.015,
        "st_feature_input_clip": 3.0,
        "lambda_st_residual": 0.05,
        "lambda_st_delta_reg": 0.001,
        "lambda_st_raw_delta_reg": 0.00015,
        "lambda_st_effective": 0.045,
        "lambda_st_effective_ratio": 0.018,
        "lambda_st_relation_reg": 0.001,
        "st_effective_align": False,
        "st_effective_max_ratio": 0.30,
        "st_lma_affine": True,
        "st_period_branch": True,
        "st_period_candidates": [2, 3, 4, 6, 8, 12],
        "st_period_min": 2,
        "st_period_max": 12,
        "st_period_temperature": 0.40,
        "st_period_max_scale": 0.08,
        "st_period_input_condition": False,
        "st_period_input_channels": 64,
        "st_period_input_alpha": 0.020,
        "st_period_input_alpha_max": 0.040,
        "st_period_input_alpha_learnable": True,
        "st_period_input_warmup_epochs": 120,
        "st_period_input_max_scale": 0.14,
        "st_period_input_init_scale": 0.02,
        "st_detach_base_for_style": False,
        "st_branch_style_calib": False,
        "st_var_relation": True,
        "st_var_relation_rank": 8,
        "st_var_relation_beta": 0.08,
        "st_var_relation_init_beta": 0.0,
        "lambda_ts": 0.08,
        "ds_warmup_epochs": 120,
        "lambda_ds_trend": 0.04,
        "lambda_ds_season": 0.04,
        "lambda_ds_freq": 0.012,
        "lambda_ds_corr": 0.004,
        "lambda_ds_dist": 0.004,
        "use_final_dist_train": True,
        "lambda_final_mean": 0.004,
        "lambda_final_std": 0.008,
        "lambda_final_diff_std": 0.016,
        "lambda_final_quantile": 0.010,
        "lambda_final_highfreq": 0.005,
        "use_period_train": False,
        "period_lma_kernels": [1, 2, 4, 6, 12],
        "period_max_lag": 12,
        "period_warmup_epochs": 120,
        "period_late_start_ratio": 0.72,
        "period_late_min_scale": 0.50,
        "lambda_period_autocorr": 0.008,
        "lambda_period_amp": 0.005,
        "lambda_period_phase": 0.0015,
    },
    # === [F3-500] f3_500 ===
    # Method: 500-epoch dynamic ST freeze driven by disc_mean/disc_std degradation.
    # Isolation: freeze-only strategy; structured/disentangled targets are disabled.
    "f3_500": {
        "epochs": 500,
        "logging_iter": 50,
        "st_freeze": True,
        "st_freeze_warmup": 200,
        "st_freeze_patience": 2,
        "st_freeze_threshold": 0.005,
        "st_freeze_hard_threshold": 0.016,
        "st_freeze_pct_threshold": 0.20,
        "st_freeze_min_abs": 0.002,
        "st_freeze_hard_pct_threshold": 0.40,
        "st_freeze_hard_min_abs": 0.006,
        "st_freeze_std_threshold": 0.004,
        "st_freeze_std_ratio": 1.75,
        "st_freeze_std_min_abs": 0.004,
        "st_freeze_lr_ratio": 0.25,
        "st_freeze_watch": False,
        "st_post_freeze_patience": 999,
        "st_post_freeze_threshold": 999.0,
        "st_post_freeze_hard_threshold": 999.0,
        "st_post_freeze_std_threshold": 999.0,
        "st_post_freeze_std_ratio": 999.0,
        "st_alpha": 0.06,
        "st_alpha_max": 0.10,
        "st_warmup_epochs": 100,
        "st_residual_calib": True,
        "st_residual_warmup_epochs": 100,
        "st_residual_target_scale": 0.40,
        "st_feature_fusion": True,
        "st_feature_channels": 64,
        "st_feature_scale_max": 0.025,
        "st_feature_init_scale": 0.004,
        "st_feature_zero_init": True,
        "st_feature_sigma_gate": True,
        "st_feature_sigma_mid": -1.0,
        "st_feature_sigma_scale": 0.75,
        "st_feature_warmup_epochs": 250,
        "st_feature_norm_max": 0.015,
        "st_feature_input_clip": 3.0,
        "lambda_st_residual": 0.05,
        "lambda_st_delta_reg": 0.001,
        "lambda_st_raw_delta_reg": 0.00015,
        "lambda_st_effective": 0.045,
        "lambda_st_effective_ratio": 0.018,
        "lambda_st_relation_reg": 0.001,
        "st_effective_align": False,
        "st_effective_max_ratio": 0.30,
        "st_lma_affine": True,
        "st_period_branch": True,
        "st_period_candidates": [2, 3, 4, 6, 8, 12],
        "st_period_min": 2,
        "st_period_max": 12,
        "st_period_temperature": 0.40,
        "st_period_max_scale": 0.08,
        "st_period_input_condition": False,
        "st_period_input_channels": 64,
        "st_period_input_alpha": 0.020,
        "st_period_input_alpha_max": 0.040,
        "st_period_input_alpha_learnable": True,
        "st_period_input_warmup_epochs": 120,
        "st_period_input_max_scale": 0.14,
        "st_period_input_init_scale": 0.02,
        "st_detach_base_for_style": False,
        "st_branch_style_calib": False,
        "st_var_relation": True,
        "st_var_relation_rank": 8,
        "st_var_relation_beta": 0.08,
        "st_var_relation_init_beta": 0.0,
        "lambda_ts": 0.08,
        "ds_warmup_epochs": 120,
        "lambda_ds_trend": 0.04,
        "lambda_ds_season": 0.04,
        "lambda_ds_freq": 0.012,
        "lambda_ds_corr": 0.004,
        "lambda_ds_dist": 0.004,
        "use_final_dist_train": True,
        "lambda_final_mean": 0.004,
        "lambda_final_std": 0.008,
        "lambda_final_diff_std": 0.016,
        "lambda_final_quantile": 0.010,
        "lambda_final_highfreq": 0.005,
        "use_period_train": False,
        "period_lma_kernels": [1, 2, 4, 6, 12],
        "period_max_lag": 12,
        "period_warmup_epochs": 120,
        "period_late_start_ratio": 0.72,
        "period_late_min_scale": 0.50,
        "lambda_period_autocorr": 0.008,
        "lambda_period_amp": 0.005,
        "lambda_period_phase": 0.0015,
    },
    }

# === [F3-500P] f3_500p ===
# Method: 500-epoch F3 tuned to stabilize the post-freeze best checkpoint window.
# Isolation: standalone full preset; safe to delete independently from f3_500.
_TRAIN_BUDGET_PRESETS["f3_500p"] = {
    "epochs": 500,
    "logging_iter": 50,
    "st_freeze": True,
    "st_freeze_warmup": 250,
    "st_freeze_patience": 2,
    "st_freeze_threshold": 0.005,
    "st_freeze_hard_threshold": 0.012,
    "st_freeze_pct_threshold": 0.20,
    "st_freeze_min_abs": 0.002,
    "st_freeze_hard_pct_threshold": 0.35,
    "st_freeze_hard_min_abs": 0.006,
    "st_freeze_std_threshold": 0.0035,
    "st_freeze_std_ratio": 1.60,
    "st_freeze_std_min_abs": 0.0035,
    "st_freeze_lr_ratio": 0.22,
    "st_freeze_watch": False,
    "st_post_freeze_patience": 999,
    "st_post_freeze_threshold": 999.0,
    "st_post_freeze_hard_threshold": 999.0,
    "st_post_freeze_std_threshold": 999.0,
    "st_post_freeze_std_ratio": 999.0,
    "st_alpha": 0.06,
    "st_alpha_max": 0.09,
    "st_warmup_epochs": 100,
    "st_residual_calib": True,
    "st_residual_warmup_epochs": 100,
    "st_residual_target_scale": 0.40,
    "st_feature_fusion": True,
    "st_feature_channels": 64,
    "st_feature_scale_max": 0.025,
    "st_feature_init_scale": 0.004,
    "st_feature_zero_init": True,
    "st_feature_sigma_gate": True,
    "st_feature_sigma_mid": -1.0,
    "st_feature_sigma_scale": 0.75,
    "st_feature_warmup_epochs": 250,
    "st_feature_norm_max": 0.015,
    "st_feature_input_clip": 3.0,
    "lambda_st_residual": 0.05,
    "lambda_st_delta_reg": 0.001,
    "lambda_st_raw_delta_reg": 0.00015,
    "lambda_st_effective": 0.045,
    "lambda_st_effective_ratio": 0.018,
    "lambda_st_relation_reg": 0.001,
    "st_effective_align": False,
    "st_effective_max_ratio": 0.30,
    "st_lma_affine": True,
    "st_period_branch": True,
    "st_period_candidates": [2, 3, 4, 6, 8, 12],
    "st_period_min": 2,
    "st_period_max": 12,
    "st_period_temperature": 0.40,
    "st_period_max_scale": 0.08,
    "st_period_input_condition": False,
    "st_period_input_channels": 64,
    "st_period_input_alpha": 0.020,
    "st_period_input_alpha_max": 0.040,
    "st_period_input_alpha_learnable": True,
    "st_period_input_warmup_epochs": 120,
    "st_period_input_max_scale": 0.14,
    "st_period_input_init_scale": 0.02,
    "st_detach_base_for_style": False,
    "st_branch_style_calib": False,
    "st_var_relation": True,
    "st_var_relation_rank": 8,
    "st_var_relation_beta": 0.08,
    "st_var_relation_init_beta": 0.0,
    "lambda_ts": 0.08,
    "ds_warmup_epochs": 120,
    "lambda_ds_trend": 0.04,
    "lambda_ds_season": 0.04,
    "lambda_ds_freq": 0.012,
    "lambda_ds_corr": 0.004,
    "lambda_ds_dist": 0.004,
    "use_final_dist_train": True,
    "lambda_final_mean": 0.004,
    "lambda_final_std": 0.008,
    "lambda_final_diff_std": 0.016,
    "lambda_final_quantile": 0.010,
    "lambda_final_highfreq": 0.005,
    "use_period_train": False,
    "period_lma_kernels": [1, 2, 4, 6, 12],
    "period_max_lag": 12,
    "period_warmup_epochs": 120,
    "period_late_start_ratio": 0.72,
    "period_late_min_scale": 0.50,
    "lambda_period_autocorr": 0.008,
    "lambda_period_amp": 0.005,
    "lambda_period_phase": 0.0015,
}

# === [PRO3] pro3 ===
# Method: internal-F3 freeze; train like f3_500, monitor ST health, then restore/freeze on internal degradation.
# Isolation: independent pro preset with checkpoint.st_health_best; does not depend on F3 checkpoints.
_TRAIN_BUDGET_PRESETS["pro3"] = {
    "epochs": 500,
    "logging_iter": 50,
    "st_freeze": False,
    "st_internal_health": True,
    "st_internal_freeze": True,
    "st_internal_freeze_warmup_ratio": 0.35,
    "st_internal_health_ema": 0.90,
    "st_internal_freeze_patience": 1,
    "st_internal_health_drop": 0.15,
    "st_internal_reliability_floor": 0.0,
    "st_internal_alignment_floor": 0.10,
    "st_internal_delta_ratio_max": 0.75,
    "st_internal_highfreq_leak_max": 1.50,
    "st_internal_saturation_ratio": 0.92,
    "st_internal_health_kernel": 5,
    "st_internal_freeze_lr_ratio": 0.22,
    "use_late_decay": False,
    "use_structured_st_target": False,
    "use_residual_reliability": False,
    "residual_reliability_min": 0.20,
    "residual_reliability_power": 1.0,
    "residual_reliability_kernels": [3, 5, 7, 11],
    "residual_reliability_freq_topk": 3,
    "residual_reliability_acf_max_lag": 12,
    "reliability_delta_reg_boost": 1.0,
    "reliability_effective_boost": 1.0,
    "st_alpha": 0.06,
    "st_alpha_max": 0.10,
    "st_warmup_epochs": 100,
    "st_residual_calib": True,
    "st_residual_warmup_epochs": 100,
    "st_residual_target_scale": 0.40,
    "st_feature_fusion": True,
    "st_feature_channels": 64,
    "st_feature_scale_max": 0.025,
    "st_feature_init_scale": 0.004,
    "st_feature_zero_init": True,
    "st_feature_sigma_gate": True,
    "st_feature_sigma_mid": -1.0,
    "st_feature_sigma_scale": 0.75,
    "st_feature_warmup_epochs": 250,
    "st_feature_norm_max": 0.015,
    "st_feature_input_clip": 3.0,
    "lambda_st_residual": 0.05,
    "lambda_st_delta_reg": 0.001,
    "lambda_st_raw_delta_reg": 0.00015,
    "lambda_st_effective": 0.045,
    "lambda_st_effective_ratio": 0.018,
    "lambda_st_relation_reg": 0.001,
    "st_effective_align": False,
    "st_effective_max_ratio": 0.30,
    "st_lma_affine": True,
    "st_period_branch": True,
    "st_period_candidates": [2, 3, 4, 6, 8, 12],
    "st_period_min": 2,
    "st_period_max": 12,
    "st_period_temperature": 0.40,
    "st_period_max_scale": 0.08,
    "st_period_input_condition": False,
    "st_period_input_channels": 64,
    "st_period_input_alpha": 0.020,
    "st_period_input_alpha_max": 0.040,
    "st_period_input_alpha_learnable": True,
    "st_period_input_warmup_epochs": 120,
    "st_period_input_max_scale": 0.14,
    "st_period_input_init_scale": 0.02,
    "st_detach_base_for_style": False,
    "st_branch_style_calib": False,
    "st_var_relation": True,
    "st_var_relation_rank": 8,
    "st_var_relation_beta": 0.08,
    "st_var_relation_init_beta": 0.0,
    "lambda_ts": 0.08,
    "ds_warmup_epochs": 120,
    "lambda_ds_trend": 0.04,
    "lambda_ds_season": 0.04,
    "lambda_ds_freq": 0.012,
    "lambda_ds_corr": 0.004,
    "lambda_ds_dist": 0.004,
    "use_final_dist_train": True,
    "lambda_final_mean": 0.004,
    "lambda_final_std": 0.008,
    "lambda_final_diff_std": 0.016,
    "lambda_final_quantile": 0.010,
    "lambda_final_highfreq": 0.005,
    "use_period_train": False,
    "period_lma_kernels": [1, 2, 4, 6, 12],
    "period_max_lag": 12,
    "period_warmup_epochs": 120,
    "period_late_start_ratio": 0.72,
    "period_late_min_scale": 0.50,
    "lambda_period_autocorr": 0.008,
    "lambda_period_amp": 0.005,
    "lambda_period_phase": 0.0015,
}


# === [PRO4] pro4 ===
# Method: residual reliability training plus predictive lag/cross-lag structure loss.
# Isolation: training-loss-only strategy; no F3/pro3 freeze and no dataset-specific eval metric dependency.
_TRAIN_BUDGET_PRESETS["pro4"] = {
    "epochs": 500,
    "logging_iter": 50,
    "st_freeze": False,
    "st_internal_health": False,
    "st_internal_freeze": False,
    "use_late_decay": False,
    "use_structured_st_target": False,
    "use_residual_reliability": True,
    "residual_reliability_min": 0.20,
    "residual_reliability_power": 1.0,
    "residual_reliability_kernels": [3, 5, 7, 11],
    "residual_reliability_freq_topk": 3,
    "residual_reliability_acf_max_lag": 12,
    "reliability_delta_reg_boost": 2.0,
    "reliability_effective_boost": 2.0,
    "use_pred_structure_loss": True,
    "pred_structure_max_lag": 6,
    "pred_structure_max_channels": 64,
    "pred_structure_include_cross": True,
    "pred_structure_no_self": True,
    "pred_structure_adaptive": True,
    "pred_structure_warmup_ratio": 0.25,
    "pred_structure_warmup_window": 0.05,
    "pred_structure_strength_floor": 0.04,
    "pred_structure_strength_scale": 0.20,
    "pred_structure_confidence_min": 0.0,
    "pred_structure_confidence_power": 1.0,
    "pred_structure_huber_beta": 0.03,
    "lambda_pred_structure": 0.003,
    "st_alpha": 0.06,
    "st_alpha_max": 0.10,
    "st_warmup_epochs": 100,
    "st_residual_calib": True,
    "st_residual_warmup_epochs": 100,
    "st_residual_target_scale": 0.40,
    "st_feature_fusion": True,
    "st_feature_channels": 64,
    "st_feature_scale_max": 0.025,
    "st_feature_init_scale": 0.004,
    "st_feature_zero_init": True,
    "st_feature_sigma_gate": True,
    "st_feature_sigma_mid": -1.0,
    "st_feature_sigma_scale": 0.75,
    "st_feature_warmup_epochs": 250,
    "st_feature_norm_max": 0.015,
    "st_feature_input_clip": 3.0,
    "lambda_st_residual": 0.05,
    "lambda_st_delta_reg": 0.001,
    "lambda_st_raw_delta_reg": 0.00015,
    "lambda_st_effective": 0.045,
    "lambda_st_effective_ratio": 0.018,
    "lambda_st_relation_reg": 0.001,
    "st_effective_align": False,
    "st_effective_max_ratio": 0.30,
    "st_lma_affine": True,
    "st_period_branch": True,
    "st_period_candidates": [2, 3, 4, 6, 8, 12],
    "st_period_min": 2,
    "st_period_max": 12,
    "st_period_temperature": 0.40,
    "st_period_max_scale": 0.08,
    "st_period_input_condition": False,
    "st_period_input_channels": 64,
    "st_period_input_alpha": 0.020,
    "st_period_input_alpha_max": 0.040,
    "st_period_input_alpha_learnable": True,
    "st_period_input_warmup_epochs": 120,
    "st_period_input_max_scale": 0.14,
    "st_period_input_init_scale": 0.02,
    "st_detach_base_for_style": False,
    "st_branch_style_calib": False,
    "st_var_relation": True,
    "st_var_relation_rank": 8,
    "st_var_relation_beta": 0.08,
    "st_var_relation_init_beta": 0.0,
    "lambda_ts": 0.08,
    "ds_warmup_epochs": 120,
    "lambda_ds_trend": 0.04,
    "lambda_ds_season": 0.04,
    "lambda_ds_freq": 0.012,
    "lambda_ds_corr": 0.004,
    "lambda_ds_dist": 0.004,
    "use_final_dist_train": True,
    "lambda_final_mean": 0.004,
    "lambda_final_std": 0.008,
    "lambda_final_diff_std": 0.016,
    "lambda_final_quantile": 0.010,
    "lambda_final_highfreq": 0.005,
    "use_period_train": False,
    "period_lma_kernels": [1, 2, 4, 6, 12],
    "period_max_lag": 12,
    "period_warmup_epochs": 120,
    "period_late_start_ratio": 0.72,
    "period_late_min_scale": 0.50,
    "lambda_period_autocorr": 0.008,
    "lambda_period_amp": 0.005,
    "lambda_period_phase": 0.0015,
}



def _resolve_train_budget(parsed_args):
    budget = getattr(parsed_args, "train_budget", None)
    available = list(_TRAIN_BUDGET_PRESETS.keys())
    if budget in (None, "", "none", "None"):
        epochs = int(getattr(parsed_args, "epochs", 0) or 0)
        if epochs and epochs <= 650:
            preferred_order = ("f3_500",)
        else:
            preferred_order = ("f3_500",)
        for preferred in preferred_order:
            if preferred in available:
                return preferred
        return available[0] if available else "f3_500"
    if budget != "auto":
        return budget
    epochs = int(getattr(parsed_args, "epochs", 0) or 0)
    if epochs and epochs <= 650:
        for preferred in ("f3_500",):
            if preferred in available:
                return preferred
    for preferred in ("f3_500",):
        if preferred in available:
            return preferred
    return available[0] if available else "f3_500"


def _apply_train_budget_preset(parsed_args):
    budget = _resolve_train_budget(parsed_args)
    parsed_args.train_budget = budget
    if budget is None:
        return parsed_args
    preset = _TRAIN_BUDGET_PRESETS.get(budget)
    if preset is None:
        raise ValueError(f"Unknown train_budget: {budget}")
    for key, value in preset.items():
        if not _cli_has_arg(key):
            setattr(parsed_args, key, value)
    return parsed_args


def _apply_st_ds_defaults(parsed_args):
    defaults = {
        "use_st_adapter": True,
        "st_channels": 64,
        "st_res_layers": 2,
        "st_nheads": 4,
        "st_freq_tier": 1,
        "st_dropout": 0.0,
        "st_input_noise": 0.0,
        "st_freeze": False,
        "st_freeze_warmup": 200,
        "st_freeze_patience": 2,
        "st_freeze_threshold": 0.005,
        "st_freeze_hard_threshold": 0.0,
        "st_freeze_std_threshold": 0.0,
        "st_freeze_std_ratio": 0.0,
        "st_freeze_lr_ratio": 0.3,
        "st_post_freeze_patience": 2,
        "st_post_freeze_threshold": 0.01,
        "st_post_freeze_hard_threshold": 0.0,
        "st_post_freeze_std_threshold": 0.0,
        "st_post_freeze_std_ratio": 0.0,
        "use_late_decay": False,
        "late_decay_start_ratio": 0.70,
        "late_decay_start_epoch": 0,
        "late_decay_min_scale": 1.0,
        "late_decay_power": 1.0,
        "late_decay_st_strength": False,
        "late_decay_style_loss": False,
        "use_structured_st_target": False,
        "structured_target_kernels": [1, 2, 4, 6, 12],
        "structured_target_blend_start_epoch": 0,
        "structured_target_blend_end_epoch": 0,
        "structured_target_blend_start_ratio": 0.0,
        "structured_target_blend_window": 0.02,
        "structured_target_max_blend": 1.0,
        "structured_adaptive_blend": False,
        "structured_confidence_min": 0.0,
        "structured_confidence_power": 1.0,
        "structured_trend_weight": 1.0,
        "structured_season_weight": 0.45,
        "structured_late_season_min": 0.20,
        "structured_late_start_ratio": 0.50,
        "structured_target_norm_ratio": 0.35,
        "structured_target_norm_ratio_final": 0.35,
        "structured_target_norm_switch": False,
        "use_residual_reliability": False,
        "residual_reliability_min": 0.20,
        "residual_reliability_power": 1.0,
        "residual_reliability_kernels": [3, 5, 7, 11],
        "residual_reliability_freq_topk": 3,
        "residual_reliability_acf_max_lag": 12,
        "reliability_delta_reg_boost": 1.0,
        "reliability_effective_boost": 1.0,
        "st_internal_health": False,
        "st_internal_health_kernel": 5,
        "st_internal_freeze": False,
        "st_internal_freeze_warmup_ratio": 0.25,
        "st_internal_health_ema": 0.95,
        "st_internal_freeze_patience": 3,
        "st_internal_health_drop": 0.20,
        "st_internal_reliability_floor": 0.30,
        "st_internal_alignment_floor": 0.05,
        "st_internal_delta_ratio_max": 0.65,
        "st_internal_highfreq_leak_max": 1.25,
        "st_internal_saturation_ratio": 0.90,
        "st_internal_freeze_lr_ratio": 0.22,
        "use_pred_structure_loss": False,
        "pred_structure_max_lag": 6,
        "pred_structure_max_channels": 64,
        "pred_structure_include_cross": True,
        "pred_structure_no_self": True,
        "pred_structure_adaptive": True,
        "pred_structure_warmup_ratio": 0.25,
        "pred_structure_warmup_window": 0.05,
        "pred_structure_strength_floor": 0.04,
        "pred_structure_strength_scale": 0.20,
        "pred_structure_confidence_min": 0.0,
        "pred_structure_confidence_power": 1.0,
        "pred_structure_huber_beta": 0.03,
        "lambda_pred_structure": 0.003,
        "st_alpha": 0.05,
        "st_alpha_max": 0.08,
        "st_alpha_learnable": True,
        "st_zero_init": True,
        "st_sigma_gate": True,
        "st_context_sigma_gate": True,
        "st_feature_fusion": True,
        "st_feature_channels": 64,
        "st_feature_scale_max": 0.05,
        "st_feature_init_scale": 0.010,
        "st_feature_zero_init": True,
        "st_feature_sigma_gate": True,
        "st_feature_sigma_mid": -1.0,
        "st_feature_sigma_scale": 0.75,
        "st_feature_warmup_epochs": 80,
        "st_feature_norm_max": 0.040,
        "st_feature_input_clip": 3.0,
        "st_trust_gate": False,
        "st_trust_init": 0.10,
        "st_trust_max": 0.60,
        "st_trust_learnable": True,
        "st_sigma_mid": 0.0,
        "st_sigma_scale": 2.0,
        "st_warmup_epochs": 200,
        "st_residual_calib": True,
        "st_residual_warmup_epochs": 80,
        "st_residual_huber_beta": 0.05,
        "st_residual_target_scale": 0.25,
        "st_effective_align": False,
        "st_effective_huber_beta": 0.05,
        "st_effective_max_ratio": 0.35,
        "st_lma_affine": True,
        "st_period_branch": True,
        "st_period_candidates": [2, 3, 4, 6, 8, 12],
        "st_period_min": 2,
        "st_period_max": 12,
        "st_period_temperature": 0.45,
        "st_period_max_scale": 0.12,
        "st_period_input_condition": False,
        "st_period_input_channels": 64,
        "st_period_input_alpha": 0.018,
        "st_period_input_alpha_max": 0.035,
        "st_period_input_alpha_learnable": True,
        "st_period_input_warmup_epochs": 120,
        "st_period_input_max_scale": 0.12,
        "st_period_input_init_scale": 0.02,
        "st_detach_base_for_style": False,
        "st_branch_style_calib": False,
        "st_var_relation": True,
        "st_var_relation_rank": 8,
        "st_var_relation_beta": 0.10,
        "st_var_relation_init_beta": 0.0,
        "st_var_relation_no_self": True,        "lambda_st_residual": 0.020,
        "lambda_st_delta_reg": 0.002,
        "lambda_st_raw_delta_reg": 0.0005,
        "lambda_st_effective": 0.05,
        "lambda_st_effective_ratio": 0.01,
        "lambda_st_relation_reg": 0.001,
        "lambda_st_branch_trend": 0.010,
        "lambda_st_branch_season": 0.010,
        "lambda_st_branch_freq": 0.005,
        "lambda_st_branch_corr": 0.002,
        "lambda_st_branch_dist": 0.002,
        "use_ds_train": True,
        "ds_train_mode": "loss_only",
        "ds_style_extractor": "multi_scale_lma",
        "ds_lma_kernels": [1, 2, 4, 6, 12],
        "ds_hierarchical": True,
        "ds_sigma_weight": True,
        "ds_warmup_epochs": 200,        "lambda_ts": 0.05,
        "lambda_ds_trend": 0.035,
        "lambda_ds_season": 0.035,
        "lambda_ds_freq": 0.012,
        "lambda_ds_corr": 0.003,
        "lambda_ds_dist": 0.003,
        "ds_inference_kernel": False,
        "use_final_dist_train": False,
        "final_dist_quantiles": [0.05, 0.25, 0.50, 0.75, 0.95],
        "final_dist_sigma_weight": True,
        "final_dist_sigma_mid": 0.0,
        "final_dist_sigma_scale": 2.0,
        "final_dist_highfreq_start_ratio": 0.50,
        "lambda_final_mean": 0.005,
        "lambda_final_std": 0.010,
        "lambda_final_diff_std": 0.020,
        "lambda_final_quantile": 0.020,
        "lambda_final_highfreq": 0.010,
        "use_period_train": False,
        "period_lma_kernels": [1, 2, 4, 6, 12],
        "period_max_lag": 12,
        "period_sigma_weight": True,
        "period_sigma_mid": 0.0,
        "period_sigma_scale": 2.0,
        "period_warmup_epochs": 200,
        "period_late_stabilize": False,
        "period_late_start_ratio": 0.80,
        "period_late_min_scale": 0.55,
        "lambda_period_autocorr": 0.010,
        "lambda_period_amp": 0.006,
        "lambda_period_phase": 0.002,
        "eval_cross_corr": True,
        "eval_context_fid": True,
        "st_eval_extra_metric_iteration": 1,
        "context_fid_ts2vec_iters": None,
        "context_fid_batch_size": 8,
        "context_fid_lr": 0.001,
        "context_fid_output_dims": 320,
        "context_fid_hidden_dims": 64,
        "context_fid_depth": 10,
        "context_fid_max_train_length": 3000,
        "context_fid_temporal_unit": 0,
    }
    for key, value in defaults.items():
        if getattr(parsed_args, key, None) is None:
            setattr(parsed_args, key, value)
    return parsed_args


def _apply_loader_defaults(parsed_args):
    if getattr(parsed_args, 'num_workers', None) is None:
        parsed_args.num_workers = 0 if os.name == 'nt' else 4
    if getattr(parsed_args, 'eval_num_workers', None) is None:
        parsed_args.eval_num_workers = 0
    if getattr(parsed_args, 'pin_memory', None) is None:
        parsed_args.pin_memory = False
    if getattr(parsed_args, 'persistent_workers', None) is None:
        parsed_args.persistent_workers = False
    if getattr(parsed_args, 'prefetch_factor', None) is None:
        parsed_args.prefetch_factor = 2
    return parsed_args


def parse_args_uncond():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--eval_num_workers', default=None, type=int,
                        help='Number of workers to use for evaluation dataloader')
    parser.add_argument('--pin_memory', type=_str2bool, default=None,
                        help='use pinned dataloader memory')
    parser.add_argument('--persistent_workers', type=_str2bool, default=None,
                        help='keep dataloader workers alive between epochs')
    parser.add_argument('--prefetch_factor', default=None, type=int,
                        help='dataloader prefetch factor when num_workers > 0')
    parser.add_argument('--resume', type=_str2bool, default=False, help='resume from checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='explicit checkpoint path to resume from')
    parser.add_argument('--resume_latest', type=_str2bool, default=True, help='prefer the latest checkpoint when resuming')
    parser.add_argument('--allow_partial_resume', type=_str2bool, default=False,
                        help='allow checkpoint loading when model/EMA keys do not exactly match')
    parser.add_argument('--save_latest', type=_str2bool, default=True, help='save an interrupt-resume checkpoint every epoch')
    parser.add_argument('--run_id', type=str, default=None,
                        help='training run id; leave empty to create a new id for training or use the latest id for evaluation')
    parser.add_argument('--train_budget', type=str,
                        choices=['a2', 'f3_500', 'f3_500p',
                                 'pro3', 'pro4', 'auto'],
                        default=None,
                        help='generic ST-DS training budget preset')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--view_dir', default='./view', help='path to save evaluation records')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'unconditional'],
                        help='tags for neptune logger', nargs='+')

    # --- diffusion process --- #
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/unconditional/TS2I/fred_md.yaml',
                        help='config file')

    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---:
    parser.add_argument('--dataset',
                        choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain',
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---:
    parser.add_argument('--use_stft', type=bool,
                        help='use stft transform - if absent, use delay embedding')  # can be base
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')

    # --- model--- :
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=100,
                        help='number of iterations between logging')

    parser.add_argument('--percent', type=int, default=100)
    _add_st_ds_args(parser)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    _apply_loader_defaults(parsed_args)
    _apply_train_budget_preset(parsed_args)
    _apply_st_ds_defaults(parsed_args)
    # for short-term benchamark
    if parsed_args.dataset in ['stocks', 'stock', 'sine', 'energy', 'mujoco']:
        parsed_args.input_size = parsed_args.input_channels
    return parsed_args


def parse_args_cond():
    """
    Parse arguments for unconditional models
    Returns: unconditioanl generation args namespace

    """
    parser = argparse.ArgumentParser()
    # --- general ---
    # NOTE: the following arguments are general, they are not present in the config file:
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Number of workers to use for dataloader')
    parser.add_argument('--eval_num_workers', default=None, type=int,
                        help='Number of workers to use for evaluation dataloader')
    parser.add_argument('--pin_memory', type=_str2bool, default=None,
                        help='use pinned dataloader memory')
    parser.add_argument('--persistent_workers', type=_str2bool, default=None,
                        help='keep dataloader workers alive between epochs')
    parser.add_argument('--prefetch_factor', default=None, type=int,
                        help='dataloader prefetch factor when num_workers > 0')
    parser.add_argument('--resume', type=_str2bool, default=False, help='resume from checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='explicit checkpoint path to resume from')
    parser.add_argument('--resume_latest', type=_str2bool, default=True, help='prefer the latest checkpoint when resuming')
    parser.add_argument('--allow_partial_resume', type=_str2bool, default=False,
                        help='allow checkpoint loading when model/EMA keys do not exactly match')
    parser.add_argument('--save_latest', type=_str2bool, default=True, help='save an interrupt-resume checkpoint every epoch')
    parser.add_argument('--run_id', type=str, default=None,
                        help='training run id; leave empty to create a new id for training or use the latest id for evaluation')
    parser.add_argument('--train_budget', type=str,
                        choices=['a2', 'f3_500', 'f3_500p',
                                 'pro3', 'pro4', 'auto'],
                        default=None,
                        help='generic ST-DS training budget preset')
    parser.add_argument('--log_dir', default='./logs', help='path to save logs')
    parser.add_argument('--view_dir', default='./view', help='path to save evaluation records')
    parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
    parser.add_argument('--tags', type=str, default=['karras', 'conditional'],
                        help='tags for neptune logger', nargs='+')

    # --- diffusion process ---
    parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='deterministic sampling')

    # ## --- config file --- # ##
    # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
    parser.add_argument('--config', type=str, default='./configs/interpolation/TS2I/physionet.yaml',
                        help='config file')

    # --- training ---
    parser.add_argument('--epochs', type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')

    # --- data ---
    parser.add_argument('--dataset',
                        choices=['kdd_cup', 'traffic_hourly', 'solar_weekly', 'temperature_rain',
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')

    # --- image transformations ---
    parser.add_argument('--use_stft', type=bool,
                        help='use stft transform - if absent, use delay embedding')  # can be base
    parser.add_argument('--n_fft', type=int, help='n_fft, only needed if using stft')
    parser.add_argument('--hop_length', type=int, help='hop_length, only needed if using stft')
    parser.add_argument('--delay', type=int,
                        help='delay for the delay embedding transformation, only needed if using delay embedding')
    parser.add_argument('--embedding', type=int,
                        help='embedding for the delay embedding transformation, only needed if using delay embedding')

    # --- model---
    parser.add_argument('--img_resolution', type=int, help='image resolution')
    parser.add_argument('--input_channels', type=int,
                        help='number of image channels, 2 if stft is used, 1 for delay embedding')
    parser.add_argument('--unet_channels', type=int, help='number of unet channels')
    parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
    parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
    parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
    parser.add_argument('--ema', type=bool, help='use ema')
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=100,
                        help='number of iterations between logging')

    parser.add_argument('--percent', type=int, default=100)
    _add_st_ds_args(parser)
    parsed_args = parser.parse_args()

    # load config file
    config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
    # override config file with command line args
    for k, v in vars(parsed_args).items():
        if v is None:
            setattr(parsed_args, k, config.get(k, None))
    # add to the parsed args, configs that are not in the parsed args but do in the config file
    # this is needed since multiple config files setups may be used
    for k, v in config.items():
        if k not in vars(parsed_args):
            setattr(parsed_args, k, v)
    _apply_loader_defaults(parsed_args)
    _apply_train_budget_preset(parsed_args)
    _apply_st_ds_defaults(parsed_args)
    return parsed_args
