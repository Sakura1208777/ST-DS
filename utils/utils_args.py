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
    parser.add_argument('--use_transition_context_film', type=_str2bool, default=None)
    parser.add_argument('--transition_context_scales', type=int, nargs='+', default=None)
    parser.add_argument('--transition_context_hidden', type=int, default=None)
    parser.add_argument('--transition_context_max_scale', type=float, default=None)
    parser.add_argument('--transition_context_init_scale', type=float, default=None)
    parser.add_argument('--transition_context_input_clip', type=float, default=None)
    parser.add_argument('--transition_context_length_mid', type=float, default=None)
    parser.add_argument('--transition_context_length_tau', type=float, default=None)
    parser.add_argument('--transition_context_zero_init', type=_str2bool, default=None)
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
    parser.add_argument('--use_mstc_token_adapter', type=_str2bool, default=None)
    parser.add_argument('--mstc_token_scales', type=int, nargs='+', default=None)
    parser.add_argument('--mstc_token_heads', type=int, default=None)
    parser.add_argument('--mstc_token_max_scale', type=float, default=None)
    parser.add_argument('--mstc_token_init_scale', type=float, default=None)
    parser.add_argument('--mstc_token_length_mid', type=float, default=None)
    parser.add_argument('--mstc_token_length_tau', type=float, default=None)
    parser.add_argument('--mstc_token_dropout', type=float, default=None)
    parser.add_argument('--mstc_token_zero_init', type=_str2bool, default=None)

    parser.add_argument('--st_input_noise', type=float, default=None)
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
    parser.add_argument('--residual_reliability_per_timestep', type=_str2bool, default=None)
    parser.add_argument('--residual_reliability_warmup_ratio', type=float, default=None)
    parser.add_argument('--residual_reliability_short_gate', type=_str2bool, default=None)
    parser.add_argument('--residual_reliability_length_mid', type=float, default=None)
    parser.add_argument('--residual_reliability_length_tau', type=float, default=None)
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
    parser.add_argument('--pred_structure_auto_lag', type=_str2bool, default=None)
    parser.add_argument('--use_delta_smooth_reg', type=_str2bool, default=None)
    parser.add_argument('--lambda_delta_smooth', type=float, default=None)
    parser.add_argument('--use_delta_spectral_reg', type=_str2bool, default=None)
    parser.add_argument('--lambda_delta_spectral', type=float, default=None)
    parser.add_argument('--use_early_st_stabilize', type=_str2bool, default=None)
    parser.add_argument('--early_st_stabilize_end_epoch', type=int, default=None)
    parser.add_argument('--early_st_stabilize_smooth_mult', type=float, default=None)
    parser.add_argument('--early_st_stabilize_spectral_mult', type=float, default=None)
    parser.add_argument('--early_st_stabilize_relation_mult', type=float, default=None)
    parser.add_argument('--st_len_adaptive_reg', type=_str2bool, default=None)
    parser.add_argument('--st_short_lambda_delta_smooth', type=float, default=None)
    parser.add_argument('--st_short_lambda_delta_spectral', type=float, default=None)
    parser.add_argument('--use_delta_explore', type=_str2bool, default=None)
    parser.add_argument('--delta_explore_std', type=float, default=None)
    parser.add_argument('--delta_explore_start_epoch', type=int, default=None)
    parser.add_argument('--delta_explore_end_ratio', type=float, default=None)
    parser.add_argument('--delta_explore_decay_power', type=float, default=None)
    parser.add_argument('--use_ctc', type=_str2bool, default=None)
    parser.add_argument('--ctc_dilations', type=int, nargs='+', default=None)
    parser.add_argument('--ctc_kernel_size', type=int, default=None)
    parser.add_argument('--ctc_max_scale', type=float, default=None)
    parser.add_argument('--ctc_init_scale', type=float, default=None)
    parser.add_argument('--ctc_length_mid', type=float, default=None)
    parser.add_argument('--ctc_length_tau', type=float, default=None)
    parser.add_argument('--ctc_zero_init', type=_str2bool, default=None)
    parser.add_argument('--ctc_dropout', type=float, default=None)
    parser.add_argument('--use_transition_teacher', type=_str2bool, default=None)
    parser.add_argument('--transition_teacher_horizons', type=int, nargs='+', default=None)
    parser.add_argument('--transition_teacher_ridge', type=float, default=None)
    parser.add_argument('--transition_teacher_max_channels', type=int, default=None)
    parser.add_argument('--transition_teacher_warmup_ratio', type=float, default=None)
    parser.add_argument('--transition_teacher_warmup_window', type=float, default=None)
    parser.add_argument('--lambda_transition_teacher', type=float, default=None)
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
    parser.add_argument('--lambda_ds_ar_residual', type=float, default=None)
    parser.add_argument('--ds_ar_order', type=int, default=None)
    parser.add_argument('--lambda_ds_multi_lag', type=float, default=None)
    parser.add_argument('--ds_multi_lag_lags', type=int, nargs='+', default=None)
    parser.add_argument('--lambda_ds_coherence', type=float, default=None)
    parser.add_argument('--ds_coherence_max_channels', type=int, default=None)
    parser.add_argument('--ds_coherence_min_channels', type=int, default=None)
    parser.add_argument('--use_ds_long_loss_gate', type=_str2bool, default=None)
    parser.add_argument('--ds_long_loss_length_mid', type=float, default=None)
    parser.add_argument('--ds_long_loss_length_tau', type=float, default=None)
    parser.add_argument('--ds_long_loss_gate_floor', type=float, default=None)
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
    parser.add_argument('--use_loss_temporal_transform', type=_str2bool, default=None)
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
    # Isolation: uses use_structured_st_target only.
    "a2": {
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
    # === [F3] f3 ===
    # Method: 500-epoch ST/DS training preset.
    # Isolation: keeps the original F3 training signals only.
    "f3": {
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

# === [PRO4] pro4 ===
# Method: residual reliability training plus predictive lag/cross-lag structure loss.
# Isolation: training-loss-only strategy with no dataset-specific eval metric dependency.
_TRAIN_BUDGET_PRESETS["pro4"] = {
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

# === [MAX1] max1: pro4 + delta smooth/spectral reg + auto-lag pred_structure + late decay ===
_TRAIN_BUDGET_PRESETS["max1"] = {
    "ema_decay": 0.999,
    # --- Late decay (prevents late-epoch overfitting) ---
    "use_late_decay": True,
    "late_decay_start_ratio": 0.60,
    "late_decay_start_epoch": 0,
    "late_decay_min_scale": 0.50,
    "late_decay_power": 1.0,
    "late_decay_st_strength": True,
    "late_decay_style_loss": False,
    # --- Delta smoothness regularization ---
    "use_delta_smooth_reg": True,
    "lambda_delta_smooth": 0.005,
    # --- Delta spectral alignment ---
    "use_delta_spectral_reg": True,
    "lambda_delta_spectral": 0.003,
    # --- Sequence-adaptive pred_structure lag ---
    "pred_structure_auto_lag": True,
    # --- Variable relation constraints (stronger than pro4) ---
    "lambda_st_relation_reg": 0.003,
    "st_var_relation_beta": 0.12,
    # --- Same as pro4 below ---
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
    "st_residual_target_scale": 0.38,
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

# === [MAX2] max2: CTC + multi-lag autocorr + AR(3) + spectral coherence + length-adaptive reg ===
_TRAIN_BUDGET_PRESETS["max2"] = {
    "ema_decay": 0.999,
    # --- Late decay (prevents late-epoch overfitting) ---
    "use_late_decay": True,
    "late_decay_start_ratio": 0.60,
    "late_decay_start_epoch": 0,
    "late_decay_min_scale": 0.50,
    "late_decay_power": 1.0,
    "late_decay_st_strength": True,
    "late_decay_style_loss": False,
    # --- Delta smoothness regularization (length-adaptive) ---
    "use_delta_smooth_reg": True,
    "lambda_delta_smooth": 0.005,
    # --- Delta spectral alignment (length-adaptive) ---
    "use_delta_spectral_reg": True,
    "lambda_delta_spectral": 0.003,
    # --- Length-adaptive regularization: short-sequence params aligned with max1 ---
    "st_len_adaptive_reg": True,
    "st_short_lambda_delta_smooth": 0.005,
    "st_short_lambda_delta_spectral": 0.003,
    # --- Sequence-adaptive pred_structure lag ---
    "pred_structure_auto_lag": True,
    # --- [MAX2] CTC: Causal Temporal Convolution for local temporal dependencies ---
    "use_ctc": True,
    "ctc_dilations": [1, 2, 4, 8],
    "ctc_kernel_size": 3,
    "ctc_max_scale": 0.015,
    "ctc_init_scale": 0.003,
    "ctc_length_mid": 128.0,
    "ctc_length_tau": 24.0,
    "ctc_zero_init": True,
    "ctc_dropout": 0.0,
    # --- [MAX2] MSTC-token disabled: replaced by CTC ---
    "use_mstc_token_adapter": False,
    # --- [MAX2] Transition context disabled: replaced by CTC ---
    "use_transition_context_film": False,
    # --- Variable relation constraints ---
    "lambda_st_relation_reg": 0.003,
    "st_var_relation_beta": 0.12,
    # --- Residual reliability: short-sequence guard, length-gated off for long sequences ---
    "use_residual_reliability": True,
    "residual_reliability_min": 0.20,
    "residual_reliability_power": 1.0,
    "residual_reliability_kernels": [3, 5, 7, 11],
    "residual_reliability_freq_topk": 3,
    "residual_reliability_acf_max_lag": 12,
    "reliability_delta_reg_boost": 2.0,
    "reliability_effective_boost": 2.0,
    "residual_reliability_short_gate": True,
    "residual_reliability_length_mid": 128.0,
    "residual_reliability_length_tau": 24.0,
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
    # --- [MAX2] DS loss: multi-lag autocorr + AR(3) + spectral coherence ---
    "lambda_ds_ar_residual": 0.006,
    "ds_ar_order": 3,
    "lambda_ds_multi_lag": 0.003,
    "ds_multi_lag_lags": [1, 5, 10, 20, 50],
    "lambda_ds_coherence": 0.002,
    "ds_coherence_max_channels": 64,
    "use_ds_long_loss_gate": True,
    "ds_long_loss_length_mid": 128.0,
    "ds_long_loss_length_tau": 24.0,
    "ds_long_loss_gate_floor": 0.0,
    "ds_coherence_min_channels": 8,
    # --- ST adapter core ---
    "st_alpha": 0.06,
    "st_alpha_max": 0.10,
    "st_warmup_epochs": 100,
    "st_residual_calib": True,
    "st_residual_warmup_epochs": 100,
    "st_residual_target_scale": 0.38,
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
    "st_var_relation_init_beta": 0.0,
    # --- DS style loss ---
    "lambda_ts": 0.08,
    "ds_warmup_epochs": 120,
    "lambda_ds_trend": 0.04,
    "lambda_ds_season": 0.04,
    "lambda_ds_freq": 0.012,
    "lambda_ds_corr": 0.004,
    "lambda_ds_dist": 0.004,
    # --- Final distribution ---
    "use_final_dist_train": True,
    "lambda_final_mean": 0.004,
    "lambda_final_std": 0.008,
    "lambda_final_diff_std": 0.016,
    "lambda_final_quantile": 0.010,
    "lambda_final_highfreq": 0.005,
    # --- Period train ---
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

# === [MAX3] max3: max1 + early ST trajectory stabilization ===
# Isolation: standalone full preset; does not inherit from max1 at runtime.
_TRAIN_BUDGET_PRESETS["max3"] = {
    "ema_decay": 0.999,
    # --- Late decay (prevents late-epoch overfitting) ---
    "use_late_decay": True,
    "late_decay_start_ratio": 0.60,
    "late_decay_start_epoch": 0,
    "late_decay_min_scale": 0.50,
    "late_decay_power": 1.0,
    "late_decay_st_strength": True,
    "late_decay_style_loss": False,
    # --- Delta smoothness regularization ---
    "use_delta_smooth_reg": True,
    "lambda_delta_smooth": 0.005,
    # --- Delta spectral alignment ---
    "use_delta_spectral_reg": True,
    "lambda_delta_spectral": 0.003,
    # --- [MAX3] Early ST trajectory stabilization ---
    "use_early_st_stabilize": True,
    "early_st_stabilize_end_epoch": 150,
    "early_st_stabilize_smooth_mult": 1.6,
    "early_st_stabilize_spectral_mult": 1.5,
    "early_st_stabilize_relation_mult": 1.5,
    # --- Sequence-adaptive pred_structure lag ---
    "pred_structure_auto_lag": True,
    # --- Variable relation constraints (same final capacity as max1) ---
    "lambda_st_relation_reg": 0.003,
    "st_var_relation_beta": 0.12,
    # --- Same as max1 below ---
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
    "st_residual_target_scale": 0.38,
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
            preferred_order = ("f3",)
        else:
            preferred_order = ("f3",)
        for preferred in preferred_order:
            if preferred in available:
                return preferred
        return available[0] if available else "f3"
    if budget != "auto":
        return budget
    epochs = int(getattr(parsed_args, "epochs", 0) or 0)
    if epochs and epochs <= 650:
        for preferred in ("f3",):
            if preferred in available:
                return preferred
    for preferred in ("f3",):
        if preferred in available:
            return preferred
    return available[0] if available else "f3"


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
        "residual_reliability_short_gate": False,
        "residual_reliability_length_mid": 96.0,
        "residual_reliability_length_tau": 16.0,
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
        "pred_structure_auto_lag": False,
        "use_delta_smooth_reg": False,
        "lambda_delta_smooth": 0.005,
        "use_delta_spectral_reg": False,
        "lambda_delta_spectral": 0.003,
        "use_early_st_stabilize": False,
        "early_st_stabilize_end_epoch": 150,
        "early_st_stabilize_smooth_mult": 1.0,
        "early_st_stabilize_spectral_mult": 1.0,
        "early_st_stabilize_relation_mult": 1.0,
        "st_len_adaptive_reg": False,
        "st_short_lambda_delta_smooth": 0.005,
        "st_short_lambda_delta_spectral": 0.004,
        "use_delta_explore": False,
        "delta_explore_std": 0.012,
        "delta_explore_start_epoch": 100,
        "delta_explore_end_ratio": 0.65,
        "delta_explore_decay_power": 1.0,
        "use_ctc": False,
        "ctc_dilations": [1, 2, 4, 8],
        "ctc_kernel_size": 3,
        "ctc_max_scale": 0.015,
        "ctc_init_scale": 0.003,
        "ctc_length_mid": 96.0,
        "ctc_length_tau": 32.0,
        "ctc_zero_init": True,
        "ctc_dropout": 0.0,
        "use_transition_teacher": False,
        "transition_teacher_horizons": [1, 2, 4],
        "transition_teacher_ridge": 0.01,
        "transition_teacher_max_channels": 64,
        "transition_teacher_warmup_ratio": 0.35,
        "transition_teacher_warmup_window": 0.10,
        "lambda_transition_teacher": 0.0005,
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
        "use_transition_context_film": False,
        "transition_context_scales": [4, 16],
        "transition_context_hidden": 64,
        "transition_context_max_scale": 0.012,
        "transition_context_init_scale": 0.002,
        "transition_context_input_clip": 3.0,
        "transition_context_length_mid": 96.0,
        "transition_context_length_tau": 32.0,
        "transition_context_zero_init": True,
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
        "st_var_relation_no_self": True,
        "use_mstc_token_adapter": False,
        "mstc_token_scales": [4, 16],
        "mstc_token_heads": 4,
        "mstc_token_max_scale": 0.015,
        "mstc_token_init_scale": 0.003,
        "mstc_token_length_mid": 96.0,
        "mstc_token_length_tau": 32.0,
        "mstc_token_dropout": 0.0,
        "mstc_token_zero_init": True,
        "lambda_st_residual": 0.020,
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
        "lambda_ds_ar_residual": 0.0,
        "ds_ar_order": 1,
        "lambda_ds_multi_lag": 0.0,
        "ds_multi_lag_lags": [1, 5, 10, 20, 50],
        "lambda_ds_coherence": 0.0,
        "ds_coherence_max_channels": 64,
        "ds_coherence_min_channels": 0,
        "use_ds_long_loss_gate": False,
        "ds_long_loss_length_mid": 96.0,
        "ds_long_loss_length_tau": 16.0,
        "ds_long_loss_gate_floor": 0.0,
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
        "use_loss_temporal_transform": False,
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
                                choices=['a2', 'f3', 'pro4', 'max1', 'max2', 'max3', 'auto'],
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
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks',
                                 'eeg', 'fmri'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco,eeg,fmri)')

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
    parser.add_argument('--ema_decay', type=float, default=None)
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=None,
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
    if parsed_args.dataset in ['stocks', 'stock', 'sine', 'energy', 'mujoco', 'eeg', 'fmri']:
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
                                choices=['a2', 'f3', 'pro4', 'max1', 'max2', 'max3', 'auto'],
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
                                 'nn5_daily', 'fred_md', 'sine', 'energy', 'mujoco', 'stocks',
                                 'eeg', 'fmri'], help='training dataset')

    parser.add_argument('--seq_len', type=int,
                        help='input sequence length,'
                             ' only needed if using short-term datasets(stocks,sine,energy,mujoco,eeg,fmri)')

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
    parser.add_argument('--ema_decay', type=float, default=None)
    parser.add_argument('--ema_warmup', type=int, help='ema warmup')

    # --- logging ---
    parser.add_argument('--logging_iter', type=int, default=None,
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
