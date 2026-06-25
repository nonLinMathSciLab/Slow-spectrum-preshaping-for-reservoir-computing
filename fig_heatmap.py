from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dysts.flows import Lorenz
from fig2_3_4 import FULL_EXPERIMENT_CONFIG, FULL_PARAMS, FIGURE_VARIANTS, SMOKE_EXPERIMENT_CONFIG, SMOKE_PARAMS
from grid_crc import CESNComprehensiveSweepV3, NumpyComplexEncoder
from plot_style import apply_publication_style
from simple_table import read_csv, sorted_unique, write_csv


PUBLIC_DIR = Path(__file__).resolve().parent
DEFAULT_DYSTS_RESULTS_ROOT = PUBLIC_DIR / "results" / "dysts_heatmap"
DEFAULT_PARAMETER_RESULTS_ROOT = PUBLIC_DIR / "results" / "parameter_heatmap"
DYSTS_PER_SEED_CSV = "dysts_heatmap_per_seed_results.csv"
DYSTS_SUMMARY_CSV = "dysts_heatmap_summary.csv"
DYSTS_HEATMAP_SVG = "dysts_heatmap_vpt.svg"
DYSTS_HEATMAP_PNG = "dysts_heatmap_vpt.png"
DYSTS_HEATMAP_COMPACT_3COL_SVG = "dysts_heatmap_vpt_compact_3col.svg"
DYSTS_HEATMAP_COMPACT_3COL_PNG = "dysts_heatmap_vpt_compact_3col.png"
DYSTS_HEATMAP_COMPACT_4COL_SVG = "dysts_heatmap_vpt_compact_4col.svg"
DYSTS_HEATMAP_COMPACT_4COL_PNG = "dysts_heatmap_vpt_compact_4col.png"
LEGACY_DYSTS_PER_SEED_CSV = "per_seed_results.csv"
LEGACY_DYSTS_SUMMARY_CSV = "heatmap_summary.csv"
PARAMETER_PER_SEED_CSV = "parameter_heatmap_per_seed_results.csv"
PARAMETER_SUMMARY_CSV = "parameter_heatmap_summary.csv"
PARAMETER_VPT_SVG = "parameter_heatmap_vpt.svg"
PARAMETER_VPT_PNG = "parameter_heatmap_vpt.png"
PARAMETER_LYAPUNOV_ERROR_SVG = "parameter_heatmap_lyapunov_error.svg"
PARAMETER_LYAPUNOV_ERROR_PNG = "parameter_heatmap_lyapunov_error.png"
LYAPUNOV_ERROR_EPSILON = -0.05

apply_publication_style()


def dysts_force_updates_for_dimension(embedding_dimension: int) -> list:
    forced_pair = ((0, 1), np.cos(np.pi / 32) + 1j * np.sin(np.pi / 32))
    if embedding_dimension >= 4:
        return [forced_pair, (2, 1.0)]
    return [forced_pair]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return dysts_heatmap_main(["--dry-run"])
    if argv[0] in {"dysts", "dysts_heatmap"}:
        return dysts_heatmap_main(argv[1:])
    if argv[0] in {"parameter", "parameter_heatmap"}:
        return parameter_heatmap_main(argv[1:])
    raise SystemExit(f"Unknown mode: {argv[0]}. Use 'dysts_heatmap' or 'parameter_heatmap'.")


def dysts_heatmap_main(argv=None):
    parser = argparse.ArgumentParser(description="Run and plot the forced-eigenvec Dysts scalar-observation heatmap.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to a timestamped results directory.")
    parser.add_argument("--dysts-data-dir", type=Path, default=None, help="Directory containing chaotic_attractors.json and cached *_30000.npy files.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per system/observed-index cell.")
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--full", action="store_true", help="Use Fig. 2-4 full-scale train/predict lengths. This is the default.")
    scale_group.add_argument("--smoke", action="store_true", help="Use the reduced smoke-test train/predict lengths.")
    parser.add_argument("--max-observed-dims", type=int, default=4, help="Number of observed-index heatmap columns.")
    parser.add_argument("--systems", nargs="*", default=None, help="Optional explicit system-name subset.")
    parser.add_argument("--limit-systems", type=int, default=None, help="Optional limit for quick checks.")
    parser.add_argument("--start-system", default=None, help="Resume from this system name after candidate filtering.")
    parser.add_argument("--dry-run", action="store_true", help="List candidate systems without running CESN experiments.")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate summary CSV and heatmap from per_seed_results.csv.")
    parser.add_argument("--force", action="store_true", help="Re-run rows that already exist in per_seed_results.csv.")
    parser.add_argument("--vmin", type=float, default=0.0, help="Heatmap color lower bound.")
    parser.add_argument("--vmax", type=float, default=None, help="Heatmap color upper bound. Defaults to the data maximum.")
    parser.add_argument("--cmap", default="YlGnBu", help="Matplotlib colormap for VPT.")
    parser.add_argument(
        "--invariant-dim-mode",
        choices=["system", "fig234"],
        default="system",
        help="Use each system dimension for eigenvector input subspaces, or keep Fig. 2-4 invariant_dim=3.",
    )
    args = parser.parse_args(argv)

    full = not args.smoke
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _new_dysts_output_dir()
    data_dir = resolve_dysts_data_dir(args.dysts_data_dir)
    metadata = load_dysts_metadata(data_dir)
    candidates = select_dysts_candidates(
        metadata=metadata,
        data_dir=data_dir,
        max_observed_dims=args.max_observed_dims,
        systems=args.systems,
        limit=args.limit_systems,
        start_system=args.start_system,
    )

    if args.dry_run:
        print_candidate_summary(candidates, data_dir)
        return candidates

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    run_metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "dysts_data_dir": str(data_dir),
        "seeds": args.seeds,
        "full_scale": full,
        "max_observed_dims": args.max_observed_dims,
        "variant": "forced_eigenvec",
        "invariant_dim_mode": args.invariant_dim_mode,
        "candidate_count": len(candidates),
        "systems": [candidate["system"] for candidate in candidates],
        "notes": [
            "ESN train/predict parameters match Fig. 2-4 forced_eigenvec.",
            "Forced reservoir eigenvalues are dimension-dependent: 3D uses the conjugate pair only; 4D also fixes eigenvalue index 2 to 1.0.",
            "GSPT and Lyapunov diagnostic plots are disabled; the heatmap metric is VPT only.",
            "Cells whose observed index exceeds the system dimension are left blank.",
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, cls=NumpyComplexEncoder)

    per_seed_path = _preferred_csv_path(output_dir, DYSTS_PER_SEED_CSV, LEGACY_DYSTS_PER_SEED_CSV)
    if not args.plot_only:
        run_dysts_heatmap_experiment(
            candidates=candidates,
            data_dir=data_dir,
            output_dir=output_dir,
            per_seed_path=per_seed_path,
            seeds=args.seeds,
            full=full,
            max_observed_dims=args.max_observed_dims,
            force=args.force,
            invariant_dim_mode=args.invariant_dim_mode,
        )

    rows = load_csv_rows(per_seed_path)
    summary_rows = summarize_heatmap_rows(candidates, rows, args.seeds, args.max_observed_dims)
    write_csv(output_dir / DYSTS_SUMMARY_CSV, summary_rows)
    write_csv(output_dir / LEGACY_DYSTS_SUMMARY_CSV, summary_rows)
    plot_dysts_vpt_heatmap(
        candidates=candidates,
        summary_rows=summary_rows,
        output_dir=output_dir,
        max_observed_dims=args.max_observed_dims,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
    )
    plot_dysts_vpt_heatmap_compact(
        candidates=candidates,
        summary_rows=summary_rows,
        output_dir=output_dir,
        max_observed_dims=args.max_observed_dims,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        block_columns=3,
        save_svg=DYSTS_HEATMAP_COMPACT_3COL_SVG,
        save_png=DYSTS_HEATMAP_COMPACT_3COL_PNG,
    )
    plot_dysts_vpt_heatmap_compact(
        candidates=candidates,
        summary_rows=summary_rows,
        output_dir=output_dir,
        max_observed_dims=args.max_observed_dims,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        block_columns=4,
        save_svg=DYSTS_HEATMAP_COMPACT_4COL_SVG,
        save_png=DYSTS_HEATMAP_COMPACT_4COL_PNG,
    )

    print(f"Generated Dysts forced-eigenvec heatmap outputs in {output_dir}")
    return output_dir


def parameter_heatmap_main(argv=None):
    parser = argparse.ArgumentParser(description="Run and plot the Lorenz parameter heatmap experiment.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to a timestamped parameter_heatmap directory.")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate summary CSV and heatmaps from existing per-seed CSV.")
    parser.add_argument("--force", action="store_true", help="Re-run rows that already exist in the per-seed CSV.")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny subset for code-path verification.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on new per-seed experiments.")
    parser.add_argument("--vpt-vmin", type=float, default=0.0)
    parser.add_argument("--vpt-vmax", type=float, default=10.0)
    parser.add_argument("--lyapunov-error-vmin", type=float, default=0.0)
    parser.add_argument("--lyapunov-error-vmax", type=float, default=None)
    parser.add_argument("--lyapunov-error-epsilon", type=float, default=LYAPUNOV_ERROR_EPSILON)
    parser.add_argument("--cmap-vpt", default="YlGnBu")
    parser.add_argument("--cmap-lyapunov-error", default="YlOrRd")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve() if args.output_dir is not None else _new_parameter_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_ranges = make_parameter_heatmap_ranges(smoke=args.smoke)
    run_metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "heatmap_type": "parameter_heatmap",
        "smoke": args.smoke,
        "lyapunov_error_formula": (
            "sum_{lambda_i >= epsilon} |lambda_i - lambda_hat_i| + "
            "sum_{lambda_i < epsilon} |lambda_i - lambda_hat_i| / |lambda_i|"
        ),
        "lyapunov_error_epsilon": args.lyapunov_error_epsilon,
        "experiment_config": make_parameter_heatmap_experiment_config(),
        "plot_config": make_vpt_only_plot_config(),
        "parameter_ranges": parameter_ranges,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, cls=NumpyComplexEncoder)

    per_seed_path = output_dir / PARAMETER_PER_SEED_CSV
    if not args.plot_only:
        run_parameter_heatmap_experiment(
            output_dir=output_dir,
            per_seed_path=per_seed_path,
            parameter_ranges=parameter_ranges,
            force=args.force,
            max_runs=args.max_runs,
            lyapunov_error_epsilon=args.lyapunov_error_epsilon,
        )

    rows = load_csv_rows(per_seed_path)
    summary_rows = summarize_parameter_rows(rows)
    write_csv(output_dir / PARAMETER_SUMMARY_CSV, summary_rows)
    plot_parameter_heatmap_grid6d(
        summary_rows=summary_rows,
        output_dir=output_dir,
        metric="mean_vpt",
        cbar_label="Mean VPT",
        save_svg=PARAMETER_VPT_SVG,
        save_png=PARAMETER_VPT_PNG,
        vmin=args.vpt_vmin,
        vmax=args.vpt_vmax,
        cmap=args.cmap_vpt,
    )
    plot_parameter_heatmap_grid6d(
        summary_rows=summary_rows,
        output_dir=output_dir,
        metric="mean_lyapunov_error",
        cbar_label="Lyapunov error",
        save_svg=PARAMETER_LYAPUNOV_ERROR_SVG,
        save_png=PARAMETER_LYAPUNOV_ERROR_PNG,
        vmin=args.lyapunov_error_vmin,
        vmax=args.lyapunov_error_vmax,
        cmap=args.cmap_lyapunov_error,
    )

    print(f"Generated parameter_heatmap outputs in {output_dir}")
    return output_dir


def make_parameter_heatmap_experiment_config() -> dict:
    experiment_config = {
        "FIRST_STEP_DT": 0.001,
        "PTS_PER_PERIOD": 100,
        "delay": 0,
        "NUM_STEPS": 30000,
        "DataStartTime": 0,
        "embedding_dimension": 3,
        "WARMUP_STEPS": 1000,
        "TRAINING_TIME_STEPS": 2000,
        "PREDICTION_STEPS": 5000,
        "PRED_START_POINT": 0,
        "PRED_START_POINT_END": 5000,
        "OVER_NUM": 50,
        "FORCE_COMPLEX_MODE": False,
        "system_name": "Lorenz",
        "sanity_check": True,
        "test_GSPT": True,
        "calc_only_ls": True,
        "test_conditional_lyapunov": False,
        "search_method": "grid",
        "FIX_DT": None,
        "gpu_mode": cupy_available(),
    }
    experiment_config["USE_DATA_LEN"] = (
        experiment_config["delay"] * experiment_config["embedding_dimension"]
        + experiment_config["WARMUP_STEPS"]
        + experiment_config["TRAINING_TIME_STEPS"]
        + experiment_config["PRED_START_POINT_END"]
        + experiment_config["PREDICTION_STEPS"]
    )
    return experiment_config


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def make_parameter_heatmap_ranges(smoke: bool = False) -> dict:
    force_updates = [
        False,
        [((0, 1), np.cos(np.pi / 16) + 1j * np.sin(np.pi / 16))],
        [((0, 1), np.cos(np.pi / 32) + 1j * np.sin(np.pi / 32))],
        [((0, 1), np.cos(np.pi / 64) + 1j * np.sin(np.pi / 64))],
        [(0, 1.01), (1, 1 / 1.01)],
        [(0, 1.05), (1, 1 / 1.05)],
        [(0, 1.1), (1, 1 / 1.1)],
    ]
    ranges = {
        "partial_observe_list": [[[0, 0]]],
        "w_res_method_list": ["symmetry"],
        "er_prob_list": [1.0],
        "W_res_force_updates_list": force_updates,
        "w_in_method_list": ["eigenvec", "random"],
        "use_eigenvecs_method_list": ["high"],
        "invariant_dim_list": [3],
        "conj_pair_num_list": ["auto"],
        "construct_option_list": ["random"],
        "input_scaling_ratio_list": [0.5, 0.75, 1.0, 1.25, 1.5],
        "input_sparsity_list": [1.0],
        "bias_method_list": ["const"],
        "seed_list": list(range(20)),
        "bias_amplitude_list": [0.01],
        "alpha_list": [0.2, 0.4, 0.6, 0.8, 1.0],
        "dim_reservoir_list": [300],
        "spectral_radius_list": [0.1, 0.3, 0.5, 0.7, 0.9],
        "regularization_list": [0.0, 1e-24, 1e-20, 1e-16, 1e-12],
        "is_intercept_list": [False],
        "how_loop_list": ["output=input"],
        "how_train_list": ["lstsq"],
    }
    if smoke:
        ranges["W_res_force_updates_list"] = [False, force_updates[2]]
        ranges["w_in_method_list"] = ["eigenvec", "random"]
        ranges["input_scaling_ratio_list"] = [0.5, 1.0]
        ranges["seed_list"] = [0]
        ranges["alpha_list"] = [0.5, 1.0]
        ranges["spectral_radius_list"] = [0.3, 0.6]
        ranges["regularization_list"] = [0.0, 1e-16]
    return ranges


def iter_parameter_heatmap_params(parameter_ranges: dict):
    keys = [
        "partial_observe_list",
        "w_res_method_list",
        "er_prob_list",
        "W_res_force_updates_list",
        "w_in_method_list",
        "use_eigenvecs_method_list",
        "invariant_dim_list",
        "conj_pair_num_list",
        "construct_option_list",
        "input_scaling_ratio_list",
        "input_sparsity_list",
        "bias_method_list",
        "seed_list",
        "bias_amplitude_list",
        "alpha_list",
        "dim_reservoir_list",
        "spectral_radius_list",
        "regularization_list",
        "is_intercept_list",
        "how_loop_list",
        "how_train_list",
    ]
    for values in itertools.product(*(parameter_ranges[key] for key in keys)):
        record = dict(zip(keys, values))
        w_in_method = record["w_in_method_list"]
        params = {
            "partial_observe": record["partial_observe_list"],
            "w_res_method": record["w_res_method_list"],
            "er_prob": record["er_prob_list"],
            "W_res_force_updates": record["W_res_force_updates_list"],
            "w_in_method": w_in_method,
            "use_eigenvecs_method_or_list": record["use_eigenvecs_method_list"],
            "invariant_dim": record["invariant_dim_list"],
            "conj_pair_num": record["conj_pair_num_list"],
            "construct_option": record["construct_option_list"],
            "input_scaling_ratio": record["input_scaling_ratio_list"],
            "input_sparsity": record["input_sparsity_list"],
            "bias_method": record["bias_method_list"],
            "seed": record["seed_list"],
            "bias_amplitude": record["bias_amplitude_list"],
            "alpha": record["alpha_list"],
            "dim_reservoir": record["dim_reservoir_list"],
            "spectral_radius": record["spectral_radius_list"],
            "regularization": record["regularization_list"],
            "is_intercept": record["is_intercept_list"],
            "how_loop": record["how_loop_list"],
            "how_train": record["how_train_list"],
        }
        if w_in_method == "random":
            params["use_eigenvecs_method_or_list"] = None
            params["conj_pair_num"] = None
        yield params


def run_parameter_heatmap_experiment(
    output_dir: Path,
    per_seed_path: Path,
    parameter_ranges: dict,
    force: bool,
    max_runs: int | None,
    lyapunov_error_epsilon: float,
) -> None:
    rows = [] if force else load_csv_rows(per_seed_path)
    completed = {
        parameter_row_key(row)
        for row in rows
        if row.get("success") is True
    }
    analyzer = CESNComprehensiveSweepV3(
        dynamical_system=Lorenz,
        destination_dir=output_dir,
        experiment_config=make_parameter_heatmap_experiment_config(),
        plot_config=make_vpt_only_plot_config(),
        parameter_ranges=parameter_ranges,
        vander_degree=0,
    )
    analyzer._setup_base_data()
    new_runs = 0
    for params in iter_parameter_heatmap_params(parameter_ranges):
        label = w_res_force_update_label(params["W_res_force_updates"])
        key = parameter_params_key(params, label)
        if key in completed:
            continue
        result = analyzer.run_single_experiment(params)
        row = result_to_parameter_row(result, params, label, analyzer, lyapunov_error_epsilon)
        rows.append(row)
        write_csv(per_seed_path, rows)
        status = "ok" if row["success"] else "failed"
        print(
            f"{status}: w_in={row['w_in_method']} W={row['W_res_force_label']} "
            f"alpha={row['alpha']} sr={row['spectral_radius']} reg={row['regularization']} "
            f"isr={row['input_scaling_ratio']} seed={row['seed']} "
            f"vpt={row['vpt_mean']} lyaperr={row['lyapunov_error_custom']}"
        )
        new_runs += 1
        if max_runs is not None and new_runs >= max_runs:
            break


def result_to_parameter_row(
    result: dict,
    params: dict,
    w_res_force_label: str,
    analyzer: CESNComprehensiveSweepV3,
    lyapunov_error_epsilon: float,
) -> dict:
    success = bool(result.get("success"))
    custom_error = lyapunov_error_from_spectra(
        true_le=analyzer.true_le,
        estimated_le=result.get("lyapunov_spectrum"),
        epsilon=lyapunov_error_epsilon,
    )
    row = {
        "partial_observe": str(params["partial_observe"]),
        "w_res_method": params["w_res_method"],
        "W_res_force_label": w_res_force_label,
        "W_res_force_updates": str(params["W_res_force_updates"]),
        "w_in_method": params["w_in_method"],
        "result_w_in_method": result.get("w_in_method", ""),
        "use_eigenvecs_method": params["use_eigenvecs_method_or_list"] or "",
        "invariant_dim": params["invariant_dim"],
        "conj_pair_num": params["conj_pair_num"] or "",
        "construct_option": params["construct_option"],
        "input_scaling_ratio": params["input_scaling_ratio"],
        "input_sparsity": params["input_sparsity"],
        "bias_method": params["bias_method"],
        "seed": params["seed"],
        "bias_amplitude": params["bias_amplitude"],
        "alpha": params["alpha"],
        "dim_reservoir": params["dim_reservoir"],
        "spectral_radius": params["spectral_radius"],
        "regularization": params["regularization"],
        "is_intercept": params["is_intercept"],
        "how_loop": params["how_loop"],
        "how_train": params["how_train"],
        "success": success,
        "vpt_mean": _finite_float(result.get("prediction_horizon_average")),
        "vpt_min": _finite_float(result.get("min_prediction_horizon")),
        "lyapunov_error_custom": _finite_float(custom_error),
        "lyapunov_error_original": _finite_float(result.get("lyapunov_error")),
        "lyapunov_s_error_original": _finite_float(result.get("lyapunov_s_error")),
        "lyapunov_spectrum": str(np.asarray(result.get("lyapunov_spectrum"), dtype=float).tolist()) if result.get("lyapunov_spectrum") is not None else "",
        "true_lyapunov_spectrum": str(np.asarray(analyzer.true_le, dtype=float).tolist()),
        "lyapunov_error_epsilon": lyapunov_error_epsilon,
        "lyapunov_time": _finite_float(analyzer.lyapunov_time),
        "dt": _finite_float(analyzer.dt),
        "error": "" if success else str(result.get("error", "")),
    }
    if not success and result.get("error_traceback"):
        error_dir = analyzer.results_dir / "parameter_heatmap_errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / (
            f"{w_res_force_label}_w_in_{params['w_in_method']}_alpha_{params['alpha']}"
            f"_sr_{params['spectral_radius']}_reg_{params['regularization']}"
            f"_isr_{params['input_scaling_ratio']}_seed_{params['seed']}.txt"
        )
        error_path.write_text(result["error_traceback"], encoding="utf-8")
        row["error_traceback_path"] = str(error_path)
    return row


def lyapunov_error_from_spectra(true_le, estimated_le, epsilon: float) -> float | None:
    if true_le is None or estimated_le is None:
        return None
    true_arr = np.asarray(true_le, dtype=float).ravel()
    estimated_arr = np.asarray(estimated_le, dtype=float).ravel()
    width = min(len(true_arr), len(estimated_arr))
    if width == 0:
        return None
    error = 0.0
    for true_value, estimated_value in zip(true_arr[:width], estimated_arr[:width]):
        diff = abs(true_value - estimated_value)
        if true_value >= epsilon:
            error += diff
        else:
            error += diff / max(abs(true_value), 1e-12)
    return float(error)


def summarize_parameter_rows(rows: list[dict]) -> list[dict]:
    group_cols = [
        "w_in_method",
        "W_res_force_label",
        "regularization",
        "input_scaling_ratio",
        "alpha",
        "spectral_radius",
    ]
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(col) for col in group_cols)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, cell_rows in grouped.items():
        success_rows = [row for row in cell_rows if row.get("success") is True]
        vpt_values = np.asarray([float(row["vpt_mean"]) for row in success_rows if _is_finite(row.get("vpt_mean"))], dtype=float)
        lyap_values = np.asarray(
            [float(row["lyapunov_error_custom"]) for row in success_rows if _is_finite(row.get("lyapunov_error_custom"))],
            dtype=float,
        )
        record = dict(zip(group_cols, key))
        record.update(
            {
                "mean_vpt": float(np.mean(vpt_values)) if vpt_values.size else "",
                "std_vpt": float(np.std(vpt_values)) if vpt_values.size else "",
                "min_vpt": float(np.min(vpt_values)) if vpt_values.size else "",
                "max_vpt": float(np.max(vpt_values)) if vpt_values.size else "",
                "mean_lyapunov_error": float(np.mean(lyap_values)) if lyap_values.size else "",
                "std_lyapunov_error": float(np.std(lyap_values)) if lyap_values.size else "",
                "min_lyapunov_error": float(np.min(lyap_values)) if lyap_values.size else "",
                "max_lyapunov_error": float(np.max(lyap_values)) if lyap_values.size else "",
                "success_count": len(success_rows),
                "failure_count": len(cell_rows) - len(success_rows),
            }
        )
        summary_rows.append(record)
    summary_rows.sort(key=lambda row: parameter_summary_sort_key(row))
    return summary_rows


def plot_parameter_heatmap_grid6d(
    summary_rows: list[dict],
    output_dir: Path,
    metric: str,
    cbar_label: str,
    save_svg: str,
    save_png: str,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
) -> None:
    if not summary_rows:
        raise ValueError("No summary rows are available for parameter_heatmap plotting")
    values1 = sorted_unique([row["alpha"] for row in summary_rows])
    values2 = sorted_unique([row["spectral_radius"] for row in summary_rows])
    values3 = sorted_unique([row["regularization"] for row in summary_rows])
    values4 = sorted_unique([row["input_scaling_ratio"] for row in summary_rows])
    values5 = sorted_unique([row["w_in_method"] for row in summary_rows])
    values6 = sorted([value for value in sorted_unique([row["W_res_force_label"] for row in summary_rows])], key=w_res_force_label_rank)

    finite_values = [float(row[metric]) for row in summary_rows if _is_finite(row.get(metric))]
    if vmin is None and finite_values:
        vmin = min(finite_values)
    if vmax is None and finite_values:
        vmax = max(finite_values)
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    lookup = {
        (
            row["w_in_method"],
            row["W_res_force_label"],
            row["regularization"],
            row["input_scaling_ratio"],
            row["alpha"],
            row["spectral_radius"],
        ): row
        for row in summary_rows
    }
    n_row3 = len(values3)
    n_col4 = len(values4)
    n_col5 = len(values5)
    n_row6 = len(values6)
    fig = plt.figure(figsize=(1.75 * n_col4 * n_col5, 1.35 * n_row3 * n_row6), layout="constrained")
    subfigs = fig.subfigures(n_row6, n_col5, wspace=0.0, hspace=0.0)
    subfigs = _as_2d_subfigures(subfigs, n_row6, n_col5)
    image = None
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    for i, value6 in enumerate(values6):
        for j, value5 in enumerate(values5):
            subfig = subfigs[i, j]
            if i == 0:
                subfig.suptitle(rf"$w_{{\mathrm{{in}}}}$: $\mathbf{{{value5}}}$", fontsize=15)
            if j == 0:
                subfig.supylabel(_format_w_res_force_label(value6), fontsize=15, fontweight="bold")
            axes = subfig.subplots(n_row3, n_col4, squeeze=False, gridspec_kw={"wspace": 0.012, "hspace": 0.001})
            for r, value3 in enumerate(values3):
                for c, value4 in enumerate(values4):
                    ax = axes[r, c]
                    matrix = np.full((len(values1), len(values2)), np.nan)
                    for row_idx, value1 in enumerate(values1):
                        for col_idx, value2 in enumerate(values2):
                            row = lookup.get((value5, value6, value3, value4, value1, value2))
                            if row is not None and _is_finite(row.get(metric)):
                                matrix[row_idx, col_idx] = float(row[metric])
                    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
                    ax.minorticks_off()
                    show_top_gamma = i == 0 and r == 0
                    show_bottom_rho = r == n_row3 - 1
                    ax.set_xticks(np.arange(len(values2)) if show_bottom_rho else [])
                    ax.set_yticks(np.arange(len(values1)) if c == 0 else [])
                    if show_bottom_rho:
                        ax.set_xticklabels([str(value) for value in values2], rotation=60, ha="right", fontsize=9, fontweight="bold")
                        ax.set_xlabel(r"$\rho_A$", fontsize=12, labelpad=1)
                    if c == 0:
                        ax.set_yticklabels([str(value) for value in values1], fontsize=9, fontweight="bold")
                        ax.set_ylabel(r"$\alpha$", fontsize=12, labelpad=1)
                    if show_top_gamma:
                        ax.set_title(rf"$\gamma=\mathbf{{{_format_math_number(value4)}}}$", fontsize=11, fontweight="bold", pad=1)
                    if c == 0:
                        ax.annotate(
                            rf"$\beta=\mathbf{{{_format_regularization_math(value3)}}}$",
                            xy=(0, 0.5),
                            xytext=(-24, 0),
                            xycoords=ax.yaxis.label,
                            textcoords="offset points",
                            ha="right",
                            va="center",
                            rotation=90,
                            fontsize=10,
                        )

    if image is not None:
        cbar_ax = fig.add_axes([1.01, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=36, fontweight="bold")
        cbar.ax.tick_params(labelsize=28)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        cbar.ax.minorticks_off()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / save_svg, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parameter_params_key(params: dict, w_res_force_label: str) -> tuple:
    return (
        params["w_in_method"],
        w_res_force_label,
        float(params["regularization"]),
        float(params["input_scaling_ratio"]),
        float(params["alpha"]),
        float(params["spectral_radius"]),
        int(params["seed"]),
    )


def parameter_row_key(row: dict) -> tuple:
    return (
        row.get("w_in_method"),
        row.get("W_res_force_label"),
        float(row.get("regularization")),
        float(row.get("input_scaling_ratio")),
        float(row.get("alpha")),
        float(row.get("spectral_radius")),
        int(row.get("seed")),
    )


def parameter_summary_sort_key(row: dict) -> tuple:
    return (
        str(row.get("w_in_method")),
        w_res_force_label_rank(str(row.get("W_res_force_label"))),
        float(row.get("regularization")),
        float(row.get("input_scaling_ratio")),
        float(row.get("alpha")),
        float(row.get("spectral_radius")),
    )


def w_res_force_update_label(update) -> str:
    if update is False or update is None:
        return "none"
    if len(update) == 1:
        value = update[0][1]
        angle = float(np.angle(value))
        denom = int(round(np.pi / angle)) if abs(angle) > 1e-12 else 0
        return f"exp_i_pi_{denom}"
    pairs = []
    for index, value in update:
        pairs.append(f"{index}:{float(np.real(value)):.5g}")
    return "real_" + "_".join(pairs)


def w_res_force_label_rank(label: str) -> int:
    order = {
        "none": 0,
        "exp_i_pi_16": 1,
        "exp_i_pi_32": 2,
        "exp_i_pi_64": 3,
        "real_0:1.01_1:0.9901": 4,
        "real_0:1.05_1:0.95238": 5,
        "real_0:1.1_1:0.90909": 6,
    }
    return order.get(str(label), len(order))


def _format_w_res_force_label(label: str) -> str:
    labels = {
        "none": "none",
        "exp_i_pi_16": r"$\pi/16$",
        "exp_i_pi_32": r"$\pi/32$",
        "exp_i_pi_64": r"$\pi/64$",
        "real_0:1.01_1:0.9901": "1.01",
        "real_0:1.05_1:0.95238": "1.05",
        "real_0:1.1_1:0.90909": "1.1",
    }
    return labels.get(str(label), str(label))


def _format_regularization_math(value) -> str:
    value = float(value)
    if value == 0:
        return "0"
    exponent = int(round(np.log10(value)))
    return rf"10^{{{exponent}}}"


def _format_math_number(value) -> str:
    return str(value)


def _as_2d_subfigures(subfigs, n_row: int, n_col: int):
    if n_row == 1 and n_col == 1:
        return np.array([[subfigs]])
    if n_row == 1:
        return np.asarray(subfigs)[np.newaxis, :]
    if n_col == 1:
        return np.asarray(subfigs)[:, np.newaxis]
    return np.asarray(subfigs)


def resolve_dysts_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir.resolve()
    return (PUBLIC_DIR / "dysts" / "data").resolve()


def load_dysts_metadata(data_dir: Path) -> dict:
    metadata_path = data_dir / "chaotic_attractors.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No Dysts metadata found at {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_dysts_candidates(
    metadata: dict,
    data_dir: Path,
    max_observed_dims: int,
    systems: list[str] | None = None,
    limit: int | None = None,
    start_system: str | None = None,
) -> list[dict]:
    requested = set(systems) if systems else None
    candidates = []
    for system, info in metadata.items():
        dim = info.get("embedding_dimension")
        if requested is not None and system not in requested:
            continue
        if info.get("delay", False) or info.get("nonautonomous", False):
            continue
        if not isinstance(dim, int) or dim < 1 or dim > max_observed_dims:
            continue
        data_path = data_dir / f"{system}_30000.npy"
        if not data_path.exists():
            continue
        mle = _metadata_mle(info)
        if mle is None or not np.isfinite(mle) or mle <= 0:
            continue
        candidates.append(
            {
                "system": system,
                "embedding_dimension": dim,
                "metadata": info,
                "data_path": data_path,
                "mle": float(mle),
                "period": float(info["period"]),
            }
        )
    candidates.sort(key=lambda item: item["system"])
    if start_system is not None:
        candidates = [candidate for candidate in candidates if candidate["system"] >= start_system]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def print_candidate_summary(candidates: list[dict], data_dir: Path) -> None:
    counts = {}
    for candidate in candidates:
        dim = candidate["embedding_dimension"]
        counts[dim] = counts.get(dim, 0) + 1
    print(f"Dysts data dir: {data_dir}")
    print(f"Candidate systems: {len(candidates)}")
    print(f"Dimensions: {counts}")
    for candidate in candidates:
        print(f"  {candidate['system']} (dim={candidate['embedding_dimension']})")


def run_dysts_heatmap_experiment(
    candidates: list[dict],
    data_dir: Path,
    output_dir: Path,
    per_seed_path: Path,
    seeds: int,
    full: bool,
    max_observed_dims: int,
    force: bool,
    invariant_dim_mode: str,
) -> None:
    rows = [] if force else load_csv_rows(per_seed_path)
    completed = {
        (str(row.get("system")), int(row.get("observed_index")), int(row.get("seed")))
        for row in rows
        if row.get("success") is True
    }

    for candidate in candidates:
        system = candidate["system"]
        experiment_config = make_dysts_experiment_config(candidate, full)
        analyzer = CESNComprehensiveSweepV3(
            dynamical_system=_named_dummy_system(system),
            destination_dir=output_dir,
            experiment_config=experiment_config,
            plot_config=make_vpt_only_plot_config(),
            parameter_ranges=None,
            vander_degree=0,
        )
        setup_cached_dysts_data(analyzer, candidate, data_dir)
        for observed_zero in range(max_observed_dims):
            observed_index = observed_zero + 1
            if observed_zero >= candidate["embedding_dimension"]:
                continue
            for seed in range(seeds):
                key = (system, observed_index, seed)
                if key in completed:
                    continue
                params = make_dysts_forced_eigenvec_params(candidate, observed_zero, seed, full, invariant_dim_mode)
                result = analyzer.run_single_experiment(params)
                row = result_to_heatmap_row(candidate, observed_index, seed, params, result, analyzer)
                rows.append(row)
                write_csv(per_seed_path, rows)
                status = "ok" if row["success"] else "failed"
                print(f"{status}: {system} observed_index={observed_index} seed={seed} vpt={row['vpt_mean']}")


def make_dysts_experiment_config(candidate: dict, full: bool) -> dict:
    experiment_config = dict(FULL_EXPERIMENT_CONFIG if full else SMOKE_EXPERIMENT_CONFIG)
    experiment_config["embedding_dimension"] = candidate["embedding_dimension"]
    experiment_config["USE_DATA_LEN"] = (
        experiment_config["delay"] * experiment_config["embedding_dimension"]
        + experiment_config["WARMUP_STEPS"]
        + experiment_config["TRAINING_TIME_STEPS"]
        + experiment_config["PRED_START_POINT_END"]
        + experiment_config["PREDICTION_STEPS"]
    )
    experiment_config["system_name"] = candidate["system"]
    experiment_config["sanity_check"] = not full
    experiment_config["test_GSPT"] = False
    experiment_config["test_conditional_lyapunov"] = False
    return experiment_config


def make_vpt_only_plot_config() -> dict:
    return {
        "mode_plot_stability": False,
        "mode_plot_projection": False,
        "mode_plot_distance": False,
        "mode_plot_eigenvalue_distribution": False,
        "mode_plot_fxp_eigenvalue_distribution": False,
        "plot_results": False,
    }


def setup_cached_dysts_data(analyzer: CESNComprehensiveSweepV3, candidate: dict, data_dir: Path) -> None:
    raw_data = np.load(candidate["data_path"])
    use_len = analyzer.experiment_config["USE_DATA_LEN"]
    if len(raw_data) < use_len:
        raise ValueError(f"{candidate['system']} cache is too short: {len(raw_data)} < {use_len}")

    use_data = np.asarray(raw_data[:use_len], dtype=float)
    data_min = np.min(use_data, axis=0)
    data_max = np.max(use_data, axis=0)
    data_range = np.where(np.abs(data_max - data_min) < 1e-12, 1.0, data_max - data_min)
    analyzer.normalized_data = (use_data - np.mean(use_data, axis=0)) * 2.0 / data_range
    analyzer.true_le = _metadata_true_le(candidate["metadata"])
    analyzer.dt = candidate["period"] / analyzer.experiment_config["PTS_PER_PERIOD"]
    analyzer.lyapunov_time = 1.0 / (candidate["mle"] * analyzer.dt)
    analyzer.system = candidate["system"]
    analyzer.experiment_config["lyapunov_time"] = analyzer.lyapunov_time
    analyzer.experiment_config["system"] = candidate["system"]
    analyzer.experiment_config["dt"] = analyzer.dt
    analyzer.dim_target_system = analyzer.normalized_data.shape[1]

    conditions_dir = analyzer.results_dir / "conditions"
    conditions_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = conditions_dir / f"{candidate['system']}.json"
    with conditions_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dysts_data_dir": str(data_dir),
                "experiment_config": analyzer.experiment_config,
                "plot_config": analyzer.plot_config,
            },
            f,
            indent=2,
            cls=NumpyComplexEncoder,
        )


def make_dysts_forced_eigenvec_params(
    candidate: dict,
    observed_zero: int,
    seed: int,
    full: bool,
    invariant_dim_mode: str,
) -> dict:
    params = dict(FULL_PARAMS if full else SMOKE_PARAMS)
    params.update(FIGURE_VARIANTS["forced_eigenvec"]["overrides"])
    params["partial_observe"] = [[observed_zero, 0]]
    params["seed"] = seed
    params["W_res_force_updates"] = dysts_force_updates_for_dimension(candidate["embedding_dimension"])
    if invariant_dim_mode == "system":
        params["invariant_dim"] = candidate["embedding_dimension"]
    return params


def result_to_heatmap_row(
    candidate: dict,
    observed_index: int,
    seed: int,
    params: dict,
    result: dict,
    analyzer: CESNComprehensiveSweepV3,
) -> dict:
    success = bool(result.get("success"))
    row = {
        "system": candidate["system"],
        "embedding_dimension": candidate["embedding_dimension"],
        "observed_index": observed_index,
        "observed_label": f"Observed index {observed_index}",
        "seed": seed,
        "W_res_force_updates": str(params["W_res_force_updates"]),
        "w_in_method": params["w_in_method"],
        "use_eigenvecs_method_or_list": params["use_eigenvecs_method_or_list"],
        "invariant_dim": params["invariant_dim"],
        "construct_option": params["construct_option"],
        "success": success,
        "vpt_mean": _finite_float(result.get("prediction_horizon_average")),
        "vpt_min": _finite_float(result.get("min_prediction_horizon")),
        "vpt_smape_mean": _finite_float(result.get("prediction_horizon_smape_average")),
        "vpt_smape_min": _finite_float(result.get("min_prediction_horizon_smape")),
        "lyapunov_time": _finite_float(analyzer.lyapunov_time),
        "dt": _finite_float(analyzer.dt),
        "error": "" if success else str(result.get("error", "")),
    }
    if not success and result.get("error_traceback"):
        error_dir = analyzer.results_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / f"{candidate['system']}_observed_{observed_index}_seed_{seed}.txt"
        error_path.write_text(result["error_traceback"], encoding="utf-8")
        row["error_traceback_path"] = str(error_path)
    return row


def summarize_heatmap_rows(
    candidates: list[dict],
    rows: list[dict],
    seeds: int,
    max_observed_dims: int,
) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (str(row.get("system")), int(row.get("observed_index")))
        grouped.setdefault(key, []).append(row)

    summary = []
    for candidate in candidates:
        system = candidate["system"]
        dim = candidate["embedding_dimension"]
        for observed_index in range(1, max_observed_dims + 1):
            key = (system, observed_index)
            cell_rows = grouped.get(key, [])
            success_rows = [row for row in cell_rows if row.get("success") is True]
            values = np.asarray([float(row["vpt_mean"]) for row in success_rows if _is_finite(row.get("vpt_mean"))], dtype=float)
            missing_dimension = observed_index > dim
            summary.append(
                {
                    "system": system,
                    "embedding_dimension": dim,
                    "observed_index": observed_index,
                    "observed_label": f"Observed index {observed_index}",
                    "mean_vpt": float(np.mean(values)) if values.size else "",
                    "std_vpt": float(np.std(values)) if values.size else "",
                    "min_vpt": float(np.min(values)) if values.size else "",
                    "max_vpt": float(np.max(values)) if values.size else "",
                    "success_count": int(len(success_rows)),
                    "failure_count": int(len(cell_rows) - len(success_rows)),
                    "expected_seed_count": 0 if missing_dimension else seeds,
                    "status": "missing_dimension" if missing_dimension else ("complete" if len(success_rows) >= seeds else "partial"),
                }
            )
    return summary


def plot_dysts_vpt_heatmap(
    candidates: list[dict],
    summary_rows: list[dict],
    output_dir: Path,
    max_observed_dims: int,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
) -> None:
    systems = [candidate["system"] for candidate in candidates]
    system_index = {system: idx for idx, system in enumerate(systems)}
    matrix = np.full((len(systems), max_observed_dims), np.nan)
    for row in summary_rows:
        value = row.get("mean_vpt")
        if value == "" or value is None:
            continue
        i = system_index[str(row["system"])]
        j = int(row["observed_index"]) - 1
        matrix[i, j] = float(value)

    masked = np.ma.masked_invalid(matrix)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    finite = matrix[np.isfinite(matrix)]
    vmax_plot = vmax
    if vmax_plot is None and finite.size:
        vmax_plot = float(np.max(finite))
    if vmax_plot is None:
        vmax_plot = 1.0

    height = max(8.0, 0.28 * max(len(systems), 1))
    fig, ax = plt.subplots(figsize=(7.2, height))
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap_obj, vmin=vmin, vmax=vmax_plot)
    ax.set_xticks(np.arange(max_observed_dims))
    ax.set_xticklabels([f"Observed index {idx}" for idx in range(1, max_observed_dims + 1)], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels(systems)
    y_label_size = 9 if len(systems) > 60 else 12
    ax.tick_params(axis="y", labelsize=y_label_size)
    ax.tick_params(axis="x", labelsize=13)
    ax.minorticks_off()
    for label in ax.get_yticklabels():
        label.set_fontsize(y_label_size)
        label.set_fontweight("bold")
    ax.set_xlabel("Scalar observation", fontsize=16)
    ax.set_ylabel("")
    ax.set_title("")
    for boundary in np.arange(-0.5, max_observed_dims, 1):
        ax.axvline(boundary, color="white", linewidth=0.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    colorbar.set_label("Mean VPT", fontsize=14)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(output_dir / DYSTS_HEATMAP_SVG, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / DYSTS_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dysts_vpt_heatmap_compact(
    candidates: list[dict],
    summary_rows: list[dict],
    output_dir: Path,
    max_observed_dims: int,
    vmin: float | None,
    vmax: float | None,
    cmap: str,
    block_columns: int,
    save_svg: str,
    save_png: str,
) -> None:
    systems = [candidate["system"] for candidate in candidates]
    if not systems:
        return

    system_index = {system: idx for idx, system in enumerate(systems)}
    matrix = np.full((len(systems), max_observed_dims), np.nan)
    for row in summary_rows:
        value = row.get("mean_vpt")
        if value == "" or value is None:
            continue
        i = system_index[str(row["system"])]
        j = int(row["observed_index"]) - 1
        matrix[i, j] = float(value)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    finite = matrix[np.isfinite(matrix)]
    vmax_plot = vmax
    if vmax_plot is None and finite.size:
        vmax_plot = float(np.max(finite))
    if vmax_plot is None:
        vmax_plot = 1.0

    block_columns = max(1, min(int(block_columns), len(systems)))
    rows_per_block = int(math.ceil(len(systems) / block_columns))
    fig_width = 2.95 * block_columns + 0.45
    fig_height = max(6.1, 0.16 * rows_per_block + 1.2)
    fig, axes = plt.subplots(
        1,
        block_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"wspace": 0.08},
    )
    axes = axes.ravel()

    image = None
    y_label_size = 6.0 if block_columns >= 4 else 6.7
    x_label_size = 8.0 if block_columns >= 4 else 8.5
    for block_idx, ax in enumerate(axes):
        start = block_idx * rows_per_block
        end = min(start + rows_per_block, len(systems))
        if start >= end:
            ax.axis("off")
            continue

        submatrix = np.ma.masked_invalid(matrix[start:end, :])
        image = ax.imshow(
            submatrix,
            aspect="equal",
            interpolation="nearest",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax_plot,
        )
        ax.set_xticks(np.arange(max_observed_dims))
        ax.set_xticklabels(
            [str(idx) for idx in range(1, max_observed_dims + 1)],
            rotation=0,
            ha="center",
            rotation_mode="default",
            fontsize=x_label_size,
        )
        ax.set_yticks(np.arange(end - start))
        ax.set_yticklabels(systems[start:end], fontsize=y_label_size, fontweight="bold")
        ax.set_xlabel("observe index", fontsize=8.5)
        ax.set_ylabel("")
        ax.set_title("")
        ax.minorticks_off()
        ax.tick_params(axis="both", which="both", top=False, right=False, length=2)
        for boundary in np.arange(-0.5, max_observed_dims, 1):
            ax.axvline(boundary, color="white", linewidth=0.45)

    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes), fraction=0.018, pad=0.015)
        colorbar.set_label("Mean VPT", fontsize=10)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.minorticks_off()

    fig.savefig(output_dir / save_svg, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _metadata_true_le(info: dict) -> list[float]:
    true_le = info.get("true_lyapunov_spectrum")
    if true_le is not None:
        return true_le
    estimated = info.get("lyapunov_spectrum_estimated")
    if estimated is not None:
        return estimated
    mle = _metadata_mle(info)
    return [float(mle)] if mle is not None else []


def _metadata_mle(info: dict) -> float | None:
    mle = info.get("maximum_lyapunov_estimated")
    if mle is not None:
        return float(mle)
    true_le = info.get("true_lyapunov_spectrum")
    if true_le:
        return float(max(true_le))
    estimated = info.get("lyapunov_spectrum_estimated")
    if estimated:
        return float(max(estimated))
    return None


def _named_dummy_system(system: str):
    return type(system, (), {})


def _new_dysts_output_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return DEFAULT_DYSTS_RESULTS_ROOT / timestamp


def _new_parameter_output_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return DEFAULT_PARAMETER_RESULTS_ROOT / timestamp


def _preferred_csv_path(output_dir: Path, preferred_name: str, legacy_name: str) -> Path:
    preferred = output_dir / preferred_name
    legacy = output_dir / legacy_name
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(read_csv(path))


def _finite_float(value):
    if value is None or value == "":
        return ""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return ""
    return scalar if math.isfinite(scalar) else ""


def _is_finite(value) -> bool:
    return _finite_float(value) != ""


if __name__ == "__main__":
    main()
