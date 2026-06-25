from __future__ import annotations

import argparse
import ast
import datetime as dt
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from crc_plots import WhyBadCESNPlotter
from dysts.flows import Lorenz
from grid_crc import CESNComprehensiveSweepV3, NumpyComplexEncoder
from simple_table import Table, read_csv, sorted_unique, write_csv


PUBLIC_DIR = Path(__file__).resolve().parent
DEFAULT_FIG56_RESULTS_ROOT = PUBLIC_DIR / "results" / "fig_5_6"
DEFAULT_FIG56_TEMPLATE = PUBLIC_DIR / "configs" / "fig5_6_template_conditions.json"
FIG56_CSV = "fig_5_6_results.csv"
FIG56_FORCE_ORDER = [
    "exp_i_pi_16",
    "exp_i_pi_32",
    "exp_i_pi_64",
    "__0__1.01____1__0.9900990099009901__",
    "__0__1.05____1__0.9523809523809523__",
    "__0__1.1____1__0.9090909090909091__",
    "none",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create Lyapunov-spectrum summary plots for Fig. 5-6.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing a grid-search CSV.")
    parser.add_argument("--run", action="store_true", help="Run the Fig. 5-6 force-condition sweep before plotting.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --run. Defaults to a timestamped directory.")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from the output/results directory.")
    parser.add_argument("--force", action="store_true", help="Re-run rows that already exist in the Fig. 5-6 CSV.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per force/spectral-radius condition.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny Fig. 5-6 subset for code-path verification.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on new experiments.")
    parser.add_argument("--template-conditions", type=Path, default=DEFAULT_FIG56_TEMPLATE, help="experiment_conditions.json used as the Fig. 5-6 parameter template.")
    parser.add_argument("--w-in-method", choices=["eigenvec", "random"], default="eigenvec", help="Run one input-weight method per output directory.")
    parser.add_argument("--construct-option", choices=["random", "just_add"], default="random", help="Reservoir construction option for this run.")
    parser.add_argument(
        "--force-update-set",
        choices=["template", "template-pi64", "angles"],
        default=None,
        help="Which W_res_force_updates_list to use. Default uses the template list with pi/48 replaced by pi/64.",
    )
    parser.add_argument(
        "--force-angle-denominators",
        type=int,
        nargs="+",
        default=None,
        help="Use W_res forced eigenvalues exp(i*pi/N) for the given denominators.",
    )
    args = parser.parse_args(argv)
    force_update_set = args.force_update_set
    if force_update_set is None:
        force_update_set = "angles" if args.force_angle_denominators is not None else "template-pi64"
    force_angle_denominators = args.force_angle_denominators or [16, 32, 64]

    if args.run or args.output_dir is not None:
        results_dir = args.output_dir.resolve() if args.output_dir is not None else _new_fig56_output_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / FIG56_CSV
        if args.run and not args.plot_only:
            run_fig56_experiment(
                output_dir=results_dir,
                csv_path=csv_path,
                seeds=args.seeds,
                smoke=args.smoke,
                force=args.force,
                max_runs=args.max_runs,
                template_conditions_path=args.template_conditions,
                w_in_method=args.w_in_method,
                construct_option=args.construct_option,
                force_update_set=force_update_set,
                force_angle_denominators=force_angle_denominators,
            )
    else:
        if args.results_dir is None:
            parser.error("--results-dir is required unless --run or --output-dir is used.")
        results_dir = args.results_dir.resolve()
        csv_path = find_csv(results_dir)

    plot_fig56_results(results_dir, csv_path)
    print(f"Generated Fig. 5-6 outputs from {csv_path}")
    return results_dir


def plot_fig56_results(results_dir: Path, csv_path: Path) -> None:
    df = clean_grid_dataframe(read_csv(csv_path))
    true_le = load_true_lyapunov(results_dir)

    for column in ["lyapunov_spectrum", "conditional_lyapunov_spectrum"]:
        if column in df.columns:
            for row in df.rows:
                row[column] = parse_array(row[column])

    plotter = WhyBadCESNPlotter(results_dir=results_dir)
    fixed_conditions = [
        {"w_in_method": "eigenvec_high"},
        {"w_in_method": "random"},
    ]
    if "w_res_method" in df.columns:
        fixed_conditions = [dict(cond, w_res_method="symmetry") for cond in fixed_conditions]

    for condition in fixed_conditions:
        df_fixed = df.copy()
        for param, value in condition.items():
            if param in df_fixed.columns:
                df_fixed = df_fixed.filter_equal(param, value)
        if df_fixed.empty:
            continue

        plotter.param3_metric_plot(
            df_fixed,
            "spectral_radius",
            "W_res_force_updates",
            "seed",
            "lyapunov_spectrum",
            true_le,
            save_filename=f"lyapunov_spectrum_{_condition_name(condition)}.svg",
            symlog_linthresh=0.1,
            ylim_range=(-100, 10.0),
            max_metric_components=4,
            metric_label_prefix="LE",
        )
        if "conditional_lyapunov_spectrum" in df_fixed.columns:
            plotter.param3_metric_plot(
                df_fixed,
                "spectral_radius",
                "W_res_force_updates",
                "seed",
                "conditional_lyapunov_spectrum",
                true_le,
                save_filename=f"conditional_lyapunov_spectrum_{_condition_name(condition)}.svg",
                symlog_linthresh=0.1,
                ylim_range=(-100, 10.0),
                max_metric_components=4,
                metric_label_prefix="CLE",
            )

    plot_fig56_le2_heatmaps(df, results_dir)
    summary = summarize_second_lyapunov(df)
    summary_path = results_dir / "le2_summary.csv"
    summary.to_csv(summary_path)
    print_table(summary)


def run_fig56_experiment(
    output_dir: Path,
    csv_path: Path,
    seeds: int,
    smoke: bool,
    force: bool,
    max_runs: int | None,
    template_conditions_path: Path,
    w_in_method: str,
    construct_option: str,
    force_update_set: str,
    force_angle_denominators: list[int],
) -> None:
    template_conditions = load_fig56_template_conditions(template_conditions_path)
    parameter_ranges = make_fig56_parameter_ranges(
        template_conditions=template_conditions,
        seeds=seeds,
        smoke=smoke,
        w_in_method=w_in_method,
        construct_option=construct_option,
        force_update_set=force_update_set,
        force_angle_denominators=force_angle_denominators,
    )
    plot_config = dict(template_conditions.get("plot_config", make_fig56_plot_config()))
    analyzer = CESNComprehensiveSweepV3(
        dynamical_system=Lorenz,
        destination_dir=output_dir,
        experiment_config=make_fig56_experiment_config(template_conditions=template_conditions, smoke=smoke),
        plot_config=plot_config,
        parameter_ranges=parameter_ranges,
        vander_degree=0,
    )
    analyzer._setup_base_data()
    write_fig56_conditions(
        output_dir,
        analyzer,
        plot_config,
        parameter_ranges,
        smoke,
        template_conditions_path,
        w_in_method,
        construct_option,
        force_update_set,
        force_angle_denominators,
        template_conditions,
    )

    rows = [] if force else load_existing_rows(csv_path)
    completed = {fig56_row_key(row) for row in rows if row.get("success") is True}
    new_runs = 0
    for params in iter_fig56_params(parameter_ranges):
        force_label = force_update_label(params["W_res_force_updates"])
        key = fig56_params_key(params, force_label)
        if key in completed:
            continue
        result = analyzer.run_single_experiment(params)
        row = result_to_fig56_row(result, params, force_label)
        rows.append(row)
        write_csv(csv_path, rows)
        status = "ok" if row["success"] else "failed"
        print(
            f"{status}: W={row['W_res_force_updates']} sr={row['spectral_radius']} "
            f"seed={row['seed']} vpt={row['prediction_horizon_average']} "
            f"le={row['lyapunov_spectrum']}"
        )
        new_runs += 1
        if max_runs is not None and new_runs >= max_runs:
            break


def load_fig56_template_conditions(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_fig56_experiment_config(template_conditions: dict, smoke: bool = False) -> dict:
    config = dict(template_conditions.get("experiment_config", {}))
    if smoke:
        config["WARMUP_STEPS"] = 200
        config["TRAINING_TIME_STEPS"] = 200
        config["PREDICTION_STEPS"] = 200
        config["PRED_START_POINT_END"] = 200
        config["OVER_NUM"] = 5
    config["test_GSPT"] = True
    config["calc_only_ls"] = True
    config["test_conditional_lyapunov"] = True
    config["gpu_mode"] = cupy_available()
    config["USE_DATA_LEN"] = (
        config["delay"] * config["embedding_dimension"]
        + config["WARMUP_STEPS"]
        + config["TRAINING_TIME_STEPS"]
        + config["PRED_START_POINT_END"]
        + config["PREDICTION_STEPS"]
    )
    return config


def make_fig56_plot_config() -> dict:
    return {
        "mode_plot_stability": False,
        "mode_plot_projection": False,
        "mode_plot_distance": False,
        "mode_plot_eigenvalue_distribution": False,
        "mode_plot_fxp_eigenvalue_distribution": False,
        "plot_results": False,
    }


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


def make_fig56_parameter_ranges(
    template_conditions: dict,
    seeds: int,
    smoke: bool = False,
    w_in_method: str = "eigenvec",
    construct_option: str = "random",
    force_update_set: str = "template-pi64",
    force_angle_denominators: list[int] | None = None,
) -> dict:
    ranges = convert_template_parameter_ranges(template_conditions["parameter_ranges"])
    if force_angle_denominators is None:
        force_angle_denominators = [16, 32, 64]
    ranges["construct_option_list"] = [construct_option]
    ranges["W_res_force_updates_list"] = make_fig56_force_updates(
        ranges["W_res_force_updates_list"],
        force_update_set=force_update_set,
        force_angle_denominators=force_angle_denominators,
    )
    ranges["w_in_method_list"] = [w_in_method]
    ranges["seed_list"] = list(range(seeds))
    if smoke:
        ranges["seed_list"] = list(range(seeds))
        ranges["spectral_radius_list"] = [0.3]
        ranges["W_res_force_updates_list"] = [force_update_for_denominator(force_angle_denominators[0])]
    return ranges


def convert_template_parameter_ranges(parameter_ranges: dict) -> dict:
    return {key: convert_template_value(value) for key, value in parameter_ranges.items()}


def convert_template_value(value):
    if isinstance(value, dict) and set(value.keys()) == {"real", "imag"}:
        return complex(value["real"], value["imag"])
    if isinstance(value, list):
        return [convert_template_value(item) for item in value]
    return value


def force_update_for_denominator(denominator: int) -> list:
    return [((0, 1), np.cos(np.pi / denominator) + 1j * np.sin(np.pi / denominator))]


def make_fig56_force_updates(template_force_updates: list, force_update_set: str, force_angle_denominators: list[int]) -> list:
    if force_update_set == "angles":
        return [force_update_for_denominator(denom) for denom in force_angle_denominators]
    if force_update_set == "template":
        return template_force_updates
    if force_update_set == "template-pi64":
        return [replace_force_update_denominator(update, old_denominator=48, new_denominator=64) for update in template_force_updates]
    raise ValueError(f"Unknown force_update_set: {force_update_set}")


def replace_force_update_denominator(update, old_denominator: int, new_denominator: int):
    if update is False or update is None or len(update) != 1:
        return update
    value = update[0][1]
    angle = float(np.angle(value))
    denominator = int(round(np.pi / angle)) if abs(angle) > 1e-12 else 0
    if denominator != old_denominator:
        return update
    return force_update_for_denominator(new_denominator)


def iter_fig56_params(parameter_ranges: dict):
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
        params = {
            "partial_observe": record["partial_observe_list"],
            "w_res_method": record["w_res_method_list"],
            "er_prob": record["er_prob_list"],
            "W_res_force_updates": record["W_res_force_updates_list"],
            "w_in_method": record["w_in_method_list"],
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
        if params["w_in_method"] == "random":
            params["use_eigenvecs_method_or_list"] = None
            params["conj_pair_num"] = None
        yield params


def result_to_fig56_row(result: dict, params: dict, force_label: str) -> dict:
    success = bool(result.get("success"))
    return {
        "partial_observe": str(params["partial_observe"]),
        "w_res_method": params["w_res_method"],
        "er_prob": params["er_prob"],
        "W_res_force_updates": force_label,
        "W_res_force_updates_raw": str(params["W_res_force_updates"]),
        "w_in_method": fig56_w_in_label(params),
        "result_w_in_method": result.get("w_in_method", ""),
        "use_eigenvecs_method_or_list": params["use_eigenvecs_method_or_list"],
        "conj_pair_num": params["conj_pair_num"],
        "invariant_dim": params["invariant_dim"],
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
        "str_used_indices": result.get("str_used_indices", ""),
        "prediction_horizon_average": _finite_float(result.get("prediction_horizon_average")),
        "min_prediction_horizon": _finite_float(result.get("min_prediction_horizon")),
        "lyapunov_spectrum": array_to_text(result.get("lyapunov_spectrum")),
        "lyapunov_spectrum_s": array_to_text(result.get("lyapunov_spectrum_s")),
        "lyapunov_error": _finite_float(result.get("lyapunov_error")),
        "lyapunov_s_error": _finite_float(result.get("lyapunov_s_error")),
        "conditional_lyapunov_spectrum": array_to_text(result.get("conditional_lyapunov_spectrum")),
        "conditional_lyapunov_spectrum_s": array_to_text(result.get("conditional_lyapunov_spectrum_s")),
        "success": success,
        "error": "" if success else str(result.get("error", "")),
        "timestamp": result.get("timestamp", dt.datetime.now().isoformat()),
    }


def plot_fig56_le2_heatmaps(df: Table, results_dir: Path) -> None:
    if "lyapunov_spectrum" not in df.columns or "W_res_force_updates" not in df.columns:
        return
    rows = [row for row in df if row.get("success") is True]
    if not rows:
        return
    force_values = ordered_force_values([row.get("W_res_force_updates") for row in rows])
    spectral_values = sorted_unique([row.get("spectral_radius") for row in rows])
    seed_values = sorted_unique([row.get("seed") for row in rows])
    seed_axis_values = [int(seed) + 1 for seed in seed_values]
    output_dir = results_dir / "fig56_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    count_lines = ["force,spectral_radius,count_le2_in_-0.1_0.1,total"]
    for force_value in force_values:
        matrix = np.full((len(spectral_values), len(seed_values)), np.nan)
        for row in rows:
            if row.get("W_res_force_updates") != force_value:
                continue
            values = np.asarray(row["lyapunov_spectrum"], dtype=float).ravel()
            if len(values) < 2:
                continue
            r = spectral_values.index(row.get("spectral_radius"))
            c = seed_values.index(row.get("seed"))
            matrix[r, c] = values[1]

        for row_idx, spectral_radius in enumerate(spectral_values):
            finite = matrix[row_idx, np.isfinite(matrix[row_idx])]
            count = int(np.sum((-0.1 <= finite) & (finite <= 0.1)))
            count_lines.append(f"{force_value},{spectral_radius},{count},{len(finite)}")

        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad(color="white")
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
        ax.minorticks_off()
        ax.set_xticks(np.arange(len(seed_axis_values)))
        ax.set_xticklabels([str(seed) for seed in seed_axis_values], rotation=0)
        ax.set_yticks(np.arange(len(spectral_values)))
        ax.set_yticklabels([str(value) for value in spectral_values])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Spectral radius")
        ax.set_title(f"RC LE 2: {force_value}")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
        colorbar.set_label("RC LE 2")
        colorbar.ax.minorticks_off()
        fig.tight_layout()
        safe_force = _safe_name(force_value)
        fig.savefig(output_dir / f"fig56_le2_heatmap_{safe_force}.svg", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"fig56_le2_heatmap_{safe_force}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    (output_dir / "fig56_le2_counts.txt").write_text("\n".join(count_lines) + "\n", encoding="utf-8")


def write_fig56_conditions(
    output_dir: Path,
    analyzer: CESNComprehensiveSweepV3,
    plot_config: dict,
    parameter_ranges: dict,
    smoke: bool,
    template_conditions_path: Path,
    w_in_method: str,
    construct_option: str,
    force_update_set: str,
    force_angle_denominators: list[int],
    template_conditions: dict,
) -> None:
    with (output_dir / "experiment_conditions.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": dt.datetime.now().isoformat(),
                "figure": "fig_5_6",
                "smoke": smoke,
                "template_conditions_path": str(Path(template_conditions_path).resolve()),
                "w_in_method_run": w_in_method,
                "construct_option_run": construct_option,
                "force_update_set": force_update_set,
                "force_angle_denominators": force_angle_denominators,
                "experiment_config": analyzer.experiment_config,
                "plot_config": plot_config,
                "parameter_ranges": parameter_ranges,
                "notes": [
                    "Parameter ranges are loaded from the template experiment_conditions.json.",
                    "construct_option_list is overridden by --construct-option.",
                    "W_res_force_updates_list is selected by --force-update-set.",
                    "w_in_method_list is overridden so eigenvec and random runs can be stored separately.",
                    "ESN train/predict calculations are CPU NumPy; Lyapunov reduced-QR calculations use CuPy when gpu_mode=True.",
                    "RC LE and CLE plots show four estimated exponents; True LE reference lines show the system spectrum.",
                ],
            },
            f,
            indent=2,
            cls=NumpyComplexEncoder,
        )


def load_true_lyapunov(results_dir: Path) -> np.ndarray:
    conditions_path = results_dir / "experiment_conditions.json"
    system_name = "Lorenz"
    if conditions_path.exists():
        with conditions_path.open("r", encoding="utf-8") as f:
            conditions = json.load(f)
        system_name = conditions.get("experiment_config", {}).get("system_name", system_name)

    metadata_path = PUBLIC_DIR / "dysts" / "data" / "chaotic_attractors.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f).get(system_name, {})
    true_le = metadata.get("true_lyapunov_spectrum")
    if true_le is None:
        true_le = [metadata.get("maximum_lyapunov_estimated", 0.0)]
    return np.asarray(true_le, dtype=float)


def parse_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if not isinstance(value, str):
        return np.asarray([value], dtype=float)

    text = value.strip()
    if not text:
        return np.asarray([], dtype=float)
    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=float)
    except (ValueError, SyntaxError):
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return np.fromstring(text.replace(",", " "), sep=" ")


def summarize_second_lyapunov(df: Table) -> Table:
    if "lyapunov_spectrum" not in df.columns:
        return Table([])

    def is_second_in_range(values) -> bool:
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            return False
        return -0.1 <= values[1] <= 0.1

    working = df.copy()
    for row in working.rows:
        row["le2_in_range"] = is_second_in_range(row["lyapunov_spectrum"])
    settings_cols = [
        col
        for col in ["w_in_method", "w_res_method", "W_res_force_updates", "spectral_radius"]
        if col in working.columns
    ]
    if not settings_cols:
        return Table([{"count_in_range": sum(int(row["le2_in_range"]) for row in working)}])

    groups = {}
    for row in working:
        key = tuple(row[col] for col in settings_cols)
        groups[key] = groups.get(key, 0) + int(row["le2_in_range"])

    summary_rows = []
    for key, count in groups.items():
        summary_row = {col: value for col, value in zip(settings_cols, key)}
        summary_row["count_in_range"] = count
        summary_rows.append(summary_row)
    summary_rows.sort(key=fig56_summary_sort_key)
    return Table(summary_rows)


def ordered_force_values(values: list) -> list:
    return sorted(sorted_unique(values), key=fig56_force_sort_key)


def fig56_force_sort_key(value) -> tuple:
    try:
        force_order = FIG56_FORCE_ORDER.index(value)
    except ValueError:
        force_order = len(FIG56_FORCE_ORDER)
    return (force_order, str(value))


def fig56_summary_sort_key(row: dict) -> tuple:
    w_in = str(row.get("w_in_method", ""))
    w_res = str(row.get("w_res_method", ""))
    force = fig56_force_sort_key(row.get("W_res_force_updates"))
    try:
        spectral = float(row.get("spectral_radius"))
    except (TypeError, ValueError):
        spectral = float("inf")
    return (w_in, w_res, force, spectral)


def print_table(table: Table) -> None:
    if table.empty:
        print("(empty summary)")
        return
    columns = sorted(table.columns)
    print(",".join(columns))
    for row in table:
        print(",".join(str(row.get(column, "")) for column in columns))


def _condition_name(condition: dict) -> str:
    return "_".join(f"{key}_{value}" for key, value in condition.items())


def _new_fig56_output_dir() -> Path:
    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return DEFAULT_FIG56_RESULTS_ROOT / timestamp


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = list(read_csv(path))
    for column in ["lyapunov_spectrum", "conditional_lyapunov_spectrum"]:
        for row in rows:
            if column in row:
                row[column] = parse_array(row[column])
    return rows


def find_csv(results_dir: Path) -> Path:
    csv_path = results_dir / FIG56_CSV
    if csv_path.exists():
        return csv_path
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {results_dir}")
    return csv_files[0]


def clean_grid_dataframe(df: Table) -> Table:
    rows = []
    for row in df:
        clean = dict(row)
        if clean.get("regularization") == 1.0000000000000001e-24:
            clean["regularization"] = 1e-24
        if clean.get("W_res_force_updates") == "False":
            clean["W_res_force_updates"] = False
        if clean.get("W_res_force_updates") == "True":
            clean["W_res_force_updates"] = True
        rows.append(clean)
    return Table(rows)


def fig56_params_key(params: dict, force_label: str) -> tuple:
    return (
        fig56_w_in_label(params),
        force_label,
        float(params["spectral_radius"]),
        int(params["seed"]),
    )


def fig56_row_key(row: dict) -> tuple:
    return (
        row.get("w_in_method"),
        row.get("W_res_force_updates"),
        float(row.get("spectral_radius")),
        int(row.get("seed")),
    )


def array_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value == "":
        return ""
    return json.dumps(np.asarray(value, dtype=float).tolist())


def force_update_label(update) -> str:
    if update is False or update is None:
        return "none"
    if len(update) == 1:
        value = update[0][1]
        angle = float(np.angle(value))
        denom = int(round(np.pi / angle)) if abs(angle) > 1e-12 else 0
        return f"exp_i_pi_{denom}"
    return _safe_name(update)


def fig56_w_in_label(params: dict) -> str:
    if params["w_in_method"] == "eigenvec" and params["use_eigenvecs_method_or_list"]:
        return f"eigenvec_{params['use_eigenvecs_method_or_list']}"
    return params["w_in_method"]


def _finite_float(value):
    if value is None or value == "":
        return ""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return ""
    return scalar if np.isfinite(scalar) else ""


def _safe_name(value) -> str:
    text = str(value)
    allowed = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)[:120]


if __name__ == "__main__":
    main()
