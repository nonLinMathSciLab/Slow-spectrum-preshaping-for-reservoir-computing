from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from crc_analyzer import grassmanian_distance
from dysts.flows import Lorenz
from grid_crc import CESNComprehensiveSweepV3, NumpyComplexEncoder
from plot_style import apply_publication_style
from simple_table import write_csv


PUBLIC_DIR = Path(__file__).resolve().parent
apply_publication_style()


FULL_EXPERIMENT_CONFIG = {
    "FIRST_STEP_DT": 0.001,
    "PTS_PER_PERIOD": 100,
    "delay": 0,
    "NUM_STEPS": 30000,
    "DataStartTime": 0,
    "WARMUP_STEPS": 1000,
    "TRAINING_TIME_STEPS": 2000,
    "PREDICTION_STEPS": 20000,
    "PRED_START_POINT": 0,
    "PRED_START_POINT_END": 5000,
    "OVER_NUM": 50,
    "FORCE_COMPLEX_MODE": False,
    "test_GSPT": True,
    "calc_only_ls": True,
    "test_conditional_lyapunov": True,
    "search_method": "grid",
    "FIX_DT": None,
}

SMOKE_EXPERIMENT_CONFIG = {
    **FULL_EXPERIMENT_CONFIG,
    "PTS_PER_PERIOD": 40,
    "NUM_STEPS": 700,
    "WARMUP_STEPS": 40,
    "TRAINING_TIME_STEPS": 80,
    "PREDICTION_STEPS": 60,
    "PRED_START_POINT_END": 20,
    "OVER_NUM": 3,
}

FULL_PARAMS = {
    "partial_observe": [[0, 0]],
    "w_res_method": "symmetry",
    "W_res_force_updates": False,
    "w_in_method": "eigenvec",
    "use_eigenvecs_method_or_list": "high",
    "conj_pair_num": "auto",
    "invariant_dim": 3,
    "construct_option": "random",
    "bias_method": "const",
    "seed": 0,
    "bias_amplitude": 0.01,
    "alpha": 1.0,
    "dim_reservoir": 300,
    "spectral_radius": 0.3,
    "regularization": 0.0,
    "er_prob": 1.0,
    "input_scaling_ratio": 1.0,
    "input_sparsity": 1.0,
    "is_intercept": False,
    "how_loop": "output=input",
    "how_train": "lstsq",
}

SMOKE_PARAMS = {
    **FULL_PARAMS,
    "dim_reservoir": 30,
    "regularization": 1e-8,
}


FIGURE_VARIANTS = {
    "original_eigenvec": {
        "label": "original_eigenvec",
        "description": "Original Fig. 2-4 setting: eigenvector input and no reservoir eigenvalue forcing.",
        "overrides": {},
    },
    "forced_eigenvec": {
        "label": "forced_eigenvec",
        "description": "Eigenvector input with the leading eigenvalue pair forced to exp(i*pi/32).",
        "overrides": {
            "W_res_force_updates": [((0, 1), np.cos(np.pi / 32) + 1j * np.sin(np.pi / 32))],
            "w_in_method": "eigenvec",
            "use_eigenvecs_method_or_list": "high",
        },
    },
    "random_input": {
        "label": "random_input",
        "description": "Random input matrix with no reservoir eigenvalue forcing.",
        "overrides": {
            "W_res_force_updates": False,
            "w_in_method": "random",
            "use_eigenvecs_method_or_list": None,
            "conj_pair_num": None,
            "invariant_dim": 3,
            "construct_option": "random",
        },
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the public Fig. 2-4 reservoir-computing figures.")
    parser.add_argument("--paper", action="store_true", help="Use the paper-scale seed20/random/spectral-radius-0.6 setup.")
    parser.add_argument("--full", action="store_true", help="Use the original paper-scale parameters.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated figures and arrays.")
    parser.add_argument("--seeds", type=int, default=None, help="Override the number of random seeds.")
    parser.add_argument("--steps-ahead", type=int, default=None, help="Prediction length used for overlay plots.")
    parser.add_argument("--spectral-radius", type=float, default=None, help="Override spectral_radius in all variants.")
    parser.add_argument("--construct-option", choices=["random", "just_add"], default=None, help="Override construct_option in all variants.")
    parser.add_argument(
        "--variant",
        choices=["original_eigenvec", "forced_eigenvec", "random_input", "all-requested"],
        default="original_eigenvec",
        help="Parameter set to run. all-requested runs the original setting plus the two requested variants.",
    )
    args = parser.parse_args(argv)

    if args.paper:
        args.full = True
        if args.seeds is None:
            args.seeds = 20
        if args.steps_ahead is None:
            args.steps_ahead = 2000
        if args.spectral_radius is None:
            args.spectral_radius = 0.6
        if args.construct_option is None:
            args.construct_option = "random"
        if args.variant == "original_eigenvec":
            args.variant = "all-requested"

    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    default_output = (
        PUBLIC_DIR / "results" / "fig2_3_4_seed20_random_sr0p6"
        if args.paper
        else PUBLIC_DIR / "results" / "fig2_3_4" / timestamp
    )
    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_count = args.seeds if args.seeds is not None else (40 if args.full else 2)
    steps_ahead = args.steps_ahead if args.steps_ahead is not None else (2000 if args.full else 60)
    variant_names = (
        ["original_eigenvec", "forced_eigenvec", "random_input"]
        if args.variant == "all-requested"
        else [args.variant]
    )
    if len(variant_names) == 1:
        output_paths = [
            run_figure_variant(
                variant_name=variant_names[0],
                output_dir=output_dir,
                full=args.full,
                seed_count=seed_count,
                steps_ahead=steps_ahead,
                spectral_radius=args.spectral_radius,
                construct_option=args.construct_option,
            )
        ]
    else:
        output_paths = []
        for variant_name in variant_names:
            output_paths.append(
                run_figure_variant(
                    variant_name=variant_name,
                    output_dir=output_dir / variant_name,
                    full=args.full,
                    seed_count=seed_count,
                    steps_ahead=steps_ahead,
                    spectral_radius=args.spectral_radius,
                    construct_option=args.construct_option,
                )
            )

    print("Generated Fig. 2-4 outputs:")
    for path in output_paths:
        print(f"  {path}")
    return output_paths[0] if len(output_paths) == 1 else output_paths


def make_experiment_config(full: bool) -> dict:
    experiment_config = dict(FULL_EXPERIMENT_CONFIG if full else SMOKE_EXPERIMENT_CONFIG)
    embedding_dimension = _metadata_for("Lorenz")["embedding_dimension"]
    experiment_config["embedding_dimension"] = embedding_dimension
    experiment_config["USE_DATA_LEN"] = (
        experiment_config["delay"] * embedding_dimension
        + experiment_config["WARMUP_STEPS"]
        + experiment_config["TRAINING_TIME_STEPS"]
        + experiment_config["PRED_START_POINT_END"]
        + experiment_config["PREDICTION_STEPS"]
    )
    experiment_config["system_name"] = "Lorenz"
    experiment_config["sanity_check"] = not full
    return experiment_config


def make_plot_config() -> dict:
    return {
        "mode_plot_stability": False,
        "mode_plot_projection": False,
        "mode_plot_distance": False,
        "mode_plot_eigenvalue_distribution": True,
        "mode_plot_fxp_eigenvalue_distribution": False,
        "plot_results": True,
    }


def make_params(variant_name: str, full: bool) -> dict:
    if variant_name not in FIGURE_VARIANTS:
        raise ValueError(f"Unknown Fig. 2-4 variant: {variant_name}")
    params = dict(FULL_PARAMS if full else SMOKE_PARAMS)
    params.update(FIGURE_VARIANTS[variant_name]["overrides"])
    if params["w_in_method"] == "random":
        params["use_eigenvecs_method_or_list"] = None
        params["conj_pair_num"] = None
    return params


def run_figure_variant(
    variant_name: str,
    output_dir: Path,
    full: bool,
    seed_count: int,
    steps_ahead: int,
    spectral_radius: float | None = None,
    construct_option: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = make_experiment_config(full)
    plot_config = make_plot_config()
    params_template = make_params(variant_name, full)
    if spectral_radius is not None:
        params_template["spectral_radius"] = spectral_radius
    if construct_option is not None:
        params_template["construct_option"] = construct_option

    metadata = {
        "variant": variant_name,
        "description": FIGURE_VARIANTS[variant_name]["description"],
        "full_scale": full,
        "seed_count": seed_count,
        "steps_ahead": steps_ahead,
        "spectral_radius": params_template["spectral_radius"],
        "construct_option": params_template["construct_option"],
        "params_template": params_template,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, cls=NumpyComplexEncoder)

    analyzer = CESNComprehensiveSweepV3(
        dynamical_system=Lorenz,
        destination_dir=output_dir,
        experiment_config=experiment_config,
        plot_config=plot_config,
        parameter_ranges=None,
        vander_degree=0,
    )
    analyzer._setup_base_data()

    results = []
    predictions = []
    lyapunov_spectra = []
    lyapunov_spectra_base = []
    conditional_spectra = []
    conditional_spectra_base = []

    for seed in range(seed_count):
        params = dict(params_template)
        params["seed"] = seed
        result = analyzer.run_single_experiment(params)
        if not result.get("success"):
            raise RuntimeError(result.get("error_traceback", result.get("error", "Unknown experiment failure")))
        prediction = analyzer.cesn.feedback_predict_start(
            analyzer.warmuped_states[0, :],
            experiment_config["WARMUP_STEPS"],
            pred_start_time=0,
            test_data=analyzer.test_data,
            no_delay_indices=analyzer.no_delay_indices,
            steps_ahead=steps_ahead,
        )
        results.append(result)
        predictions.append(prediction)
        lyapunov_spectra.append(result["lyapunov_spectrum"])
        lyapunov_spectra_base.append(result["lyapunov_spectrum_s"])
        conditional_spectra.append(result["conditional_lyapunov_spectrum"])
        conditional_spectra_base.append(result["conditional_lyapunov_spectrum_s"])

    _save_results(output_dir, results)
    np.save(output_dir / "prediction_list.npy", np.asarray(predictions, dtype=object))
    np.save(output_dir / "lyapunov_spectrum_list.npy", np.asarray(lyapunov_spectra, dtype=object))
    np.save(output_dir / "lyapunov_spectrum_s_list.npy", np.asarray(lyapunov_spectra_base, dtype=object))
    np.save(output_dir / "conditional_lyapunov_spectrum_list.npy", np.asarray(conditional_spectra, dtype=object))
    np.save(output_dir / "conditional_lyapunov_spectrum_s_list.npy", np.asarray(conditional_spectra_base, dtype=object))

    plot_prediction_overlay(
        params_template["partial_observe"],
        predictions,
        analyzer.test_data,
        experiment_config["WARMUP_STEPS"],
        pred_start_time=0,
        pred_steps=steps_ahead,
        threshold=0.5,
        lyapunov_time=analyzer.lyapunov_time,
        save_dir=output_dir,
    )
    plot_reservoir_geometries(analyzer, params_template, output_dir)
    plot_reservoir_geometries_plotly(analyzer, output_dir)
    plot_lyapunov_spectrum(lyapunov_spectra, analyzer.true_le, output_dir / "lyapunov_spectrum.svg")
    plot_lyapunov_spectrum(conditional_spectra, analyzer.true_le, output_dir / "conditional_lyapunov_spectrum.svg")

    return output_dir


def plot_prediction_overlay(
    partial_observe,
    pred_data_list,
    test_data,
    warmup_step,
    pred_start_time,
    pred_steps,
    threshold,
    lyapunov_time,
    save_dir,
    title_prefix="CESN prediction",
):
    true_solution = np.asarray(test_data[pred_start_time : pred_start_time + pred_steps]).real
    pred_steps = min(pred_steps, len(true_solution))
    true_solution = true_solution[:pred_steps]
    variance = np.var(test_data, axis=0)
    variance = np.where(variance < 1e-12, 1.0, variance)

    horizons = []
    pred_start_idx = warmup_step + pred_start_time
    for pred_data in pred_data_list:
        pred_slice = np.asarray(pred_data[pred_start_idx : pred_start_idx + pred_steps]).real
        nrmse = np.sqrt(np.average(((pred_slice - true_solution) ** 2) / variance, axis=1))
        exceeded = np.where(nrmse > threshold)[0]
        horizons.append(int(exceeded[0]) if len(exceeded) else pred_steps - 1)

    horizon_steps = np.asarray(horizons, dtype=float)
    vpt_min = float(np.min(horizon_steps) / lyapunov_time)
    vpt_mean = float(np.mean(horizon_steps) / lyapunov_time)
    vpt_max = float(np.max(horizon_steps) / lyapunov_time)
    best_idx = int(np.argmax(horizons))
    worst_idx = int(np.argmin(horizons))
    best_pred = pred_data_list[best_idx][pred_start_idx : pred_start_idx + pred_steps].real
    worst_pred = pred_data_list[worst_idx][pred_start_idx : pred_start_idx + pred_steps].real

    ndim = true_solution.shape[1]
    labels = ["X", "Y", "Z"][:ndim] if ndim <= 3 else [f"Dim {idx + 1}" for idx in range(ndim)]
    if all(x == partial_observe[0] for x in partial_observe):
        labels = [f"Observed variable {partial_observe[0][0]}"]
        ndim = 1

    time_axis = np.arange(pred_steps) / lyapunov_time
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.8 * ndim + 1), sharex=True, squeeze=False)
    axes = axes.ravel()

    for dim in range(ndim):
        axes[dim].plot(time_axis, true_solution[:, dim], color="black", linewidth=2, label="Ground truth")
        axes[dim].plot(
            time_axis,
            best_pred[:, dim],
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
            label="Best prediction",
        )
        axes[dim].plot(
            time_axis[: max(horizons[worst_idx], 1)],
            worst_pred[: max(horizons[worst_idx], 1), dim],
            color="tab:orange",
            linestyle="-.",
            linewidth=1.8,
            label="Worst prediction",
        )
        axes[dim].axvline(
            vpt_min,
            color="tab:blue",
            linestyle=":",
            linewidth=2,
            label=f"Min VPT = {vpt_min:.2f} LT" if dim == 0 else None,
        )
        axes[dim].axvline(
            vpt_mean,
            color="tab:green",
            linestyle="-.",
            linewidth=2,
            label=f"Mean VPT = {vpt_mean:.2f} LT" if dim == 0 else None,
        )
        axes[dim].axvline(
            vpt_max,
            color="tab:purple",
            linestyle="--",
            linewidth=2,
            label=f"Max VPT = {vpt_max:.2f} LT" if dim == 0 else None,
        )
        axes[dim].set_ylabel(labels[dim])
        axes[dim].set_ylim(-1.1, 1.1)
        axes[dim].grid(alpha=0.25)
        if dim == 0:
            axes[dim].legend(markerscale=1.0, ncol=2)

    axes[-1].set_xlabel("Lyapunov time")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "prediction_overlay.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reservoir_geometries(analyzer, params, save_dir: Path) -> None:
    states = np.asarray(analyzer.warmuped_states).real
    time = np.arange(states.shape[0])
    plot_3d(states[:, :3], time, save_dir / "reservoir_nodes_1_2_3.svg", "Reservoir nodes 1-3")

    if analyzer.cesn_component.P is not None and analyzer.cesn_component.P.shape[1] >= 3:
        q_input, _ = np.linalg.qr(analyzer.cesn_component.P[:, :3].real, mode="reduced")
        plot_3d(states @ q_input[:, :3], time, save_dir / "input_subspace_projection.svg", "Input subspace projection")

    essential = (1.0 - analyzer.cesn.alpha) * np.eye(analyzer.cesn.dim_reservoir) + analyzer.cesn.alpha * analyzer.cesn.W_close
    eigvals, eigvecs = np.linalg.eig(essential)
    basis, labels = leading_real_eigenbasis(eigvals, eigvecs, 3)
    q_close, _ = np.linalg.qr(basis.real, mode="reduced")
    projected = states @ q_close[:, :3]
    plot_3d(projected, time, save_dir / "closed_loop_subspace_projection.svg", "Closed-loop dominant subspace")

    if analyzer.cesn_component.P is not None and analyzer.cesn_component.P.shape[1] >= 3:
        g_dist = grassmanian_distance(q_input[:, :3], q_close[:, :3])
        (save_dir / "subspace_distance.txt").write_text(
            f"Grassmannian distance between input and closed-loop subspaces: {g_dist:.6f}\n"
            f"Closed-loop basis labels: {labels}\n",
            encoding="utf-8",
        )

    centered = states - states.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plot_3d(centered @ vh[:3].T, time, save_dir / "pca_projection.svg", "Reservoir PCA projection")


def plot_reservoir_geometries_plotly(analyzer, save_dir: Path, show: bool = False) -> None:
    """Write interactive 3D HTML plots and optionally display them inline."""

    try:
        import plotly.express as px
    except ImportError:
        px = None

    states = np.asarray(analyzer.warmuped_states).real
    time = np.arange(states.shape[0])
    _plotly_3d(states[:, :3], time, save_dir / "reservoir_nodes_1_2_3.html", "Reservoir nodes 1-3", px, show=show)

    if analyzer.cesn_component.P is not None and analyzer.cesn_component.P.shape[1] >= 3:
        q_input, _ = np.linalg.qr(analyzer.cesn_component.P[:, :3].real, mode="reduced")
        _plotly_3d(
            states @ q_input[:, :3],
            time,
            save_dir / "input_subspace_projection.html",
            "Input subspace projection",
            px,
            show=show,
        )

    essential = (1.0 - analyzer.cesn.alpha) * np.eye(analyzer.cesn.dim_reservoir) + analyzer.cesn.alpha * analyzer.cesn.W_close
    eigvals, eigvecs = np.linalg.eig(essential)
    basis, _ = leading_real_eigenbasis(eigvals, eigvecs, 3)
    q_close, _ = np.linalg.qr(basis.real, mode="reduced")
    _plotly_3d(
        states @ q_close[:, :3],
        time,
        save_dir / "closed_loop_subspace_projection.html",
        "Closed-loop dominant subspace",
        px,
        show=show,
    )

    centered = states - states.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    _plotly_3d(centered @ vh[:3].T, time, save_dir / "pca_projection.html", "Reservoir PCA projection", px, show=show)


def _plotly_3d(points: np.ndarray, color_values: np.ndarray, path: Path, title: str, px, show: bool = False) -> None:
    if px is None:
        _plotly_3d_fallback_html(points, color_values, path, title)
        return

    axis_labels = _plotly_axis_labels(path.stem)
    fig = px.scatter_3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        color=color_values,
        labels={"x": axis_labels[0], "y": axis_labels[1], "z": axis_labels[2], "color": "Time index"},
        title=title,
        color_continuous_scale="viridis",
    )
    fig.update_traces(
        mode="lines+markers",
        marker=dict(
            size=0.75,
            opacity=0.7,
            colorbar=dict(title=dict(text="Time index", font=dict(size=24)), tickfont=dict(size=20)),
        ),
        line=dict(width=0.5, color="gray"),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=32)),
        font=dict(size=24),
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title=dict(text=axis_labels[0], font=dict(size=28)), tickfont=dict(size=22)),
            yaxis=dict(title=dict(text=axis_labels[1], font=dict(size=28)), tickfont=dict(size=22)),
            zaxis=dict(title=dict(text=axis_labels[2], font=dict(size=28)), tickfont=dict(size=22)),
        ),
        hoverlabel=dict(font_size=24),
    )
    fig.write_html(path, config=_plotly_export_config(path.stem), include_mathjax="cdn")
    if show:
        fig.show()


def _plotly_3d_fallback_html(points: np.ndarray, color_values: np.ndarray, path: Path, title: str) -> None:
    payload = {
        "x": points[:, 0].tolist(),
        "y": points[:, 1].tolist(),
        "z": points[:, 2].tolist(),
        "color": color_values.tolist(),
        "title": title,
        "axis_labels": _plotly_axis_labels(path.stem),
    }
    config = _plotly_export_config(path.stem)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <title>{title}</title>
</head>
<body>
  <div id="plot" style="width: 100%; height: 92vh;"></div>
  <script>
    const payload = {json.dumps(payload)};
    const trace = {{
      x: payload.x,
      y: payload.y,
      z: payload.z,
      mode: "lines+markers",
      type: "scatter3d",
      marker: {{
        size: 0.75,
        opacity: 0.7,
        color: payload.color,
        colorscale: "Viridis",
        colorbar: {{
          title: {{text: "Time index", font: {{size: 24}}}},
          tickfont: {{size: 20}}
        }}
      }},
      line: {{width: 0.5, color: "gray"}}
    }};
    const layout = {{
      title: {{text: payload.title, font: {{size: 32}}}},
      font: {{size: 24}},
      scene: {{
        aspectmode: "cube",
        xaxis: {{title: {{text: payload.axis_labels[0], font: {{size: 28}}}}, tickfont: {{size: 22}}}},
        yaxis: {{title: {{text: payload.axis_labels[1], font: {{size: 28}}}}, tickfont: {{size: 22}}}},
        zaxis: {{title: {{text: payload.axis_labels[2], font: {{size: 28}}}}, tickfont: {{size: 22}}}}
      }},
      hoverlabel: {{font: {{size: 24}}}},
      margin: {{l: 0, r: 0, b: 0, t: 48}}
    }};
    const config = {json.dumps(config)};
    Plotly.newPlot("plot", [trace], layout, config);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _plotly_axis_labels(stem: str) -> tuple[str, str, str]:
    if stem == "input_subspace_projection":
        return (
            "W̃<sub>in</sub> basis 1",
            "W̃<sub>in</sub> basis 2",
            "W̃<sub>in</sub> basis 3",
        )
    return ("P1", "P2", "P3")


def _plotly_export_config(filename: str) -> dict:
    return {
        "responsive": True,
        "toImageButtonOptions": {
            "format": "svg",
            "filename": f"{filename}_view",
            "width": 1200,
            "height": 900,
            "scale": 1,
        },
    }


def leading_real_eigenbasis(eigvals: np.ndarray, eigvecs: np.ndarray, n_cols: int) -> tuple[np.ndarray, list[str]]:
    order = np.argsort(np.abs(eigvals))[::-1]
    columns = []
    labels = []
    used: set[int] = set()
    for idx in order:
        if idx in used or len(columns) >= n_cols:
            continue
        value = eigvals[idx]
        vector = eigvecs[:, idx]
        if np.isclose(value.imag, 0.0, atol=1e-10):
            columns.append(vector.real)
            labels.append(f"v{idx}")
            used.add(idx)
        else:
            columns.append(vector.real)
            labels.append(f"Re(v{idx})")
            if len(columns) < n_cols:
                columns.append(vector.imag)
                labels.append(f"Im(v{idx})")
            used.add(idx)
    return np.column_stack(columns[:n_cols]), labels[:n_cols]


def plot_3d(points: np.ndarray, color_values: np.ndarray, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_values, cmap="viridis", s=3, alpha=0.8)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="0.55", linewidth=0.35, alpha=0.6)
    ax.set_xlabel("P1")
    ax.set_ylabel("P2")
    ax.set_zlabel("P3")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="Time index")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lyapunov_spectrum(lyapunov_spectrum_list, true_le, save_path: Path, ylim_range=None) -> None:
    data = np.asarray([np.asarray(item, dtype=float) for item in lyapunov_spectrum_list])
    data = np.sort(data, axis=1)[:, ::-1]
    n_seeds = data.shape[0]
    n_exp = min(data.shape[1], 4)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    markers = ["o", "s", "^", "D"]
    seed_axis = np.arange(1, n_seeds + 1)

    fig, ax = plt.subplots(figsize=(12, 7.25))
    for idx in range(n_exp):
        ax.scatter(seed_axis, data[:, idx], color=colors[idx], marker=markers[idx], s=55, label=fr"$\hat\lambda_{idx + 1}$")
    for idx, value in enumerate(np.asarray(true_le, dtype=float)[:n_exp]):
        ax.axhline(value, color=colors[idx], linestyle="--", linewidth=1.5, label=f"True LE {idx + 1}: {value:.2f}")

    ax.set_yscale("symlog", linthresh=0.1)
    if ylim_range is not None:
        ax.set_ylim(ylim_range)
    ax.set_xlim(0.5, n_seeds + 0.5)
    seed_ticks = [1] + [tick for tick in range(10, n_seeds + 1, 10)]
    if n_seeds not in seed_ticks:
        seed_ticks.append(n_seeds)
    ax.set_xticks(seed_ticks)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Lyapunov exponent")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_results(output_dir: Path, results: list[dict]) -> None:
    with (output_dir / "single_experiment_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyComplexEncoder)
    rows = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if np.isscalar(value) or isinstance(value, (str, bool, int, float, type(None)))
        }
        rows.append(row)
    write_csv(output_dir / "single_experiment_results.csv", rows)


def _metadata_for(system_name: str) -> dict:
    metadata_path = PUBLIC_DIR / "dysts" / "data" / "chaotic_attractors.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)[system_name]


if __name__ == "__main__":
    main()
