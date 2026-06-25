from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_publication_style
from simple_table import Table, sorted_unique


apply_publication_style()


class WhyBadCESNPlotter:
    """Plotting utilities needed by the public figure scripts."""

    def __init__(self, results_dir=None) -> None:
        self.is_show = results_dir is None
        self.results_dir = Path(results_dir) if results_dir is not None else None

    def plot_eigenvalue_distribution(
        self,
        alpha,
        left_plot_eigenvals,
        right_plot_eigenvals,
        left_spectral_radius,
        right_spectral_radius,
        left_fix_point_norm,
        right_fix_point_norm,
        list_used_indices,
        inv_dim,
        spectral_radius,
        grassmanian_distance,
        filename,
        save_dir,
        is_legend=True,
    ) -> None:
        left_plot_eigenvals = np.asarray(left_plot_eigenvals)
        right_plot_eigenvals = np.asarray(right_plot_eigenvals)
        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        unit_circle = np.exp(1j * theta)
        boundary = alpha * spectral_radius * unit_circle + (1.0 - alpha)

        used_flat: list[int] = []
        if list_used_indices:
            for group in list_used_indices:
                used_flat.extend(group)

        axes[0].scatter(left_plot_eigenvals.real, left_plot_eigenvals.imag, s=24, alpha=0.7)
        if used_flat:
            axes[0].scatter(
                left_plot_eigenvals.real[used_flat],
                left_plot_eigenvals.imag[used_flat],
                color="red",
                marker="x",
                s=30,
                label="Eigenvalues used for input subspace",
            )

        axes[1].scatter(right_plot_eigenvals.real, right_plot_eigenvals.imag, s=24, alpha=0.7, color="tab:red")

        for ax in axes:
            ax.plot(unit_circle.real, unit_circle.imag, color="black", linestyle="--", linewidth=1, label="Unit circle")
            ax.plot(
                boundary.real,
                boundary.imag,
                color="tab:green",
                linestyle="--",
                linewidth=1,
                label=(
                    r"Eigenvalue bounds of $A_{\mathrm{eff}}$:"
                    + "\n"
                    + rf"Center $1-\alpha={1.0 - alpha:.2f}$, Radius $\alpha\rho(A)={alpha * spectral_radius:.2f}$"
                ),
            )
            ax.axhline(0, color="0.8", linewidth=0.8)
            ax.axvline(0, color="0.8", linewidth=0.8)
            ax.set_xlabel("Real")
            ax.set_ylabel("Imag")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
            if is_legend:
                ax.legend(markerscale=1.0)
        fig.tight_layout()

        if save_dir is None:
            plt.show()
        else:
            path = save_dir / f"{_safe_name(filename)}_eigenvalues.svg"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    def param3_metric_plot(
        self,
        df,
        sweep_param1,
        sweep_param2,
        sweep_param3,
        metric,
        metric_true,
        save_filename=None,
        symlog_linthresh=None,
        ylim_range=None,
        max_metric_components=None,
        metric_label_prefix="LE",
    ) -> None:
        table = _as_table(df)
        filtered, fixed = _filter_to_modes(
            table,
            exclude=[sweep_param1, sweep_param2, sweep_param3, metric],
            fixed_conditions=None,
        )
        if filtered.empty:
            filtered = table
            fixed = {}

        values1 = _ordered_unique(sweep_param1, filtered.values(sweep_param1))
        values2 = _ordered_unique(sweep_param2, filtered.values(sweep_param2))
        n_row = max(len(values1), 1)
        n_col = max(len(values2), 1)
        fig, axes = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3.8 * n_row), squeeze=False)
        metric_true = np.asarray(metric_true, dtype=float).ravel()
        requested_components = max_metric_components or max(len(metric_true), 1)
        colors = plt.cm.tab10(np.linspace(0, 1, max(requested_components, len(metric_true), 2)))
        panel_counts = []

        for row, value1 in enumerate(values1):
            for col, value2 in enumerate(values2):
                ax = axes[row, col]
                rows = [
                    rec
                    for rec in filtered
                    if rec.get(sweep_param1) == value1 and rec.get(sweep_param2) == value2
                ]
                count_in_range = 0
                for rec in rows:
                    values = np.asarray(rec[metric], dtype=float).ravel()
                    count_in_range += int(_second_le_in_range(values))
                    x_value = _seed_axis_value(rec[sweep_param3]) if sweep_param3 == "seed" else rec[sweep_param3]
                    for idx, value in enumerate(values[:requested_components]):
                        ax.scatter(x_value, value, color=colors[idx], s=32, label=f"{metric_label_prefix} {idx + 1}")
                panel_counts.append(
                    {
                        sweep_param1: value1,
                        sweep_param2: value2,
                        "count_in_range": count_in_range,
                        "total": len(rows),
                    }
                )
                for idx, true_value in enumerate(metric_true):
                    ax.axhline(true_value, color=colors[idx], linestyle="--", linewidth=1, label=f"True LE {idx + 1}")
                if symlog_linthresh is not None:
                    ax.set_yscale("symlog", linthresh=symlog_linthresh)
                if ylim_range is not None:
                    ax.set_ylim(ylim_range)
                ax.set_xlabel("Seed" if row == n_row - 1 and sweep_param3 == "seed" else (sweep_param3 if row == n_row - 1 else ""))
                ax.set_ylabel(metric if col == 0 else "")
                ax.tick_params(labelbottom=row == n_row - 1, labelleft=col == 0)
                if sweep_param3 == "seed":
                    seed_ticks = _seed_ticks(rows)
                    ax.set_xticks(seed_ticks)
                    ax.set_xlim(0.5, _max_seed_axis_value(rows) + 0.5)
                ax.grid(alpha=0.25)

        handles, labels = axes[0, -1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axes[0, -1].legend(unique.values(), unique.keys(), loc="best", markerscale=1.0)
        fig.tight_layout()
        output_filename = save_filename or f"{metric}_metric3.svg"
        self._write_param3_count_summary("param3_metric_plots", output_filename, metric, fixed, panel_counts)
        self._finish(fig, "param3_metric_plots", output_filename)

    def _finish(self, fig, subdir: str, filename: str) -> None:
        if self.is_show:
            plt.show()
            return
        assert self.results_dir is not None
        output_dir = self.results_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        if not filename.endswith((".svg", ".pdf", ".png")):
            filename = f"{filename}.svg"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_param3_count_summary(self, subdir: str, filename: str, metric: str, fixed: dict, rows: list[dict]) -> None:
        if self.is_show:
            return
        assert self.results_dir is not None
        output_dir = self.results_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        path = output_dir / f"{stem}_le2_counts.txt"
        lines = [
            f"metric: {metric}",
            "count: number of rows where the second LE is in [-0.1, 0.1]",
            f"fixed: {fixed}",
            "",
        ]
        if rows:
            columns = [key for key in rows[0].keys()]
            lines.append(",".join(columns))
            for row in rows:
                lines.append(",".join(str(row.get(column, "")) for column in columns))
        _write_text(path, "\n".join(lines) + "\n")


def _as_table(obj) -> Table:
    if isinstance(obj, Table):
        return obj
    if isinstance(obj, list):
        return Table(obj)
    raise TypeError("Expected simple_table.Table or list of row dictionaries")


def _write_text(path: Path, text: str) -> None:
    target = _windows_long_path(path)
    target.write_text(text, encoding="utf-8")


def _windows_long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return path
    return Path("\\\\?\\" + resolved)


def _filter_to_modes(table: Table, exclude: list[str], fixed_conditions=None) -> tuple[Table, dict]:
    candidate_params = [
        "partial_observe",
        "how_loop",
        "w_in_method",
        "w_res_method",
        "bias_amplitude",
        "alpha",
        "dim_reservoir",
        "spectral_radius",
        "regularization",
        "W_res_force_updates",
        "input_scaling_ratio",
    ]
    fixed = dict(fixed_conditions or {})
    for param in candidate_params:
        if param in table.columns and param not in exclude and param not in fixed:
            fixed[param] = table.mode(param)

    filtered = table
    for param, value in fixed.items():
        if param in filtered.columns:
            filtered = filtered.filter_equal(param, value)
    return filtered, fixed


def _ordered_unique(param: str, values: list) -> list:
    unique_values = sorted_unique(values)
    if param != "W_res_force_updates":
        return unique_values

    preferred = [
        "exp_i_pi_16",
        "exp_i_pi_32",
        "exp_i_pi_64",
        "__0__1.01____1__0.9900990099009901__",
        "__0__1.05____1__0.9523809523809523__",
        "__0__1.1____1__0.9090909090909091__",
        "none",
    ]
    order = {value: idx for idx, value in enumerate(preferred)}
    return sorted(unique_values, key=lambda value: (order.get(value, len(order)), str(value)))


def _second_le_in_range(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float).ravel()
    return len(values) > 1 and -0.1 <= values[1] <= 0.1


def _seed_axis_value(value) -> float:
    return float(value) + 1.0


def _seed_ticks(rows: list[dict]) -> list[int]:
    if not rows:
        return [1]
    max_seed = max(int(row["seed"]) for row in rows) + 1
    return [tick for tick in [1, 10, 20] if tick <= max_seed]


def _max_seed_axis_value(rows: list[dict]) -> int:
    if not rows:
        return 1
    return max(int(row["seed"]) for row in rows) + 1


def _safe_name(value) -> str:
    text = str(value)
    allowed = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)[:120]
