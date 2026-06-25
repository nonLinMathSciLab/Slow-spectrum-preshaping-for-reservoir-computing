from __future__ import annotations

import datetime as _datetime
import json
import traceback
from pathlib import Path

import numpy as np

from crc import ComplexEchoStateNetwork, CreateCESN_Component
from crc_analyzer import (
    distance_from_subspace,
    eachpoint_tangent_normal_analysis,
    find_fixed_point_numba,
    grassmanian_distance,
    similarity_with_subspace,
)
from crc_plots import WhyBadCESNPlotter
from dysts.setup import setup_experiment


class NumpyComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (complex, np.complexfloating)):
            return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


class CESNComprehensiveSweepV3:
    """Compact single-experiment runner used by the public figure scripts."""

    def __init__(
        self,
        dynamical_system,
        destination_dir,
        experiment_config,
        plot_config,
        parameter_ranges=None,
        vander_degree: int = 0,
    ) -> None:
        self.dynamical_system = dynamical_system
        self.experiment_config = experiment_config
        self.plot_config = plot_config
        self.parameter_ranges = parameter_ranges
        self.vander_degree = vander_degree
        self.opt_alpha = None
        self.optdly_searched = {}
        self.results = []

        if destination_dir is None:
            timestamp = _datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.results_dir = Path("results") / "manual_crc" / timestamp
        else:
            self.results_dir = Path(destination_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.current_sweep_dir = self.results_dir

    def _setup_base_data(self) -> None:
        data_gen_params = {
            "pts_per_period": self.experiment_config["PTS_PER_PERIOD"],
            "num_steps": self.experiment_config["NUM_STEPS"],
            "first_step_dt": self.experiment_config["FIRST_STEP_DT"],
        }
        if self.experiment_config.get("FIX_DT") is not None:
            data_gen_params["fix_dt"] = self.experiment_config["FIX_DT"]

        experiment_data = setup_experiment(self.dynamical_system, data_gen_params)
        self.system = experiment_data["system"]
        self.true_le = experiment_data["true_le"]
        self.dt = experiment_data["dt"]
        raw_data = experiment_data["ALLDATA"]

        start = self.experiment_config["DataStartTime"]
        stop = start + self.experiment_config["USE_DATA_LEN"]
        use_data = raw_data[start:stop]
        data_min = np.min(use_data, axis=0)
        data_max = np.max(use_data, axis=0)
        data_range = np.where(np.abs(data_max - data_min) < 1e-12, 1.0, data_max - data_min)
        self.normalized_data = (use_data - np.mean(use_data, axis=0)) * 2.0 / data_range

        self.lyapunov_time = 1.0 / (experiment_data["mle"] * self.dt)
        self.experiment_config["lyapunov_time"] = self.lyapunov_time
        self.experiment_config["system"] = self.system
        self.experiment_config["dt"] = self.dt
        self.dim_target_system = self.normalized_data.shape[1]

        with (self.results_dir / "experiment_conditions.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_config": self.experiment_config,
                    "plot_config": self.plot_config,
                },
                f,
                indent=2,
                cls=NumpyComplexEncoder,
            )

    def _apply_partial_observe_and_delay(self, partial_observe):
        delays = [entry[1] for entry in partial_observe]
        if any(delays):
            raise ValueError("The compact public code supports the no-delay figure setting.")

        complex_data = self.normalized_data.astype(complex)
        input_data = complex_data[:, [entry[0] for entry in partial_observe]]
        if self.vander_degree:
            vand = np.vander(complex_data[:, 0], self.vander_degree + 1, increasing=True)[:, 1:]
            vand_range = np.ptp(vand[:, 1:], axis=0)
            vand_range = np.where(np.abs(vand_range) < 1e-12, 1.0, vand_range)
            vand[:, 1:] = (vand[:, 1:] - np.mean(vand[:, 1:], axis=0)) * 2.0 / vand_range
            input_data = vand
        dim_inputs = input_data.shape[1]
        no_delay_indices = [idx for idx, (_, delay) in enumerate(partial_observe) if delay == 0]
        return input_data, complex_data, partial_observe, dim_inputs, 0, no_delay_indices

    def run_single_experiment(self, params: dict) -> dict:
        params = params.copy()
        try:
            (
                input_data,
                original_data,
                partial_observe,
                dim_inputs,
                max_delay,
                self.no_delay_indices,
            ) = self._apply_partial_observe_and_delay(params["partial_observe"])
            params["partial_observe"] = partial_observe

            if params["how_loop"] != "output=input":
                raise ValueError("The compact public code supports how_loop='output=input'.")
            if params["how_train"] != "lstsq":
                raise ValueError("The compact public code supports how_train='lstsq'.")
            dim_outputs = dim_inputs

            self.cesn_component = CreateCESN_Component(
                dim_inputs=dim_inputs,
                dim_reservoir=params["dim_reservoir"],
                spectral_radius=params["spectral_radius"],
                alpha=params["alpha"],
                seed=params["seed"],
                gpu_mode=False,
            )
            components = self.cesn_component.create_all_components(
                w_res_method=params["w_res_method"],
                w_in_method=params["w_in_method"],
                conj_pair_num=params["conj_pair_num"],
                invariant_dim=params["invariant_dim"],
                construct_option=params["construct_option"],
                bias_method=params["bias_method"],
                bias_amplitude=params["bias_amplitude"],
                input_scaling=params["input_scaling_ratio"] * np.sqrt(params["dim_reservoir"] / dim_inputs),
                use_eigenvecs_method_or_list=params["use_eigenvecs_method_or_list"],
                W_res_force_updates=params["W_res_force_updates"],
                er_prob=params["er_prob"],
                input_sparsity=params["input_sparsity"],
            )

            method_name = _method_name(params, self.cesn_component)
            param_name = _param_name(params)
            method_param_name = f"{method_name}_{param_name}"

            self.cesn = ComplexEchoStateNetwork(
                dim_inputs=dim_inputs,
                dim_reservoir=params["dim_reservoir"],
                dim_outputs=dim_outputs,
                is_realdata=np.allclose(input_data.imag, 0.0),
                W_in=self.cesn_component.w_in,
                W_res=self.cesn_component.w_res,
                bias=self.cesn_component.bias,
                alpha=params["alpha"],
                regularization=params["regularization"],
                is_intercept=params.get("is_intercept", False),
                partial_observe_list=params["partial_observe"],
                delay=self.experiment_config["delay"],
                max_delay=max_delay,
                force_complex_mode=self.experiment_config["FORCE_COMPLEX_MODE"],
                gpu_mode=False,
            )

            warmup_end = self.experiment_config["WARMUP_STEPS"]
            training_steps = self.experiment_config["TRAINING_TIME_STEPS"]
            training_end = warmup_end + training_steps
            listen_len = training_end + self.experiment_config["PRED_START_POINT_END"]

            if not self.cesn.complex_mode:
                input_data = input_data.real
                original_data = original_data.real

            training_data = input_data[warmup_end:training_end]
            self.test_data = input_data[warmup_end:, self.no_delay_indices]

            fixed_point, fixed_point_norm, _, _ = find_fixed_point_numba(
                self.cesn.W_res,
                self.cesn.bias,
                tol=1e-12,
            )
            _, self.warmuped_states = self.cesn.listen(
                U=input_data[:listen_len],
                warm_up=warmup_end,
                initial_state=fixed_point,
            )

            train_states = self.warmuped_states[: training_data.shape[0]]
            self.cesn.training_lstsq(states=train_states, train_data=training_data)

            (
                prediction_horizon_average,
                horizon_min,
                horizons,
                prediction_horizon_smape_average,
                horizon_smape_min,
                horizons_smape,
            ) = self.cesn.average_prediction_horizon_numba(
                test_data=self.test_data,
                max_delay=max_delay,
                no_delay_indices=self.no_delay_indices,
                warmuped=self.warmuped_states,
                training_step=training_steps,
                pred_start_point=self.experiment_config["PRED_START_POINT"],
                pred_start_point_end=self.experiment_config["PRED_START_POINT_END"],
                over_num=self.experiment_config["OVER_NUM"],
                steps_ahead=self.experiment_config["PREDICTION_STEPS"],
                lyapunov_time=self.lyapunov_time,
            )

            essential_close = (1.0 - self.cesn.alpha) * np.eye(self.cesn.dim_reservoir) + self.cesn.alpha * self.cesn.W_close
            close_eigvals, close_eigvecs = np.linalg.eig(essential_close)
            close_order = np.argsort(np.abs(close_eigvals))[::-1]
            close_eigvals_sorted = close_eigvals[close_order]
            close_eigvecs_sorted = close_eigvecs[:, close_order]

            close_fixed_point, close_fixed_point_norm, _, _ = find_fixed_point_numba(
                self.cesn.W_close,
                self.cesn.bias,
                tol=1e-12,
            )
            res_matrix = (1.0 - self.cesn.alpha) * np.eye(self.cesn.dim_reservoir) + self.cesn.alpha * np.diag(1.0 - fixed_point**2) @ self.cesn.W_res
            close_matrix = (1.0 - self.cesn.alpha) * np.eye(self.cesn.dim_reservoir) + self.cesn.alpha * np.diag(1.0 - close_fixed_point**2) @ self.cesn.W_close
            res_fxp_eigvals = np.linalg.eigvals(res_matrix)
            close_fxp_eigvals = np.linalg.eigvals(close_matrix)

            tangent_normal = {}
            if self.experiment_config.get("test_GSPT", True):
                tangent_normal = eachpoint_tangent_normal_analysis(
                    attractor_points=self.warmuped_states[:training_steps],
                    alpha=params["alpha"],
                    W_cl=self.cesn.W_close,
                    P=close_eigvecs_sorted,
                    dim_invariant=self.experiment_config["embedding_dimension"],
                    dt=self.dt,
                    calc_only_ls=self.experiment_config.get("calc_only_ls", True),
                    true_ls=self.true_le,
                    gpu_mode=self.experiment_config.get("gpu_mode", False),
                )

            conditional_le = {}
            if self.experiment_config.get("test_conditional_lyapunov", True):
                essential_res = (1.0 - self.cesn.alpha) * np.eye(self.cesn.dim_reservoir) + self.cesn.alpha * self.cesn.W_res
                res_eigvals, res_eigvecs = np.linalg.eig(essential_res)
                res_order = np.argsort(np.abs(res_eigvals))[::-1]
                conditional_le = eachpoint_tangent_normal_analysis(
                    attractor_points=self.warmuped_states[:training_steps],
                    alpha=params["alpha"],
                    W_cl=essential_res,
                    P=res_eigvecs[:, res_order],
                    dim_invariant=self.experiment_config["embedding_dimension"],
                    dt=self.dt,
                    calc_only_ls=True,
                    gpu_mode=self.experiment_config.get("gpu_mode", False),
                )

            grass_dist = grassmanian_distance(
                self.cesn_component.w_in,
                close_eigvecs[:, close_order[: self.cesn_component.dim_inputs]],
            )
            cossim = similarity_with_subspace(
                close_eigvecs_sorted[:, : self.cesn_component.dim_inputs],
                close_eigvecs_sorted,
            )
            cossim_base_fiber = float(np.average(cossim[self.cesn_component.dim_inputs :]))
            distance_from_base = distance_from_subspace(
                data_points=self.warmuped_states[:-1],
                basis=close_eigvecs_sorted,
                dim_principal=min(3, self.cesn.dim_reservoir),
            )

            singular_values = np.linalg.svd(train_states, compute_uv=False, full_matrices=False)
            normalized_singular_values = singular_values / max(singular_values[0], 1e-300)
            rank_threshold = singular_values[0] * max(train_states.shape) * np.finfo(singular_values.dtype).eps
            effective_rank = int(np.sum(singular_values > rank_threshold))

            result_params = params.copy()
            if result_params["w_in_method"] == "eigenvec":
                result_params["w_in_method"] = f"eigenvec_{self.cesn_component.str_used_indices}"
                result_params["str_used_indices"] = self.cesn_component.str_used_indices
            result = {
                **result_params,
                "dim_outputs": dim_outputs,
                "W_res_trace": float(np.real(components["trace"])),
                "W_res_determinant": float(np.real(components["determinant"])),
                "W_res_eigenvals": self.cesn_component.eigenvals_scaled,
                "W_res_Actual_spectral_radius": float(np.max(np.abs(self.cesn_component.eigenvals_scaled))),
                "W_res_singular_value": float(np.linalg.norm(self.cesn.W_res, ord=2)),
                "W_out_singular_value": float(np.linalg.norm(self.cesn.W_out, ord=2)),
                "W_close_eigenvals": close_eigvals_sorted,
                "W_close_spectral_radius": float(np.max(np.abs(close_eigvals))),
                "W_close_rank": int(np.linalg.matrix_rank(essential_close, 1e-10)),
                "W_close_singular_value": float(np.linalg.norm(essential_close, ord=2)),
                "fxp_W_res_eigvals": res_fxp_eigvals,
                "fxp_W_res_spectral_radius": float(np.max(np.abs(res_fxp_eigvals))),
                "fxp_W_res_norm": fixed_point_norm,
                "fxp_W_close_eigvals": close_fxp_eigvals,
                "fxp_W_close_spectral_radius": float(np.max(np.abs(close_fxp_eigvals))),
                "fxp_W_close_norm": close_fixed_point_norm,
                "grassmanian_distance": grass_dist,
                "cossim_base_fiber": cossim_base_fiber,
                "prediction_horizon_average": prediction_horizon_average,
                "prediction_horizon_variance": float(np.var(horizons)),
                "min_prediction_horizon": horizon_min,
                "horizons": horizons,
                "prediction_horizon_smape_average": prediction_horizon_smape_average,
                "min_prediction_horizon_smape": horizon_smape_min,
                "horizons_smape": horizons_smape,
                "normalized_states_singvals": normalized_singular_values,
                "effective_rank": effective_rank,
                "average_distance_from_base": float(np.mean(distance_from_base)),
                "std_distance_from_base": float(np.std(distance_from_base)),
                "lyapunov_spectrum": tangent_normal.get("lyapunov_spectrum"),
                "lyapunov_spectrum_s": tangent_normal.get("lyapunov_spectrum_s"),
                "lyapunov_error": tangent_normal.get("lyapunov_error"),
                "lyapunov_s_error": tangent_normal.get("lyapunov_s_error"),
                "conditional_lyapunov_spectrum": conditional_le.get("lyapunov_spectrum"),
                "conditional_lyapunov_spectrum_s": conditional_le.get("lyapunov_spectrum_s"),
                "success": True,
                "methods_param_name": method_param_name,
                "timestamp": _datetime.datetime.now().isoformat(),
            }

            self._generate_analysis_plots(
                params=params,
                result=result,
                close_eigvecs_sorted=close_eigvecs_sorted,
                invariant_dim=params["invariant_dim"],
            )
            return result
        except Exception as exc:
            error_traceback = traceback.format_exc()
            return {
                **params,
                "error": str(exc),
                "error_traceback": error_traceback,
                "success": False,
                "timestamp": _datetime.datetime.now().isoformat(),
            }

    def _generate_analysis_plots(
        self,
        params: dict,
        result: dict,
        close_eigvecs_sorted: np.ndarray,
        invariant_dim: int,
    ) -> None:
        plotter = WhyBadCESNPlotter(self.current_sweep_dir)
        if self.plot_config.get("mode_plot_eigenvalue_distribution", False):
            plotter.plot_eigenvalue_distribution(
                alpha=params["alpha"],
                left_plot_eigenvals=result["W_res_eigenvals"],
                right_plot_eigenvals=result["W_close_eigenvals"],
                left_spectral_radius=result["W_res_Actual_spectral_radius"],
                right_spectral_radius=result["W_close_spectral_radius"],
                left_fix_point_norm=None,
                right_fix_point_norm=None,
                list_used_indices=self.cesn_component.list_used_indices,
                inv_dim=invariant_dim,
                spectral_radius=params["spectral_radius"],
                grassmanian_distance=result["grassmanian_distance"],
                filename=result["methods_param_name"][:100],
                save_dir=self.current_sweep_dir / "eigenvalues",
            )


def _method_name(params: dict, component: CreateCESN_Component) -> str:
    if params["use_eigenvecs_method_or_list"] is None:
        return f"{params['w_res_method']}_{params['w_in_method']}_{params['bias_method']}"
    if isinstance(params["use_eigenvecs_method_or_list"], str):
        return (
            f"{params['w_res_method']}_{params['w_in_method']}_{params['use_eigenvecs_method_or_list']}"
            f"_pair_{params['conj_pair_num']}_invdim_{params['invariant_dim']}_{params['construct_option']}_{params['bias_method']}"
        )
    return (
        f"{params['w_res_method']}_{params['w_in_method']}_{component.str_used_indices}"
        f"_pair_{params['conj_pair_num']}_invdim_{params['invariant_dim']}_{params['construct_option']}_{params['bias_method']}"
    )


def _param_name(params: dict) -> str:
    return (
        f"seed{params['seed']}_bamp_{params['bias_amplitude']}_isr_{params['input_scaling_ratio']}"
        f"_alpha_{params['alpha']}_dim{params['dim_reservoir']}_sp_{params['spectral_radius']}"
        f"_rg_{params['regularization']}_po_{params['partial_observe']}"
        f"_hl_{params['how_loop']}_tr_{params['how_train']}"
    )
