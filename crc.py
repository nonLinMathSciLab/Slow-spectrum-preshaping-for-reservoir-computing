from __future__ import annotations

import numpy as np


def _complex_tanh(z: np.ndarray) -> np.ndarray:
    abs_z = np.abs(z)
    scale = np.ones_like(abs_z, dtype=float)
    mask = abs_z > 0
    scale[mask] = np.tanh(abs_z[mask]) / abs_z[mask]
    return scale * z


def _real_tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


class CreateCESN_Component:
    """Create the reservoir, input matrix, and bias used by the CESN."""

    def __init__(
        self,
        dim_inputs: int,
        dim_reservoir: int,
        spectral_radius: float,
        alpha: float,
        seed: int = 42,
        gpu_mode: bool = False,
    ) -> None:
        if gpu_mode:
            raise ValueError("The public compact code uses NumPy CPU mode only.")
        self.dim_inputs = dim_inputs
        self.dim_reservoir = dim_reservoir
        self.spectral_radius = spectral_radius
        self.alpha = alpha
        self.seed = seed
        self.w_res: np.ndarray | None = None
        self.w_in: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.eigenvals_scaled: np.ndarray | None = None
        self.eigenvecs_sorted: np.ndarray | None = None
        self.conj_pair_idx_list: list[list[int]] = []
        self.list_used_indices: list[list[int]] | None = None
        self.str_used_indices: str | None = None
        self.P: np.ndarray | None = None

    @staticmethod
    def normalize_columns(matrix: np.ndarray) -> np.ndarray:
        normalized = np.array(matrix, copy=True)
        for idx in range(normalized.shape[1]):
            norm = np.linalg.norm(normalized[:, idx])
            if norm > 1e-12:
                normalized[:, idx] /= norm
        return normalized

    def _build_conjugate_groups(self) -> None:
        groups: list[list[int]] = []
        used: set[int] = set()
        assert self.eigenvals_scaled is not None

        for idx, eigval in enumerate(self.eigenvals_scaled):
            if idx in used:
                continue
            if np.isclose(eigval.imag, 0.0, atol=1e-10):
                groups.append([idx])
                used.add(idx)
                continue
            partner = None
            for cand, cand_eigval in enumerate(self.eigenvals_scaled):
                if cand == idx or cand in used:
                    continue
                if np.isclose(cand_eigval, np.conj(eigval), atol=1e-10):
                    partner = cand
                    break
            if partner is None:
                groups.append([idx])
                used.add(idx)
            else:
                groups.append([idx, partner])
                used.update({idx, partner})
        self.conj_pair_idx_list = groups

    def modify_eigenstructure(self, matrix: np.ndarray, updates) -> np.ndarray:
        """Apply the invariant-subspace eigenvalue replacement used in sweeps."""

        if updates is False or updates is None:
            return matrix

        evals, right = np.linalg.eig(matrix)
        # For a diagonalizable matrix, columns of inv(right).T are left eigenvectors.
        left = np.linalg.inv(right).T

        x_cols = []
        y_cols = []
        blocks = []
        for indices, new_value in updates:
            if isinstance(indices, int):
                x_cols.append(right[:, indices].real)
                y_cols.append(left[:, indices].real)
                blocks.append(np.array([[np.real(new_value)]], dtype=float))
            elif isinstance(indices, (tuple, list)) and len(indices) == 2:
                i, j = indices
                if np.iscomplexobj(evals[i]) and not np.isclose(evals[i].imag, 0):
                    x_cols.extend([right[:, i].real, right[:, i].imag])
                    y_cols.extend([left[:, i].real, left[:, i].imag])
                else:
                    x_cols.extend([right[:, i].real, right[:, j].real])
                    y_cols.extend([left[:, i].real, left[:, j].real])
                if np.iscomplexobj(new_value) and not np.isclose(np.imag(new_value), 0):
                    a = np.real(new_value)
                    b = np.imag(new_value)
                    blocks.append(np.array([[a, b], [-b, a]], dtype=float))
                else:
                    a = np.real(new_value)
                    blocks.append(np.array([[a, 0.0], [0.0, a]], dtype=float))
            else:
                raise ValueError("Unsupported W_res_force_updates entry")

        if not x_cols:
            return matrix

        x_basis = np.column_stack(x_cols)
        y_raw = np.column_stack(y_cols)
        y_basis = y_raw @ np.linalg.inv(y_raw.T @ x_basis).T
        old_block = y_basis.T @ matrix @ x_basis
        new_block = _block_diag(blocks)
        return matrix + x_basis @ (new_block - old_block) @ y_basis.T

    def _scale_matrix(self, matrix: np.ndarray, W_res_force_updates=False) -> None:
        eigvals = np.linalg.eigvals(matrix)
        radius = np.max(np.abs(eigvals))
        if radius == 0:
            raise ValueError("Reservoir matrix has zero spectral radius")
        self.w_res = matrix * (self.spectral_radius / radius)
        self.w_res = self.modify_eigenstructure(self.w_res, W_res_force_updates)

        essential = (1.0 - self.alpha) * np.eye(self.dim_reservoir) + self.alpha * self.w_res
        eigvals, eigvecs = np.linalg.eig(essential)
        order = np.argsort(eigvals.real)[::-1]
        self.eigenvals_scaled = eigvals[order]
        self.eigenvecs_sorted = eigvecs[:, order]
        self._build_conjugate_groups()

    def create_reservoir_matrix(
        self,
        method: str,
        W_res_force_updates=False,
        er_prob: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)

        if method == "symmetry":
            raw = rng.uniform(-1.0, 1.0, (self.dim_reservoir, self.dim_reservoir))
            matrix = (raw + raw.T) / 2.0
        elif method == "random":
            matrix = rng.uniform(-1.0, 1.0, (self.dim_reservoir, self.dim_reservoir))
        elif method == "ER":
            mask = rng.random((self.dim_reservoir, self.dim_reservoir)) < er_prob
            matrix = mask * rng.uniform(-1.0, 1.0, (self.dim_reservoir, self.dim_reservoir))
        else:
            raise ValueError(f"Unsupported reservoir method: {method}")

        self._scale_matrix(matrix, W_res_force_updates)
        assert self.w_res is not None
        assert self.eigenvals_scaled is not None
        assert self.eigenvecs_sorted is not None
        if np.allclose(self.w_res.imag, 0.0, atol=1e-12):
            self.w_res = self.w_res.real
        return self.w_res, self.eigenvals_scaled, self.eigenvecs_sorted

    def _select_invariant_indices(
        self,
        use_method,
        conj_pair_num,
        invariant_dim: int,
    ) -> list[list[int]]:
        selected: list[list[int]] = []
        current_dim = 0

        if conj_pair_num != "auto":
            raise ValueError("The compact public code supports conj_pair_num='auto'.")

        if use_method == "high":
            search_list = self.conj_pair_idx_list
        elif use_method == "low":
            search_list = list(reversed(self.conj_pair_idx_list))
        elif use_method == "intermediate":
            mid = len(self.conj_pair_idx_list) // 2
            search_list = sorted(
                self.conj_pair_idx_list,
                key=lambda group: abs(self.conj_pair_idx_list.index(group) - mid),
            )
        elif isinstance(use_method, list):
            search_list = [self.conj_pair_idx_list[idx] for idx in use_method]
        else:
            raise ValueError(f"Unsupported eigenvector selection: {use_method}")

        for group in search_list:
            group_dim = len(group)
            if current_dim + group_dim <= invariant_dim:
                selected.append(group)
                current_dim += group_dim
            if current_dim == invariant_dim:
                break

        if current_dim != invariant_dim:
            raise ValueError(
                f"Could not select exactly {invariant_dim} invariant dimensions; got {current_dim}."
            )
        return selected

    def _construct_invariant_subspace(
        self,
        selected_groups: list[list[int]],
        eigenvecs_sorted: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        columns = []
        labels = []

        for group in selected_groups:
            if len(group) == 1:
                idx = group[0]
                vec = eigenvecs_sorted[:, idx]
                columns.append(vec.real)
                labels.append(str(idx))
            elif len(group) == 2:
                i, j = group
                vec_i = eigenvecs_sorted[:, i]
                vec_j = eigenvecs_sorted[:, j]
                columns.append((vec_i + vec_j).real)
                columns.append((vec_i - vec_j).imag)
                labels.extend([f"Re({i})", f"Im({i})"])
            else:
                raise ValueError("Only real eigenvectors and conjugate pairs are supported.")

        basis = np.column_stack(columns)
        return self.normalize_columns(basis), "[" + ",".join(labels) + "]"

    def _construct_w_in_from_subspace(
        self,
        basis: np.ndarray,
        construct_option: str,
        input_dim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        invariant_dim = basis.shape[1]
        if construct_option == "sonomama":
            if input_dim != invariant_dim:
                raise ValueError("sonomama requires input_dim == invariant_dim")
            return basis
        if construct_option == "just_add":
            if invariant_dim % input_dim != 0:
                raise ValueError("just_add requires invariant_dim to be a multiple of input_dim")
            w_in = np.zeros((basis.shape[0], input_dim), dtype=basis.dtype)
            each_dim = invariant_dim // input_dim
            for idx in range(input_dim):
                start = idx * each_dim
                stop = invariant_dim if idx == input_dim - 1 else (idx + 1) * each_dim
                w_in[:, idx] = basis[:, start:stop].sum(axis=1)
            return w_in
        if construct_option == "random":
            q_basis, _ = np.linalg.qr(basis, mode="reduced")
            coeffs = rng.standard_normal((invariant_dim, input_dim))
            return q_basis @ coeffs
        raise ValueError(f"Unsupported construct_option: {construct_option}")

    def create_input_weight_matrix(
        self,
        method: str,
        w_res: np.ndarray,
        eigenvecs_sorted: np.ndarray,
        invariant_dim: int,
        conj_pair_num="auto",
        construct_option: str = "random",
        input_scaling: float = 1.0,
        use_eigenvecs_method_or_list=None,
        sparsity: float = 1.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        self.list_used_indices = None
        self.str_used_indices = None

        if method == "eigenvec":
            if use_eigenvecs_method_or_list is None:
                raise ValueError("eigenvec input weights require a selection rule")
            selected = self._select_invariant_indices(
                use_eigenvecs_method_or_list,
                conj_pair_num,
                invariant_dim,
            )
            self.list_used_indices = selected
            basis, used_label = self._construct_invariant_subspace(selected, eigenvecs_sorted)
            self.str_used_indices = used_label

            remaining = [group for group in self.conj_pair_idx_list if group not in selected]
            if remaining:
                complement, _ = self._construct_invariant_subspace(remaining, eigenvecs_sorted)
                self.P = np.column_stack([basis, complement])
            else:
                self.P = basis

            w_in = self._construct_w_in_from_subspace(
                basis=basis,
                construct_option=construct_option,
                input_dim=self.dim_inputs,
                rng=rng,
            )
        elif method == "random":
            w_in = rng.uniform(-1.0, 1.0, (self.dim_reservoir, self.dim_inputs))
        elif method == "sparse_random":
            w_in = rng.uniform(-1.0, 1.0, (self.dim_reservoir, self.dim_inputs))
            mask = rng.random(w_in.shape) < sparsity
            w_in = w_in * mask
        else:
            raise ValueError(f"Unsupported input method: {method}")

        self.w_in = input_scaling * self.normalize_columns(w_in)
        return self.w_in

    def create_bias_vector(
        self,
        method: str,
        eigenvecs_sorted: np.ndarray,
        bias_amplitude: float,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if method == "const":
            bias = bias_amplitude * np.ones(self.dim_reservoir)
        elif method == "zero":
            bias = np.zeros(self.dim_reservoir)
        elif method == "random":
            bias = bias_amplitude * rng.uniform(-1.0, 1.0, self.dim_reservoir)
        elif method == "eigenvec":
            bias = bias_amplitude * eigenvecs_sorted[:, 0].real
        else:
            raise ValueError(f"Unsupported bias method: {method}")
        self.bias = bias
        return bias

    def create_all_components(
        self,
        w_res_method: str,
        w_in_method: str,
        conj_pair_num,
        invariant_dim: int,
        construct_option: str,
        bias_method: str,
        bias_amplitude: float,
        input_scaling: float = 1.0,
        use_eigenvecs_method_or_list=None,
        W_res_force_updates=False,
        er_prob: float = 1.0,
        input_sparsity: float = 1.0,
        inv_dim_list=None,
    ) -> dict:
        w_res, eigenvals, eigenvecs = self.create_reservoir_matrix(
            w_res_method,
            W_res_force_updates=W_res_force_updates,
            er_prob=er_prob,
        )
        w_in = self.create_input_weight_matrix(
            w_in_method,
            w_res,
            eigenvecs,
            invariant_dim,
            conj_pair_num=conj_pair_num,
            construct_option=construct_option,
            input_scaling=input_scaling,
            use_eigenvecs_method_or_list=use_eigenvecs_method_or_list,
            sparsity=input_sparsity,
        )
        bias = self.create_bias_vector(bias_method, eigenvecs, bias_amplitude)
        return {
            "W_res": w_res,
            "W_in": w_in,
            "bias": bias,
            "eigenvals_sorted": eigenvals,
            "eigenvecs_sorted": eigenvecs,
            "trace": np.trace(w_res),
            "determinant": np.linalg.det(w_res),
        }


class ComplexEchoStateNetwork:
    """Complex-valued echo state network used by the figure experiments."""

    def __init__(
        self,
        dim_inputs: int,
        dim_reservoir: int,
        dim_outputs: int,
        is_realdata: bool,
        W_in: np.ndarray,
        W_res: np.ndarray,
        alpha: float,
        bias: np.ndarray,
        regularization: float,
        is_intercept: bool,
        partial_observe_list: list,
        delay: int = 0,
        max_delay: int = 0,
        force_complex_mode: bool = False,
        gpu_mode: bool = False,
    ) -> None:
        if gpu_mode:
            raise ValueError("The public compact code uses NumPy CPU mode only.")
        if max_delay:
            raise ValueError("The compact public code supports the no-delay figure setting.")

        self.dim_inputs = dim_inputs
        self.dim_reservoir = dim_reservoir
        self.dim_outputs = dim_outputs
        self.alpha = alpha
        self.regularization = regularization
        self.is_intercept = is_intercept
        self.partial_observe_list = partial_observe_list
        self.delay = delay
        self.max_delay = max_delay
        self.W_in = _as_real_if_possible(W_in)
        self.W_res = _as_real_if_possible(W_res)
        self.bias = _as_real_if_possible(np.asarray(bias))
        self.complex_mode = (
            force_complex_mode
            or (not is_realdata)
            or np.iscomplexobj(self.W_in)
            or np.iscomplexobj(self.W_res)
            or np.iscomplexobj(self.bias)
        )
        self.activation = _complex_tanh if self.complex_mode else _real_tanh
        dtype = np.complex128 if self.complex_mode else np.float64
        self.intercept = np.zeros((self.dim_outputs, 1), dtype=dtype)
        self.W_out: np.ndarray | None = None
        self.W_close: np.ndarray | None = None
        self.W_in_intercept = np.zeros(self.dim_reservoir, dtype=dtype)

    def _update(self, u: np.ndarray, state: np.ndarray) -> np.ndarray:
        pre_activation = self.W_in @ u + self.W_res @ state + self.bias
        return self.alpha * self.activation(pre_activation) + (1.0 - self.alpha) * state

    def listen(
        self,
        U: np.ndarray,
        warm_up: int,
        initial_seed: int | None = None,
        init_state_amp: float = 0.1,
        initial_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        U = _as_real_if_possible(np.asarray(U))
        dtype = np.complex128 if self.complex_mode else np.float64
        washout = np.zeros((warm_up + 1, self.dim_reservoir), dtype=dtype)
        states = np.zeros((U.shape[0] - warm_up + 1, self.dim_reservoir), dtype=dtype)

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=dtype)
        elif initial_seed is not None:
            rng = np.random.default_rng(initial_seed)
            state = init_state_amp * rng.uniform(-1.0, 1.0, self.dim_reservoir).astype(dtype)
        else:
            state = init_state_amp * np.ones(self.dim_reservoir, dtype=dtype)

        washout[0] = state
        t = 0
        for idx in range(warm_up):
            washout[idx + 1] = self._update(U[t], washout[idx])
            t += 1

        states[0] = washout[-1]
        for idx in range(U.shape[0] - warm_up):
            states[idx + 1] = self._update(U[t], states[idx])
            t += 1
        return washout, states

    def training_lstsq(
        self,
        states: np.ndarray,
        train_data: np.ndarray,
        P: np.ndarray | None = None,
        inv_dim: int | None = None,
    ) -> None:
        if len(states) != len(train_data):
            raise ValueError("states and train_data must have the same length")

        states = np.asarray(states)
        train_data = _as_real_if_possible(np.asarray(train_data))
        if P is not None and inv_dim is not None:
            p_inv = np.linalg.inv(P)
            states = states @ p_inv[:, :inv_dim]

        if self.is_intercept:
            design = np.hstack([states, np.ones((states.shape[0], 1), dtype=states.dtype)])
        else:
            design = states

        reg = np.sqrt(self.regularization) * np.eye(design.shape[1], dtype=design.dtype)
        if self.is_intercept:
            reg[-1, :] = 0
            reg[:, -1] = 0
        design_ridge = np.vstack([design, reg])
        target_ridge = np.vstack([train_data, np.zeros((design.shape[1], train_data.shape[1]), dtype=train_data.dtype)])
        solution, *_ = np.linalg.lstsq(design_ridge, target_ridge, rcond=None)

        if self.is_intercept:
            self.intercept = solution[-1, :].reshape(-1, 1)
            solution = solution[:-1, :]
        else:
            self.intercept = np.zeros((self.dim_outputs, 1), dtype=solution.dtype)

        self.W_out = solution.T
        if P is not None and inv_dim is not None:
            p_inv = np.linalg.inv(P)
            full = np.hstack([self.W_out, np.zeros((self.W_out.shape[0], P.shape[1] - inv_dim))])
            self.W_out = full @ p_inv

        self.W_close = self.W_res + self.W_in @ self.W_out
        self.W_in_intercept = (self.W_in @ self.intercept).squeeze()

    def average_prediction_horizon_numba(
        self,
        test_data: np.ndarray,
        max_delay: int,
        no_delay_indices: list[int],
        warmuped: np.ndarray,
        training_step: int,
        pred_start_point: int,
        pred_start_point_end: int,
        over_num: int,
        steps_ahead: int,
        lyapunov_time: float = 1.0,
    ) -> tuple[float, float, np.ndarray, float, float, np.ndarray]:
        if max_delay:
            raise ValueError("The compact public code supports the no-delay figure setting.")
        if self.W_out is None:
            raise ValueError("The network must be trained before prediction")

        starts = np.linspace(pred_start_point, pred_start_point_end, over_num, dtype=int)
        variance = np.var(test_data, axis=0)
        variance = np.where(variance < 1e-12, 1.0, variance)
        states = warmuped[training_step + starts].copy()
        output_indices = np.asarray(no_delay_indices, dtype=int)
        horizons_arr = np.full(over_num, np.nan, dtype=float)
        horizons_smape_arr = np.full(over_num, np.nan, dtype=float)
        running_smape = np.zeros(over_num, dtype=float)

        available_steps = len(test_data) - training_step - starts
        max_steps = int(min(steps_ahead, np.min(available_steps)))
        active_horizon = np.ones(over_num, dtype=bool)
        active_smape = np.ones(over_num, dtype=bool)

        for step in range(max_steps):
            rc_prediction = states @ self.W_out.T + self.intercept.ravel()
            prediction = rc_prediction[:, output_indices]
            truth = test_data[training_step + starts + step]

            nrmse = np.sqrt(np.average((np.abs(truth - prediction) ** 2) / variance, axis=1))
            newly_bad = active_horizon & (nrmse > 0.5)
            horizons_arr[newly_bad] = (step + 1) / lyapunov_time
            active_horizon[newly_bad] = False

            denom = np.abs(truth) + np.abs(prediction)
            denom = np.where(denom < 1e-12, 1.0, denom)
            step_smape = 200.0 * np.mean(np.abs(truth - prediction) / denom, axis=1)
            running_smape = (step * running_smape + step_smape) / (step + 1)
            newly_smape_bad = active_smape & (running_smape > 50.0)
            horizons_smape_arr[newly_smape_bad] = (step + 1) / lyapunov_time
            active_smape[newly_smape_bad] = False

            states = self.alpha * self.activation(
                states @ self.W_res.T + rc_prediction @ self.W_in.T + self.bias
            ) + (1.0 - self.alpha) * states

            if not active_horizon.any() and not active_smape.any():
                break

        horizons_arr[np.isnan(horizons_arr)] = max_steps / lyapunov_time
        horizons_smape_arr[np.isnan(horizons_smape_arr)] = max_steps / lyapunov_time
        return (
            float(np.mean(horizons_arr)),
            float(np.min(horizons_arr)),
            horizons_arr,
            float(np.mean(horizons_smape_arr)),
            float(np.min(horizons_smape_arr)),
            horizons_smape_arr,
        )

    def _closed_loop_step_from_output(self, state: np.ndarray, output: np.ndarray) -> np.ndarray:
        pre_activation = self.W_res @ state + self.W_in @ output + self.bias
        return self.alpha * self.activation(pre_activation) + (1.0 - self.alpha) * state

    def feedback_predict_start(
        self,
        warmup_state: np.ndarray,
        warmup_step: int,
        pred_start_time: int,
        test_data: np.ndarray,
        no_delay_indices: list[int],
        steps_ahead: int = 100,
    ) -> np.ndarray:
        if self.W_out is None:
            raise ValueError("The network must be trained before prediction")
        dtype = np.complex128 if self.complex_mode else np.float64
        predictions = np.zeros((warmup_step + pred_start_time + steps_ahead, self.dim_outputs), dtype=dtype)
        state = np.asarray(warmup_state, dtype=dtype).copy()
        start_idx = warmup_step + pred_start_time
        for step in range(steps_ahead):
            rc_prediction = self.W_out @ state + self.intercept.squeeze()
            predictions[start_idx + step] = rc_prediction[no_delay_indices]
            state = self._closed_loop_step_from_output(state, rc_prediction)
        return predictions


def _as_real_if_possible(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array) and np.allclose(array.imag, 0.0, atol=1e-12):
        return array.real.astype(np.float64)
    return array


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        width = block.shape[0]
        result[cursor : cursor + width, cursor : cursor + width] = block
        cursor += width
    return result
