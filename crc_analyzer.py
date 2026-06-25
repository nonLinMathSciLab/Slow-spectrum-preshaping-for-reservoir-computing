from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


def grassmanian_distance(P1: np.ndarray, P2: np.ndarray) -> float:
    """Grassmann distance between the column spaces of two matrices."""

    q1, _ = np.linalg.qr(P1, mode="reduced")
    q2, _ = np.linalg.qr(P2, mode="reduced")
    singular_values = np.linalg.svd(q1.conj().T @ q2, compute_uv=False)
    singular_values = np.clip(singular_values.real, -1.0, 1.0)
    angles = np.arccos(singular_values)
    return float(np.sqrt(np.sum(angles**2)))


def find_fixed_point_numba(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float, int, bool]:
    """Newton solve for x = tanh(Ax + b). Name kept for API compatibility."""

    identity = np.eye(A.shape[0], dtype=A.dtype)
    try:
        x = np.linalg.solve(identity - A, b)
    except np.linalg.LinAlgError:
        x = np.zeros(A.shape[0], dtype=A.dtype)

    for iteration in range(max_iter):
        z = A @ x + b
        tanh_z = np.tanh(z)
        residual = x - tanh_z
        if np.linalg.norm(residual) < tol:
            return x, float(np.linalg.norm(x)), iteration, True

        sech2 = 1.0 - tanh_z**2
        jacobian = identity - sech2[:, None] * A
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            break
        x = x + step

    return x, float(np.linalg.norm(x)), max_iter, False


def eachpoint_tangent_normal_analysis(
    attractor_points: np.ndarray,
    alpha: float,
    W_cl: np.ndarray,
    P: np.ndarray,
    dim_invariant: int,
    dt: float,
    calc_only_ls: bool,
    true_ls=None,
    gpu_mode: bool = False,
) -> dict:
    """Compute Lyapunov-spectrum and optional tangent-normal diagnostics."""

    if gpu_mode and calc_only_ls:
        if cp is None:
            raise RuntimeError("gpu_mode=True requires CuPy for Lyapunov-spectrum calculations.")
        return _eachpoint_tangent_normal_analysis_cupy_ls(
            attractor_points=attractor_points,
            alpha=alpha,
            W_cl=W_cl,
            P=P,
            dim_invariant=dim_invariant,
            dt=dt,
            true_ls=true_ls,
        )

    attractor_points = np.asarray(attractor_points)
    P = np.asarray(P, dtype=np.complex128)
    P_inv = np.linalg.inv(P)
    n_points, dim_reservoir = attractor_points.shape
    ls_nums = min(dim_reservoir, dim_invariant * 2)
    # Keep Q thin so the Lyapunov QR update is always reduced.
    q_full = np.eye(dim_reservoir, ls_nums, dtype=np.complex128)
    q_base = np.eye(dim_invariant, dtype=np.complex128)
    lyapunov = np.zeros(ls_nums)
    lyapunov_base = np.zeros(dim_invariant)

    fiber_spectral_radii = np.zeros(n_points)
    base_min_eigenval_list = np.zeros(n_points)
    base_spectral_radii = np.zeros(n_points)
    fiber_to_fiber_max_sigval_list = np.zeros(n_points)
    base_to_fiber_max_sigval_list = np.zeros(n_points)
    all_to_fiber_max_sigval_list = np.zeros(n_points)

    unstable_eig = 0
    unstable_fiber_sv = 0
    unstable_all_fiber_sv = 0
    gap_count = 0
    calctime = 1.0

    for idx in range(max(n_points - 1, 0)):
        x_i = (attractor_points[idx + 1] - (1.0 - alpha) * attractor_points[idx]) / alpha
        D = np.diag(1.0 - x_i**2)
        jacobian = (1.0 - alpha) * np.eye(dim_reservoir) + alpha * D @ W_cl
        jacobian = jacobian.astype(np.complex128)
        transformed = P_inv @ jacobian @ P
        base_block = transformed[:dim_invariant, :dim_invariant]

        q_full, r_full = np.linalg.qr(jacobian @ q_full, mode="reduced")
        diag_full = np.diag(r_full)
        for j in range(ls_nums):
            log_val = np.log(max(np.abs(diag_full[j]), 1e-300))
            lyapunov[j] += (log_val - lyapunov[j]) / calctime

        q_base, r_base = np.linalg.qr(base_block @ q_base, mode="reduced")
        diag_base = np.diag(r_base)
        for j in range(dim_invariant):
            log_val = np.log(max(np.abs(diag_base[j]), 1e-300))
            lyapunov_base[j] += (log_val - lyapunov_base[j]) / calctime

        calctime += 1.0

        if not calc_only_ls and dim_invariant < dim_reservoir:
            fiber_block = transformed[dim_invariant:, dim_invariant:]
            fiber_eigs = np.linalg.eigvals(fiber_block)
            fiber_radius = float(np.max(np.abs(fiber_eigs)))
            fiber_spectral_radii[idx] = fiber_radius

            base_eigs = np.linalg.eigvals(base_block)
            base_radius = float(np.max(np.abs(base_eigs)))
            base_min = float(np.min(np.abs(base_eigs)))
            base_spectral_radii[idx] = base_radius
            base_min_eigenval_list[idx] = base_min

            base_to_fiber = float(np.linalg.norm(transformed[dim_invariant:, :dim_invariant], ord=2))
            fiber_to_fiber = float(np.linalg.norm(fiber_block, ord=2))
            all_to_fiber = float(np.linalg.norm(transformed[dim_invariant:, :], ord=2))
            base_to_fiber_max_sigval_list[idx] = base_to_fiber
            fiber_to_fiber_max_sigval_list[idx] = fiber_to_fiber
            all_to_fiber_max_sigval_list[idx] = all_to_fiber

            unstable_eig += int(fiber_radius >= 1.0)
            unstable_fiber_sv += int(fiber_to_fiber >= 1.0)
            unstable_all_fiber_sv += int(all_to_fiber >= 1.0)
            gap_count += int(base_min > fiber_radius)

    divisor = max(dt, 1e-300)
    lyapunov = lyapunov / divisor
    lyapunov_base = lyapunov_base / divisor

    count = max(n_points, 1)
    stability_ratio_eigenval = None
    stability_ratio_fb_fb_singval = None
    stability_ratio_all_fb_singval = None
    gap_ratio_eigenval = None
    ave_log_all_to_fiber = None
    if not calc_only_ls:
        stability_ratio_eigenval = (count - unstable_eig) / count * 100.0
        stability_ratio_fb_fb_singval = (count - unstable_fiber_sv) / count * 100.0
        stability_ratio_all_fb_singval = (count - unstable_all_fiber_sv) / count * 100.0
        gap_ratio_eigenval = (count - gap_count) / count * 100.0
        positive = all_to_fiber_max_sigval_list[all_to_fiber_max_sigval_list > 0]
        if len(positive):
            ave_log_all_to_fiber = float(np.mean(np.log(positive)))

    lyapunov_error = None
    lyapunov_s_error = None
    if true_ls is not None:
        true_ls = np.asarray(true_ls, dtype=float)
        scale = max(abs(true_ls[0]), 1e-12)
        width = min(len(true_ls), len(lyapunov), len(lyapunov_base))
        lyapunov_error = float(np.sum(np.abs(lyapunov[:width] - true_ls[:width]) / scale))
        lyapunov_s_error = float(np.sum(np.abs(lyapunov_base[:width] - true_ls[:width]) / scale))

    return {
        "fiber_spectral_radii": fiber_spectral_radii,
        "base_min_eigenval_list": base_min_eigenval_list,
        "base_spectral_radii": base_spectral_radii,
        "fiber_to_fiber_max_sigval_list": fiber_to_fiber_max_sigval_list,
        "base_to_fiber_max_sigval_list": base_to_fiber_max_sigval_list,
        "all_to_fiber_max_sigval_list": all_to_fiber_max_sigval_list,
        "stability_ratio_eigenval": stability_ratio_eigenval,
        "stability_ratio_fb_fb_singval": stability_ratio_fb_fb_singval,
        "stability_ratio_all_fb_singval": stability_ratio_all_fb_singval,
        "gap_ratio_eigenval": gap_ratio_eigenval,
        "lyapunov_spectrum": lyapunov,
        "lyapunov_spectrum_s": lyapunov_base,
        "num_attractor_points": n_points,
        "ave_log_all_to_fiber_max_sigval": ave_log_all_to_fiber,
        "lyapunov_error": lyapunov_error,
        "lyapunov_s_error": lyapunov_s_error,
    }


def _eachpoint_tangent_normal_analysis_cupy_ls(
    attractor_points: np.ndarray,
    alpha: float,
    W_cl: np.ndarray,
    P: np.ndarray,
    dim_invariant: int,
    dt: float,
    true_ls=None,
) -> dict:
    """Compute only Lyapunov spectra on GPU using the same reduced-QR update."""

    attractor_points_gpu = cp.asarray(attractor_points)
    W_cl_gpu = cp.asarray(W_cl, dtype=cp.complex128)
    P_gpu = cp.asarray(P, dtype=cp.complex128)
    P_inv = cp.linalg.inv(P_gpu)
    n_points, dim_reservoir = attractor_points_gpu.shape
    ls_nums = min(dim_reservoir, dim_invariant * 2)
    eye_reservoir = cp.eye(dim_reservoir, dtype=cp.complex128)
    q_full = cp.eye(dim_reservoir, ls_nums, dtype=cp.complex128)
    q_base = cp.eye(dim_invariant, dtype=cp.complex128)
    lyapunov = cp.zeros(ls_nums)
    lyapunov_base = cp.zeros(dim_invariant)

    calctime = 1.0
    for idx in range(max(n_points - 1, 0)):
        x_i = (attractor_points_gpu[idx + 1] - (1.0 - alpha) * attractor_points_gpu[idx]) / alpha
        diag_values = 1.0 - x_i**2
        jacobian = (1.0 - alpha) * eye_reservoir + alpha * diag_values[:, None] * W_cl_gpu
        transformed = P_inv @ jacobian @ P_gpu
        base_block = transformed[:dim_invariant, :dim_invariant]

        q_full, r_full = cp.linalg.qr(jacobian @ q_full, mode="reduced")
        full_logs = cp.log(cp.maximum(cp.abs(cp.diag(r_full)), 1e-300))
        lyapunov += (full_logs - lyapunov) / calctime

        q_base, r_base = cp.linalg.qr(base_block @ q_base, mode="reduced")
        base_logs = cp.log(cp.maximum(cp.abs(cp.diag(r_base)), 1e-300))
        lyapunov_base += (base_logs - lyapunov_base) / calctime

        calctime += 1.0

    divisor = max(dt, 1e-300)
    lyapunov = cp.asnumpy(lyapunov / divisor)
    lyapunov_base = cp.asnumpy(lyapunov_base / divisor)
    n_points_int = int(n_points)

    lyapunov_error = None
    lyapunov_s_error = None
    if true_ls is not None:
        true_ls = np.asarray(true_ls, dtype=float)
        scale = max(abs(true_ls[0]), 1e-12)
        width = min(len(true_ls), len(lyapunov), len(lyapunov_base))
        lyapunov_error = float(np.sum(np.abs(lyapunov[:width] - true_ls[:width]) / scale))
        lyapunov_s_error = float(np.sum(np.abs(lyapunov_base[:width] - true_ls[:width]) / scale))

    return {
        "fiber_spectral_radii": np.zeros(n_points_int),
        "base_min_eigenval_list": np.zeros(n_points_int),
        "base_spectral_radii": np.zeros(n_points_int),
        "fiber_to_fiber_max_sigval_list": np.zeros(n_points_int),
        "base_to_fiber_max_sigval_list": np.zeros(n_points_int),
        "all_to_fiber_max_sigval_list": np.zeros(n_points_int),
        "stability_ratio_eigenval": None,
        "stability_ratio_fb_fb_singval": None,
        "stability_ratio_all_fb_singval": None,
        "gap_ratio_eigenval": None,
        "lyapunov_spectrum": lyapunov,
        "lyapunov_spectrum_s": lyapunov_base,
        "num_attractor_points": n_points_int,
        "ave_log_all_to_fiber_max_sigval": None,
        "lyapunov_error": lyapunov_error,
        "lyapunov_s_error": lyapunov_s_error,
    }


def distance_from_subspace(
    data_points: np.ndarray,
    basis: np.ndarray,
    dim_principal: int,
    slide: np.ndarray | None = None,
) -> np.ndarray:
    """Distance from each row in data_points to the leading basis subspace."""

    principal = _real_basis_from_eigenvectors(basis, dim_principal)
    data = data_points - slide if slide is not None else data_points
    q_basis, _ = np.linalg.qr(principal, mode="reduced")
    projected = data @ q_basis @ q_basis.T
    return np.linalg.norm(data - projected, axis=1)


def similarity_with_subspace(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Projection norm of each column of W onto the column space of V."""

    u, singular_values, _ = np.linalg.svd(V, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    if rank == 0:
        return np.zeros(W.shape[1])
    basis = u[:, :rank]
    norms = np.linalg.norm(W, axis=0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    W_normalized = W / safe_norms
    return np.linalg.norm(basis.conj().T @ W_normalized, axis=0)


def _real_basis_from_eigenvectors(basis: np.ndarray, dim_principal: int) -> np.ndarray:
    columns = []
    used: set[int] = set()
    idx = 0
    while len(columns) < dim_principal and idx < basis.shape[1]:
        if idx in used:
            idx += 1
            continue
        vec = basis[:, idx]
        if np.allclose(vec.imag, 0.0, atol=1e-10):
            columns.append(vec.real)
            used.add(idx)
            idx += 1
            continue
        partner = None
        for cand in range(idx + 1, basis.shape[1]):
            if cand in used:
                continue
            if np.allclose(basis[:, cand], np.conj(vec), atol=1e-10):
                partner = cand
                break
        columns.append(vec.real)
        used.add(idx)
        if partner is not None:
            used.add(partner)
        idx += 1
    if len(columns) < dim_principal:
        raise ValueError("Could not build the requested real subspace basis")
    return np.column_stack(columns[:dim_principal])
