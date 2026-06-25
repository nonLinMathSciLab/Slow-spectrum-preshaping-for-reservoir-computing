from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


class Lorenz:
    """Lorenz-63 system with a minimal DYSTS-compatible interface."""

    period = 1.5008

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 2.667,
        initial_state: tuple[float, float, float] = (-9.7869288, -15.03852, 20.533978),
    ) -> None:
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.initial_state = np.asarray(initial_state, dtype=float)

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            [
                self.sigma * (y - x),
                x * (self.rho - z) - y,
                x * y - self.beta * z,
            ],
            dtype=float,
        )

    def make_trajectory(
        self,
        n: int,
        dt: float = 0.001,
        resample: bool = True,
        pts_per_period: float = 100,
        transient_steps: int = 100,
        **_kwargs,
    ) -> np.ndarray:
        """Generate a Lorenz trajectory sampled at a fixed output interval."""

        if n <= 1:
            raise ValueError("n must be larger than 1")

        sample_dt = self.period / pts_per_period if resample else dt
        total = int(n + transient_steps)
        t_eval = np.arange(total, dtype=float) * sample_dt
        t_span = (0.0, float(t_eval[-1]))

        solution = solve_ivp(
            self.rhs,
            t_span,
            self.initial_state,
            t_eval=t_eval,
            first_step=dt,
            max_step=max(dt, sample_dt / 5.0),
            rtol=1e-9,
            atol=1e-11,
        )
        if not solution.success:
            raise RuntimeError(f"Lorenz integration failed: {solution.message}")

        trajectory = solution.y.T
        return trajectory[transient_steps : transient_steps + n]
