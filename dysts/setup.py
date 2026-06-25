from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
METADATA_PATH = DATA_DIR / "chaotic_attractors.json"


def load_metadata(system_name: str) -> dict:
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get(system_name, {})


def setup_experiment(model_class, data_gen_params: dict) -> dict:
    """Create or load the time series used by the CESN experiments."""

    model = model_class()
    system_name = model.__class__.__qualname__
    metadata = load_metadata(system_name)
    if not metadata:
        raise ValueError(f"No metadata is available for {system_name}")

    true_le = metadata.get("true_lyapunov_spectrum")
    if true_le is None:
        true_le = metadata.get("lyapunov_spectrum_estimated", [metadata["maximum_lyapunov_estimated"]])
    mle = metadata.get("maximum_lyapunov_estimated", true_le[0])

    num_steps = int(data_gen_params["num_steps"])
    first_step_dt = float(data_gen_params["first_step_dt"])
    if data_gen_params.get("fix_dt") is not None:
        dt = float(data_gen_params["fix_dt"])
        pts_per_period = model.period / dt
        cache_name = f"{system_name}_{num_steps}_{dt}.npy"
    else:
        pts_per_period = float(data_gen_params["pts_per_period"])
        dt = model.period / pts_per_period
        cache_name = f"{system_name}_{num_steps}.npy"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / cache_name
    if cache_path.exists():
        all_data = np.load(cache_path)
    else:
        all_data = model.make_trajectory(
            n=num_steps,
            dt=first_step_dt,
            resample=True,
            pts_per_period=pts_per_period,
        )
        np.save(cache_path, all_data)

    return {
        "system": system_name,
        "true_le": true_le,
        "mle": mle,
        "ALLDATA": all_data,
        "dt": dt,
    }
