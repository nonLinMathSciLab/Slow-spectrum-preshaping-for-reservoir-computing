from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
METADATA_PATH = DATA_DIR / "chaotic_attractors.json"


def load_metadata(system_name: str) -> dict[str, Any]:
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f).get(system_name, {})


def setup_experiment(model_class, data_gen_params: dict) -> dict:
    """Load or generate a Dysts-aligned trajectory for the CESN experiments.

    Cached trajectories are used when available. On a cache miss, this uses the
    Dysts model metadata: the internal integration step comes from metadata
    ``dt``/``model.dt``, while the returned sampling interval is
    ``period / pts_per_period`` unless ``fix_dt`` is explicitly requested.
    """

    model = model_class()
    system_name = getattr(model, "name", model.__class__.__qualname__)
    metadata = load_metadata(system_name)
    if not metadata:
        raise ValueError(f"No metadata is available for {system_name}")

    true_le = metadata.get("true_lyapunov_spectrum")
    if true_le is None:
        true_le = metadata.get("lyapunov_spectrum_estimated", [metadata["maximum_lyapunov_estimated"]])
    mle = metadata.get("maximum_lyapunov_estimated", true_le[0])

    num_steps = int(data_gen_params["num_steps"])
    requested_fix_dt = data_gen_params.get("fix_dt")
    if requested_fix_dt is not None:
        sample_dt = float(requested_fix_dt)
        pts_per_period = float(model.period) / sample_dt
        cache_name = f"{system_name}_{num_steps}_{sample_dt}.npy"
    else:
        pts_per_period = float(data_gen_params["pts_per_period"])
        sample_dt = float(model.period) / pts_per_period
        cache_name = f"{system_name}_{num_steps}.npy"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / cache_name
    force_regenerate = bool(data_gen_params.get("force_regenerate", False))
    if cache_path.exists() and not force_regenerate:
        all_data = np.load(cache_path)
        source = "cache"
    else:
        integration_dt = _integration_dt(model, metadata, data_gen_params)
        all_data = model.make_trajectory(
            n=num_steps,
            dt=integration_dt,
            resample=True,
            pts_per_period=pts_per_period,
            timescale=data_gen_params.get("timescale", "Fourier"),
            method=data_gen_params.get("method", "Radau"),
            rtol=float(data_gen_params.get("rtol", 1e-12)),
            atol=float(data_gen_params.get("atol", 1e-12)),
        )
        if all_data is None:
            raise RuntimeError(f"{system_name} trajectory generation returned None")
        np.save(cache_path, all_data)
        source = "generated"

    return {
        "system": system_name,
        "true_le": true_le,
        "mle": mle,
        "ALLDATA": all_data,
        "dt": sample_dt,
        "data_source": source,
        "cache_path": str(cache_path),
    }


def _integration_dt(model, metadata: dict[str, Any], data_gen_params: dict) -> float:
    if data_gen_params.get("integration_dt") is not None:
        return float(data_gen_params["integration_dt"])
    if getattr(model, "dt", None) is not None:
        return float(model.dt)
    if metadata.get("dt") is not None:
        return float(metadata["dt"])
    return float(data_gen_params["first_step_dt"])
