from __future__ import annotations

import shutil
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from fig2_3_4 import main as run_fig2
from fig_5_6 import main as run_fig56
from fig_heatmap import main as run_heatmap


def main() -> None:
    output_root = SOURCE_DIR / "results" / "smoke_test"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_fig2([
        "--output-dir",
        str(output_root / "fig2_3_4"),
        "--seeds",
        "10",
        "--steps-ahead",
        "60",
        "--variant",
        "all-requested",
        "--spectral-radius",
        "0.6",
        "--construct-option",
        "random",
    ])
    for w_in_method in ["eigenvec", "random"]:
        run_fig56([
            "--run",
            "--output-dir",
            str(output_root / "fig5_6" / w_in_method),
            "--w-in-method",
            w_in_method,
            "--construct-option",
            "random",
            "--force-update-set",
            "template-pi64",
            "--seeds",
            "10",
            "--smoke",
        ])
    run_heatmap([
        "parameter_heatmap",
        "--output-dir",
        str(output_root / "parameter_heatmap"),
        "--smoke",
        "--max-runs",
        "64",
    ])
    run_heatmap([
        "dysts_heatmap",
        "--output-dir",
        str(output_root / "dysts_heatmap"),
        "--smoke",
        "--seeds",
        "1",
        "--limit-systems",
        "10",
    ])
    print(f"Smoke test outputs: {output_root}")


if __name__ == "__main__":
    main()
