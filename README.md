# Slow-Spectrum-Preshaped Reservoir Computing

An implementation of "[Attractor reconstruction in attracting subspaces: Slow-spectrum preshaping for reservoir computing under partial observation](
https://doi.org/10.48550/arXiv.2606.24303)" proposed by Satoshi Oishi, Hiroshi Yamashita, Hideyuki Suzuki, Sho Shirasaka.

## Method Summary

The implementation has three main steps.  Some class names are historical
implementation identifiers; in this public code, read them simply as ESN
components and ESN dynamics.

1. Build a reservoir matrix `W_res` and optionally preshape selected
   eigenvalues.  This corresponds to Algorithm 1, slow-spectrum preshaping.
   The operation changes specified eigenvalues while preserving the associated
   left/right eigenspaces as much as possible.

   Implementation:

   - `CreateCESN_Component.create_reservoir_matrix(...)` creates and scales
     `W_res`.
   - `CreateCESN_Component.modify_eigenstructure(...)` applies the eigenvalue
     replacements.
   - `CreateCESN_Component._scale_matrix(...)` computes the effective matrix
     locally as `essential` and stores its eigenpairs for later use.

   Main variables:

   - `w_res_method`: reservoir initialization, e.g. `symmetry`.
   - `spectral_radius`: target spectral radius of `W_res`.
   - `W_res_force_updates`: selected eigenvalue replacements.  `False` means
     no slow-spectrum preshaping.
   - `alpha`: leaking rate, used in
     `W_res_eff = (1 - alpha) I + alpha W_res`.

2. Construct the input matrix `W_in` from an invariant subspace of
   `W_res_eff = (1 - alpha) I + alpha W_res`.  This corresponds to Algorithm
   2, deterministic input-layer initialization.  Real eigenvectors are used
   directly; complex conjugate eigenvectors are converted to real and imaginary
   basis directions.  The basis is orthonormalized, mixed by a coefficient
   matrix, column-normalized, and scaled.

   Implementation:

   - `CreateCESN_Component.create_input_weight_matrix(...)` is the public
     entry point for `W_in`.
   - `CreateCESN_Component._select_invariant_indices(...)` selects the
     eigenvector groups.
   - `CreateCESN_Component._construct_invariant_subspace(...)` turns selected
     real or conjugate-pair eigenvectors into a real basis.
   - `CreateCESN_Component._construct_w_in_from_subspace(...)` maps that basis
     to the final input matrix.

   Main variables:

   - `w_in_method`: `eigenvec` for invariant-subspace input, or `random` for a
     standard random input layer.
   - `use_eigenvecs_method_or_list`: selection rule for the invariant subspace.
     The paper uses `high`.
   - `invariant_dim`: dimension `D'` of the designed input subspace.
   - `construct_option`: how the selected basis is converted into `W_in`.  The
     paper-scale runs use `random`, i.e. QR basis times Gaussian coefficients.
   - `input_scaling_ratio`: scaling factor before the final
     `sqrt(dim_reservoir / dim_inputs)` normalization used by the figure
     runners.

3. Drive the reservoir with the observed time series, train only the output
   matrix `W_out` by least squares, and evaluate the autonomous closed-loop
   system where `W_close = W_res + W_in W_out`.

   Implementation:

   - `ComplexEchoStateNetwork.listen(...)` runs the listening phase.
   - `ComplexEchoStateNetwork.training_lstsq(...)` fits `W_out`.
   - `ComplexEchoStateNetwork.average_prediction_horizon_numba(...)`
     computes VPT over multiple prediction start points.
   - `ComplexEchoStateNetwork.feedback_predict_start(...)` generates a
     closed-loop trajectory for plotting.

   Main variables:

   - `WARMUP_STEPS`: number of initial driven steps discarded as washout.
   - `TRAINING_TIME_STEPS`: number of reservoir states used for fitting
     `W_out`.
   - `PRED_START_POINT`, `PRED_START_POINT_END`, `OVER_NUM`: prediction start
     points used for VPT averaging.
   - `PREDICTION_STEPS`: maximum closed-loop prediction length.
   - `regularization`: ridge parameter for the output regression.

## Core Classes

### `CreateCESN_Component`

Defined in `crc.py`.  This class creates the fixed ESN components:

- `W_res`: reservoir transition matrix.
- `W_in`: input matrix.
- `bias`: reservoir bias vector.

Important methods:

- `create_reservoir_matrix(...)` creates a symmetric, dense random, or
  Erdos-Renyi reservoir and scales it to the requested spectral radius.
- `modify_eigenstructure(...)` implements Algorithm 1.  It replaces selected
  real eigenvalues or conjugate-pair blocks using the corresponding right and
  left eigenvectors.
- `create_input_weight_matrix(...)` implements Algorithm 2 when
  `method="eigenvec"`.  It selects an invariant subspace, converts complex
  eigendirections into a real basis, and builds `W_in`.
- `create_all_components(...)` is the usual entry point used by the experiment
  runners.

For standard RC baselines, use `w_in_method="random"` and
`W_res_force_updates=False`.  For the proposed construction, use
`w_in_method="eigenvec"` together with non-false `W_res_force_updates`.

### `ComplexEchoStateNetwork`

Defined in `crc.py`.  This class runs the ESN after the components have been
created.

The driven update is

```text
r_{t+1} = alpha * tanh(W_in u_t + W_res r_t + b)
          + (1 - alpha) * r_t .
```

After training, the closed-loop prediction uses the model output as the next
input.  The effective closed-loop matrix tracked in the analysis is
`W_close = W_res + W_in W_out`.

Important methods:

- `listen(...)`: drives the reservoir with the observed signal and records the
  reservoir state sequence after warm-up.
- `training_lstsq(...)`: fits `W_out` by least squares with optional ridge
  regularization.
- `average_prediction_horizon_numba(...)`: evaluates valid prediction time
  over multiple prediction start points.
- `feedback_predict_start(...)`: generates one closed-loop prediction
  trajectory from a selected start point.

The public implementation supports the no-delay settings used in the paper
figures.  It uses NumPy for ESN training and prediction.

### `CESNComprehensiveSweepV3`

Defined in `grid_crc.py`.  This is the compact experiment runner.  It loads and
normalizes a dynamical-system time series, applies the requested partial
observation, creates the ESN components, trains the network, evaluates VPT,
computes Lyapunov diagnostics, and writes figure-ready result records.

### Lyapunov Analysis

`crc_analyzer.py` contains the reduced-QR Lyapunov-spectrum routines used for
the paper diagnostics.  When CuPy is installed, the figure scripts use the GPU
path for this Lyapunov calculation only; the ESN training and prediction remain
NumPy-based.

## File Map

- `crc.py`: ESN component construction and closed-loop ESN dynamics.
- `grid_crc.py`: single-experiment runner connecting data, model, training,
  prediction, and diagnostics.
- `crc_analyzer.py`: Lyapunov-spectrum and subspace diagnostics.
- `crc_plots.py`: plotting helpers for eigenvalue distributions, spectra, and
  heatmaps.
- `fig2_3_4.py`: Lorenz partial-observation demonstration.
- `fig_5_6.py`: conditional and closed-loop Lyapunov-spectrum sweeps.
- `fig_heatmap.py`: Lorenz parameter heatmap and Dysts scalar-observation
  heatmap.
- `configs/`: fixed configuration records for the public runs.
- `dysts/`: minimal Dysts-derived metadata and cached trajectories used by
  the paper heatmap.
- `THIRD_PARTY_NOTICES.md`: attribution and license notes for Dysts-derived
  files.

## Environment

CPU-only setup:

```bash
python -m pip install -r requirements.txt
```

Optional GPU support for Lyapunov-spectrum calculations:

```bash
python -m pip install -r requirements.txt -r requirements-gpu.txt
```

The provided devcontainer installs the GPU-capable environment.  If a GPU is
not available, remove the `runArgs` section from `.devcontainer/devcontainer.json`.

## Quick Check

```bash
python tests/smoke_test.py
```

The smoke test uses one seed and reduced parameter ranges.  It checks that the
main code paths run; it is not intended to reproduce paper-scale statistics.

## Paper-Scale Entry Points

The following scripts run the fixed public experiments.  They can take many
hours.

```bash
python fig2_3_4.py --paper
python fig_5_6.py --run --output-dir results/fig_5_6_construct_random/eigenvec --w-in-method eigenvec --construct-option random --force-update-set template-pi64 --seeds 20
python fig_5_6.py --run --output-dir results/fig_5_6_construct_random/random --w-in-method random --construct-option random --force-update-set template-pi64 --seeds 20
python fig_heatmap.py parameter_heatmap --output-dir results/parameter_heatmap_construct_random --lyapunov-error-vmax 10
python fig_heatmap.py dysts_heatmap --output-dir results/dysts_heatmap --full --seeds 20 --max-observed-dims 4 --vmin 0 --vmax 10
```

The bundled `dysts/data` directory contains the selected Dysts metadata and
cached trajectories needed for the paper heatmap.  To use a different cached
Dysts data directory, pass it explicitly:

```bash
python fig_heatmap.py dysts_heatmap --dysts-data-dir path/to/dysts/data
```

Heatmap scripts also support `--plot-only` for regenerating figures from
existing per-seed CSV files without rerunning the experiments.

## Third-Party Data

The `dysts/` directory includes a minimal subset of data derived from the
Dysts project by William Gilpin / GilpinLab.  The full Dysts package is not
vendored here; only the metadata and cached trajectories needed for this
paper's heatmap are included.  See `THIRD_PARTY_NOTICES.md` and
`dysts/LICENSE.md` for attribution and license details.
