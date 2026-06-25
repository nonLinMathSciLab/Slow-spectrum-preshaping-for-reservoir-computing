# Third-Party Notices

## Dysts

This repository includes a minimal local subset of data derived from the
Dysts project:

- Project: Dysts
- Repository: https://github.com/GilpinLab/dysts
- Authors: William Gilpin / GilpinLab and contributors
- License: Apache License 2.0

The Apache License 2.0 text is included at `dysts/LICENSE.md`.

### What Is Included Here

The `dysts/` directory is not a full copy of the Dysts package.  It contains
only the files needed by the paper reproduction scripts:

- a small compatibility wrapper for the Lorenz system and local metadata
  loading;
- `dysts/data/chaotic_attractors.json`, filtered to the autonomous,
  delay-free, three- and four-dimensional systems used by the Dysts heatmap;
- cached `*_30000.npy` trajectories for those selected systems;
- a small `Lorenz_700.npy` cache used by quick checks.

The cached trajectories and metadata are included so that the public
repository can reproduce the Dysts heatmap without relying on data outside the
repository.

### Modifications

This repository uses a reduced, paper-specific layout rather than the full
Dysts API.  The local loader and Lorenz wrapper are simplified for the figure
scripts, and the metadata file is filtered to the subset of systems used by
the paper heatmap.

### Citation

If using the Dysts-based experiments, please also cite the Dysts project:

William Gilpin. "Chaos as an interpretable benchmark for forecasting and
data-driven modelling." Advances in Neural Information Processing Systems
(NeurIPS), 2021.
