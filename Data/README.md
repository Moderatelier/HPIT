# Data directory

No research dataset is distributed in this repository.

Place training samples in `Data/Train/` and independent evaluation samples in `Data/Test/`. Each sample must be a NumPy `.npy` array with shape `N x 17`, where `N` is the number of spatial points. The default column mapping is documented in [`docs/data_format.md`](../docs/data_format.md).

The data paths, selected feature columns, target column, split ratio, and logarithmic target floor are controlled by `CODE/Computation/Config/default.json`.
