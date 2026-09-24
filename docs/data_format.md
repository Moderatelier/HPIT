# Data format

Each `.npy` file stores one point-cloud sample as an `N x 17` floating-point array.

| Columns | Meaning used by the released code |
|---|---|
| 0-2 | Cartesian coordinates `x`, `y`, and `z` |
| 3-6 | Reserved fields retained for compatibility; ignored by the model |
| 7-9 | Geometry or condition descriptors retained in the source data; ignored by the default model configuration |
| 10-15 | Six structural/material physical features supplied to the network |
| 16 | Positive cumulative thermal-deformation target `dy` |

Coordinates and physical features are standardised. The target is floored at the configured positive value, transformed with the natural logarithm, and then standardised. The fitted statistics are stored inside every training checkpoint and in the experiment metadata.

All files in one mini-batch must contain the same number of points. The default batch size is one, so samples may contain different point counts.
