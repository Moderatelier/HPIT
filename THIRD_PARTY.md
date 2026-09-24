# Third-party components and provenance

The repository does not vendor third-party source code.

- PyTorch: BSD-style licence; tensor operations, automatic differentiation, optimisation, and neural-network layers.
- NumPy: BSD-3-Clause licence; array input/output and numerical preprocessing.
- Transolver: MIT licence, copyright (c) 2024 THUML @ Tsinghua University. The slice-based physics-attention design in `CODE/Computation/hpitd/attention.py` was developed from the Transolver research code. The required notice is preserved in `licenses/Transolver-MIT.txt`. Upstream: <https://github.com/thuml/Transolver>.
- CRA-PCN: the cross-resolution transformer is obtained by each user directly from <https://github.com/EasyRy/CRA-PCN> at commit `e87d11a5134332366c507f7cf784a03ad13a4d00`. This repository contains a compatibility patch and runtime adapter, while the upstream source is excluded. The upstream repository did not expose a software licence during the 2026-09-23 review. Users remain responsible for assessing their permitted use of the separately obtained upstream code.
- `pointnet2_ops` is optional. When installed, CUDA point-cloud kernels are used automatically. The included pure-PyTorch operations are used otherwise.
