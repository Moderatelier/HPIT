# Hierarchical Physics-Integrated Transformer for predicting cumulative thermal deformation of ceramic matrix composites

This repository provides the implementation of the hierarchical physics-integrated transformer (HPIT) for pointwise cumulative thermal-deformation prediction from spatial coordinates and structural/material properties.

Research data, trained weights, manuscript files, figures, generated outputs, notebooks, compiled extensions, and vendored third-party projects are excluded.

## Repository scope

The repository includes the model implementation, configuration, tests, data-format documentation, and a reproducible command-line entry point. CRA-PCN source code is not distributed. Users obtain the fixed upstream commit and apply the included compatibility patch.

No research dataset or pretrained checkpoint is included. Reproducing the quantitative values reported in the article requires research data and trained parameters that are outside this repository.

## Repository structure

```text
hierarchical-physics-integrated-transformer/
├─ CODE/
│  └─ Computation/
│     ├─ Main.py                    # formal command-line entry
│     ├─ Config/
│     │  └─ default.json              # model, data, and training settings
│     └─ hpitd/
│        ├─ attention.py               # slice aggregation and cross-attention
│        ├─ cli.py                     # train, evaluate, and self-test commands
│        ├─ config.py                  # project-relative configuration handling
│        ├─ cra_pcn_backend.py         # external CRA-PCN loader and checks
│        ├─ data.py                    # point-cloud loading and normalisation
│        ├─ evaluate.py                # physical-scale evaluation
│        ├─ io_utils.py                # UTF-8 logs and status files
│        ├─ metrics.py                 # regression metrics
│        ├─ model.py                   # HPIT network definition
│        ├─ point_ops.py               # CUDA extension or PyTorch backend
│        ├─ runtime.py                 # devices, seeds, and parameter counts
│        └─ train.py                   # training and checkpoint workflow
├─ Data/
│  └─ README.md                       # data placement and access policy
├─ Results/
│  ├─ Logs/
│  └─ README.md                       # generated-output structure
├─ docs/
│  ├─ cra_pcn_setup.md                # fixed upstream dependency setup
│  └─ data_format.md                  # `N x 17` column definition
├─ patches/
│  └─ cra-pcn-hpit-compatibility.patch
├─ tests/
│  └─ test_smoke.py                  # data-free CPU smoke tests
├─ CITATION.cff
├─ CONTRIBUTING.md
├─ LICENSE
├─ NOTICE.md
├─ THIRD_PARTY.md
├─ licenses/
│  └─ Transolver-MIT.txt
├─ environment.yml
├─ pyproject.toml
└─ requirements.txt
```

## Environment

The verified development environment used Python 3.12, PyTorch 2.7.1 with CUDA 12.6, and NumPy 2.0.2. PyTorch 2.4 or later is declared as the supported baseline because the network uses native scaled dot-product attention.

Create a clean environment with either Conda or pip:

```bash
conda env create -f environment.yml
conda activate hpitd
```

or

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

The optional `pointnet2_ops` package accelerates farthest-point sampling, grouping, nearest-neighbour search, and interpolation on CUDA devices. A pure-PyTorch implementation is selected on CPU or when the extension is unavailable.

## External CRA-PCN dependency

The CRT implementation is loaded from a user-supplied CRA-PCN checkout. From the repository root, run:

```bash
git clone https://github.com/EasyRy/CRA-PCN.git third_party/CRA-PCN
git -C third_party/CRA-PCN checkout e87d11a5134332366c507f7cf784a03ad13a4d00
git -C third_party/CRA-PCN apply ../../patches/cra-pcn-hpit-compatibility.patch
```

The checkout is excluded by `.gitignore`. Detailed setup, alternative locations, and verification instructions are provided in [`docs/cra_pcn_setup.md`](docs/cra_pcn_setup.md).

## Data preparation

Create `Data/Train/` and `Data/Test/`, then place one `N x 17` NumPy array in each `.npy` file. The default model reads coordinates from columns 0-2, six physical features from columns 10-15, and the positive cumulative thermal-deformation target from column 16. See [`docs/data_format.md`](docs/data_format.md) for the complete mapping.

The repository does not copy validation data into the independent test directory. Training writes a deterministic `split_manifest.json`, which records the training and validation filenames selected from `Data/Train/`.

## Commands

After preparing CRA-PCN, run the data-free CPU self-test:

```bash
python CODE/Computation/Main.py self-test --device cpu
```

Run the optional CUDA-path self-test:

```bash
python CODE/Computation/Main.py self-test --device cuda:0
```

Inspect all resolved paths and active settings:

```bash
python CODE/Computation/Main.py show-config
```

Start training:

```bash
python CODE/Computation/Main.py train \
  --config CODE/Computation/Config/default.json \
  --run-name hpitd_baseline \
  --device cuda:0
```

Evaluate the best checkpoint on `Data/Test/`:

```bash
python CODE/Computation/Main.py evaluate \
  --config CODE/Computation/Config/default.json \
  --run-name hpitd_baseline \
  --device cuda:0
```

Run the unit tests:

```bash
python -m unittest discover -s tests
```

## Outputs

Each run writes to `Results/Experiments/<run-name>/`:

- `Logs/`: UTF-8 training or evaluation logs;
- `run_status.json`: running, completed, or failed state;
- `split_manifest.json`: deterministic training/validation split;
- `normalisation.json`: fitted coordinate, feature, and target statistics;
- `Checkpoints/best_model.pth`: best validation checkpoint after the configured start epoch;
- `Checkpoints/resume_checkpoint.pth`: complete resumable state;
- `Metrics/history.json`: training loss, validation loss, and learning-rate history;
- `training_summary.json`: machine-readable training summary;
- `Evaluation/evaluation_summary.json`: aggregate and per-sample physical-scale metrics.

Prediction arrays are disabled by default. Set `evaluation.save_predictions` to `true` when pointwise arrays are needed.

## Reproducibility notes

- Python, NumPy, PyTorch, and CUDA random generators are seeded.
- Checkpoints include the model, optimiser, scheduler, random states, normalisation statistics, and active configuration.
- Paths are resolved from the repository location and do not depend on the shell's current directory.
- Input and output logs use UTF-8 encoding.
- The default training parameters retain the effective settings of the research implementation: eight layers, hidden width 128, eight attention heads, 64 slices, batch size one, AdamW, Smooth L1 loss, and cosine annealing.
- Training does not copy validation data into the independent test directory. The selected split is recorded in a manifest, and independent test samples remain outside the repository.

## Excluded material

The following materials are intentionally absent:

- paper drafts and reviewer material;
- raw, processed, validation, and test datasets;
- pretrained and resume checkpoints;
- historical logs, metrics, plots, Origin projects, and presentation files;
- Jupyter notebooks used during data preparation;
- local copies of `timm`, `einops`, and PointNet++;
- CRA-PCN source code, which is obtained separately by each user;
- Python caches, compiled CUDA binaries, build directories, and environment-specific paths.

## Citation

The recommended citation is recorded in `CITATION.cff`:

Huanwei Pei, Hang Xu, Ping Liu, Shenshen Liu, Yanxia Du, Chong Wei, and Qi Liu, “Hierarchical Physics-Integrated Transformer for predicting cumulative thermal deformation of ceramic matrix composites,” *Thin-Walled Structures*.

## Licence

Code owned by Hang Xu is distributed under GPL-3.0-only. Transolver-derived portions retain the included MIT notice. CRA-PCN remains an external user-obtained dependency and is not sublicensed by this repository. Other third-party packages remain governed by their respective licences; see `THIRD_PARTY.md`.
