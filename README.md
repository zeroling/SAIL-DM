# CACDM

[English](README.md) | [简体中文](README_zh-CN.md)

CACDM is a compact, reproducible implementation of **Cluster-Aware Class-Distribution Matching** for medical-image dataset condensation. It extends the strict IDM training path with three cumulative components while retaining a small synthetic set, low storage overhead, and direct ConvNet/cross-architecture evaluation.

This repository contains source code and configurations only. Datasets, synthetic images, checkpoints, logs, caches, and reported experiment outputs are intentionally excluded.

## Method overview

CACDM introduces three cumulative components:

1. **Adaptive class-wise cluster initialization.** Images from each class are resized, flattened, reduced with PCA, and partitioned by class-wise K-means. The number of clusters is

   ```text
   K_c = min(IPC, K_max, max(1, floor(N_c / S + 0.5))),
   ```

   where `N_c` is the number of real training images in class `c`, `S=100` images per cluster by default, and `K_max=10`. Synthetic canvases are allocated in proportion to cluster support. Center-to-edge initialization and partition-and-expansion (P&E) provide diverse starting points without changing the stored IPC.

2. **Cluster-size-weighted feature-mean matching.** Class-conditional feature matching is decomposed over coarse clusters and weighted by their real sample counts. This prevents a small cluster and a dominant cluster from contributing as if they had equal support.

3. **Controlled within-cluster spread matching.** Radial feature quantiles and diagonal feature standard deviations preserve intra-class diversity. If the auxiliary spread gradient conflicts with the main IDM gradient, its conflicting component is projected away and its norm is capped at 15% of the main-gradient norm.

The components are enabled cumulatively:

```text
IDM                  : omit --idea
CACDM-I1             : --idea 1
CACDM-I1+I2          : --idea 1 2
Full CACDM            : --idea 1 2 3
```

## Supported settings

| Key | Dataset / resolution | Classes | Default IPC |
|---|---|---:|---|
| `pathmnist` | PathMNIST, 32x32 | 9 | 1, 5, 10, 100 |
| `bloodmnist` | BloodMNIST, 32x32 | 8 | 1, 10, 50, 100 |
| `dermamnist` | DermaMNIST, 32x32 | 7 | 1, 10, 50 |
| `organamnist` | OrganAMNIST, 32x32 | 11 | 1, 10, 50 |
| `pathmnist224` | PathMNIST+, 224x224 | 9 | 1, 10, 100 |

PathMNIST at 32x32 and 224x224 uses the same official samples at two resolutions; they are two evaluation settings rather than independent datasets. All MedMNIST datasets use their official train/validation/test splits.

## Repository layout

```text
CACDM/
|-- configs/                  Dataset and condensation protocols
|-- Core/                     Configuration, data, runtime, I/O, checkpoints
|-- Net/
|   |-- Classification/       ConvNet and transfer architectures
|   `-- Condensation/         IDM, clustering, and CACDM losses
|-- Pipeline/
|   |-- Stages/condense.py    Condensation and online validation
|   |-- data.py
|   `-- evaluate.py           Best-validation classifier evaluation
|-- tests/                    CPU regression and 224x224 smoke tests
|-- download_datasets.py      Download and integrity-check MedMNIST NPZ files
|-- run_experiment.py         Unified experiment entry point
|-- runtime_compat.py         Windows/Linux runtime compatibility
`-- requirements.txt
```

## Installation

Python 3.10 or 3.11 is recommended. A CUDA-capable PyTorch build is strongly recommended for condensation.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the CUDA build of PyTorch and torchvision that matches your driver by following the official PyTorch selector, then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default `pixel_pca` CACDM protocol requires no pretrained network. The optional ResNet-18/DINOv2 clustering-descriptor ablations require separately supplied, checksum-matched weights under `pretrained/`; those weights are not included.

## Data preparation

Download all registered datasets:

```bash
python download_datasets.py
```

Download selected datasets only:

```bash
python download_datasets.py pathmnist bloodmnist dermamnist organamnist
```

PathMNIST+ at 224x224 is approximately 12.6 GB and is intentionally not needed for the 32x32 experiments:

```bash
python download_datasets.py pathmnist224
```

Validate existing files without network access:

```bash
python download_datasets.py --check-only
```

Files are placed under `data/<DatasetName>/`. The downloader verifies MD5 hashes and official train/validation/test counts. The entire `data/` directory is ignored by Git.

## Quick verification

Both included checks are CPU-safe and do not load the real datasets:

```bash
python -m unittest tests.test_online_microbatch
python tests/smoke_pathmnist224.py
```

Preview a run plan without starting an experiment:

```bash
python run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --idea 1 2 3 --stage all --jobs 1 --dry-run
```

## Running CACDM

Run the complete method on one 32x32 setting:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --idea 1 2 3 --stage all --jobs 1 --eval-reports-per-job 1
```

Run the paper-style three condensation seeds and five classifier repetitions:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 2 43 --idea 1 2 3 --stage all --repeats 5 --jobs 1 --eval-reports-per-job 1
```

Run multiple 32x32 datasets through the shared queue:

```bash
python -u run_experiment.py --dataset pathmnist bloodmnist dermamnist organamnist --ipc 10 --seed 1 2 43 --idea 1 2 3 --stage all --jobs 1 --eval-reports-per-job 1
```

Run the 224x224 PathMNIST+ settings on a 16 GB GPU:

```bash
python -u run_experiment.py --dataset pathmnist224 --ipc 1 10 100 --seed 1 --idea 1 2 3 --stage all --jobs 1 --eval-reports-per-job 1
```

The 224x224 configuration preserves all four P&E views and the full loss. Activation microbatching changes memory use, not the mathematical objective. Use one job on a 16 GB GPU.

Run condensation and evaluation separately:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --idea 1 2 3 --stage condense --jobs 1
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --idea 1 2 3 --stage evaluate --evaluation-architectures convnet --repeats 5 --jobs 1 --eval-reports-per-job 1
```

Re-running the same command resumes compatible checkpoints and skips compatible completed results.

## Evaluation protocol

- Condensation uses only the official training split.
- Online synthetic-set selection uses the official validation split every 1,000 condensation iterations by default.
- The test split never selects synthetic images or classifier checkpoints.
- Final classifiers are selected on validation accuracy, with validation loss and earlier epoch as deterministic tie-breakers.
- Test accuracy, balanced accuracy, macro-F1, per-class recall, and the confusion matrix are written to JSON.
- The default paper protocol uses 20,000 condensation iterations, 1,000 classifier epochs, three condensation seeds (`1`, `2`, `43`), and five classifier repetitions per synthetic set.

## Outputs

Generated files are written below `outputs/` and are excluded from version control. A typical full-method run contains:

```text
outputs/idea_123/<dataset>/ipc_<IPC>/condense_seed_<seed>/
|-- synthetic.pt
|-- summary.json
|-- online_evaluation.json
|-- checkpoint_last.pt
|-- cluster_summary.json
`-- evaluation/<architecture>/repeat_<n>/
    |-- result.json
    |-- model_selection.json
    `-- checkpoint_best_val.pt
```

`synthetic.pt` is the validation-selected synthetic set. `checkpoint_last.pt` supports interruption recovery. Results from different datasets, IPC values, condensation seeds, architectures, and classifier repetitions are intentionally kept separate.

## Reproducibility and resource notes

- Use `--jobs 1` on a single 16 GB GPU. Increasing jobs changes concurrency, not the method, but may cause out-of-memory failures.
- Do not compare runs with different P&E factors as if they had the same effective training-set size; the stored IPC is always reported separately from expanded patches.
- Default clustering is train-only pixel-PCA K-means. Validation and test images are never clustered.
- Adaptive class-wise `K_c` uses half-up rounding and is clamped by class support, IPC, and `max_clusters_per_class`.
- The 224x224 configuration uses a deeper ConvNet and smaller microbatches but retains the three CACDM components.

## Extending to another medical dataset

Add an entry to `configs/datasets.yaml`. The loader supports:

- MedMNIST-compatible NPZ files;
- folder layouts with train/validation/test class directories;
- CSV, JSON, or JSONL manifests;
- standard raster images, NumPy arrays, Torch tensors, and optional DICOM files.

Every dataset must define stable class names, image size/channels, normalization, official or explicitly constructed splits, and supported IPC values.

## Troubleshooting

**CUDA out of memory.** Set `--jobs 1`, keep `--eval-reports-per-job 1`, close unrelated GPU processes, and rely on the configured microbatching. Do not reduce P&E or silently drop synthetic views.

**Windows worker or DLL warning.** Use the unified `run_experiment.py` entry point. `runtime_compat.py` configures the supported Windows runtime before PyTorch/scikit-learn are imported.

**Download blocked or slow.** Download the official NPZ file on another machine, copy it to the expected `data/<DatasetName>/` directory, and run `python download_datasets.py <key> --check-only`.

**Interrupted experiment.** Run the identical command again. A changed method, IPC, seed, or incompatible configuration is deliberately not resumed as if it were the same experiment.

## Citation

The paper citation will be added when the manuscript metadata is public. Until then, cite this repository as **CACDM: Cluster-Aware Class-Distribution Matching for Medical Image Dataset Condensation** and include the commit hash used in your experiments.

## License

CACDM is released under the [MIT License](LICENSE).
