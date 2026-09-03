# SAIL-DM

[English](README.md) | [简体中文](README_zh-CN.md)

This repository is the official implementation of **SAIL-DM: Support-Adaptive
Intra-Class Local Distribution Matching for Transferable Medical Dataset
Condensation**. SAIL-DM builds on the strict IDM optimization path and adapts
the granularity of intra-class local distributions to the amount of training
support available in each class, while preserving the prescribed
images-per-class (IPC) storage budget exactly.

This repository contains source code and configurations only. Datasets, synthetic images, checkpoints, logs, caches, and reported experiment outputs are intentionally excluded.

## Method overview

SAIL-DM combines three complementary components:

1. **Support-adaptive partition and mass-aware allocation.** Images from each
   class are resized to `16x16`, flattened, projected onto at most 64 principal
   components, and partitioned by class-conditional K-means. The number of
   statistically supported local components is

   ```text
   K_c = min(IPC, K_max, max(1, floor(N_c / S + 0.5))),
   ```

   where `N_c` is the number of real training images in class `c`, `S=100`
   (the paper's `tau`) is the target support per component, and `K_max=10`.
   A mass-aware lower-bounded residual allocation assigns every retained
   component at least one canvas and always returns exactly the requested IPC.

2. **Center-to-edge stratified P&E initialization.** Samples in each local
   component are ordered by their distance from its pixel-PCA center and split
   into one radial stratum per allocated canvas. Four approximately equally
   spaced samples initialize the canvas's `2x2` partition-and-expansion (P&E)
   views, improving central-to-peripheral coverage without increasing storage.

3. **Local distribution and dispersion matching.** Local feature means are
   weighted by empirical component mass. Radial feature quantiles and
   coordinate-wise standard deviations preserve complementary aspects of
   within-component geometry. A conflict-aware bounded gradient mixer removes
   components that oppose the primary objective and limits the auxiliary
   geometry gradient to 15% of the update budget.

The public runner always executes the complete SAIL-DM method. The command line
does not expose switches that disable individual contributions.

## Results reported in the manuscript

Under the strictly aligned local protocol, SAIL-DM improves mean accuracy over
IDM in all 14 low-resolution ConvNet settings and in 15 of 16 IPC=10
cross-architecture comparisons. At native `224x224` resolution and IPC=100,
SAIL-DM obtains `90.12 +/- 0.58%` on PathMNIST. The manuscript treats published
results obtained with different resolutions or protocols as contextual rather
than controlled comparisons.

## Supported settings

| Key | Dataset / resolution | Classes | Manuscript IPC |
|---|---|---:|---|
| `pathmnist` | PathMNIST, 32x32 | 9 | 1, 5, 10, 100 |
| `bloodmnist` | BloodMNIST, 32x32 | 8 | 1, 10, 50, 100 |
| `dermamnist` | DermaMNIST, 32x32 | 7 | 1, 10, 50 |
| `organamnist` | OrganAMNIST, 32x32 | 11 | 1, 10, 50 |
| `pathmnist224` | PathMNIST, 224x224 | 9 | 100 |

PathMNIST at 32x32 and 224x224 uses the same official samples at two resolutions; they are two evaluation settings rather than independent datasets. All MedMNIST datasets use their official train/validation/test splits.

## Repository layout

```text
SAIL-DM/
|-- configs/                  Dataset and condensation protocols
|-- Core/                     Configuration, data, runtime, I/O, checkpoints
|-- Net/
|   |-- Classification/       ConvNet and transfer architectures
|   `-- Condensation/         IDM, clustering, and SAIL-DM losses
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

The public SAIL-DM protocol is fixed to the `pixel_pca` partition descriptor and
requires no pretrained network.

## Data preparation

Download all registered datasets:

```bash
python download_datasets.py
```

Download selected datasets only:

```bash
python download_datasets.py pathmnist bloodmnist dermamnist organamnist
```

The 224x224 PathMNIST archive is approximately 12.6 GB and is intentionally not needed for the 32x32 experiments:

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
python run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage all --jobs 1 --dry-run
```

## Running SAIL-DM

Run the complete method on one 32x32 setting:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage all --jobs 1 --eval-reports-per-job 1
```

Run the paper-style three condensation seeds and five classifier repetitions:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 2 43 --stage all --repeats 5 --jobs 1 --eval-reports-per-job 1
```

Run multiple 32x32 datasets through the shared queue:

```bash
python -u run_experiment.py --dataset pathmnist bloodmnist dermamnist organamnist --ipc 10 --seed 1 2 43 --stage all --jobs 1 --eval-reports-per-job 1
```

Run the native 224x224 PathMNIST setting on a 16 GB GPU:

```bash
python -u run_experiment.py --dataset pathmnist224 --ipc 1 10 100 --seed 1 --stage all --jobs 1 --eval-reports-per-job 1
```

The 224x224 configuration preserves all four P&E views and the full loss. Activation microbatching changes memory use, not the mathematical objective. Use one job on a 16 GB GPU.

Run condensation and evaluation separately:

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage condense --jobs 1
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage evaluate --evaluation-architectures convnet --repeats 5 --jobs 1 --eval-reports-per-job 1
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
outputs/sail_dm/<dataset>/ipc_<IPC>/condense_seed_<seed>/
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
- The 224x224 configuration uses a deeper ConvNet and smaller microbatches but retains the complete SAIL-DM objective.

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

The publication metadata will be added after acceptance. Until then, cite this
repository using the manuscript title **SAIL-DM: Support-Adaptive Intra-Class
Local Distribution Matching for Transferable Medical Dataset Condensation**
and include the commit hash used in your experiments.

## License

SAIL-DM is released under the [MIT License](LICENSE).
