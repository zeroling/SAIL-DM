# SAIL-DM

[English](README.md) | [简体中文](README_zh-CN.md)

本仓库是 **SAIL-DM: Support-Adaptive Intra-Class Local Distribution
Matching for Transferable Medical Dataset Condensation（面向可迁移医学数据集
凝练的支持度自适应类内局部分布匹配）** 的官方实现。SAIL-DM 建立在严格 IDM
优化路径之上，根据每个类别拥有的训练样本支持度，自适应确定类内局部分布的表示
粒度，同时严格保持规定的每类图像数（IPC）存储预算。

本仓库只包含源代码和配置。数据集、合成图像、检查点、日志、缓存和已跑出的实验结果均被有意排除。

## 方法概述

SAIL-DM 由三个同时启用的互补组件组成：

1. **支持度自适应划分与质量感知预算分配。** 每类图像先缩放到 `16x16`、
   展平并投影到至多 64 个主成分，再进行类条件 K-means。统计上可支持的局部
   成分数为：

   ```text
   K_c = min(IPC, K_max, max(1, floor(N_c / S + 0.5))),
   ```

   其中 `N_c` 是类别 `c` 的真实训练图像数，`S=100`（论文中的 `tau`）是每个
   局部成分的目标样本支持度，`K_max=10`。质量感知的带下界余量分配规则保证
   每个保留成分至少获得一个画布，并且分配总数始终严格等于指定 IPC。

2. **中心到边缘的分层 P&E 初始化。** 每个局部成分中的样本按照其与 pixel-PCA
   中心的距离排序，并按分配到的画布数切分为径向分层。每层抽取四个近似等距
   样本初始化对应画布的 `2x2` P&E 视图，在不增加存储量的情况下覆盖从中心到
   边缘的形态变化。

3. **局部分布与离散度匹配。** 局部特征均值按照成分的经验质量加权；径向特征
   分位数与逐坐标标准差互补地保持成分内部几何结构。当辅助几何梯度与主目标
   冲突时，先投影去除冲突分量，再将辅助梯度限制在更新预算的 15% 以内。

公开入口始终运行完整 SAIL-DM，命令行不提供关闭单个组件的开关。

## 论文报告结果

在严格对齐的本地协议下，SAIL-DM 在全部 14 个低分辨率 ConvNet 设置中均取得
高于 IDM 的平均准确率，并在 IPC=10 的 16 个跨架构比较中胜出 15 个。在原生
`224x224`、IPC=100 的 PathMNIST 设置上，SAIL-DM 达到 `90.12 +/- 0.58%`。
论文将分辨率或评测协议不同的已发表结果仅作为背景比较，而不作为严格受控对照。

## 支持的设置

| 数据集键 | 数据集/分辨率 | 类别数 | 论文 IPC |
|---|---|---:|---|
| `pathmnist` | PathMNIST, 32x32 | 9 | 1, 5, 10, 100 |
| `bloodmnist` | BloodMNIST, 32x32 | 8 | 1, 10, 50, 100 |
| `dermamnist` | DermaMNIST, 32x32 | 7 | 1, 10, 50 |
| `organamnist` | OrganAMNIST, 32x32 | 11 | 1, 10, 50 |
| `pathmnist224` | PathMNIST, 224x224 | 9 | 100 |

32x32 和 224x224 PathMNIST 使用同一批官方样本，应记为同一数据集的两个分辨率评测设置，不是两个独立数据集。所有 MedMNIST 数据均使用官方 train/validation/test 划分。

## 仓库结构

```text
SAIL-DM/
|-- configs/                  数据集与蒸馏协议
|-- Core/                     配置、数据、运行时、I/O、检查点
|-- Net/
|   |-- Classification/       ConvNet 及跨架构评测网络
|   `-- Condensation/         IDM、局部划分和 SAIL-DM 损失
|-- Pipeline/
|   |-- Stages/condense.py    蒸馏与在线验证
|   |-- data.py
|   `-- evaluate.py           基于最佳验证的分类器评测
|-- tests/                    CPU 回归测试和 224x224 结构烟测
|-- download_datasets.py      下载并完整性检查 MedMNIST NPZ
|-- run_experiment.py         统一实验入口
|-- runtime_compat.py         Windows/Linux 运行时兼容
`-- requirements.txt
```

## 环境安装

推荐 Python 3.10 或 3.11。蒸馏训练强烈建议使用 CUDA 版 PyTorch。

```bash
python -m venv .venv
```

激活环境：

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

先根据显卡驱动和 CUDA 环境，按 PyTorch 官方安装器选择安装 `torch` 和 `torchvision`，然后安装其余依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

公开 SAIL-DM 协议固定使用 `pixel_pca` 划分表示，不需要预训练网络。

## 数据准备

下载所有已注册数据集：

```bash
python download_datasets.py
```

只下载选定的 32x32 数据集：

```bash
python download_datasets.py pathmnist bloodmnist dermamnist organamnist
```

224x224 PathMNIST 数据包约为 12.6 GB，运行 32x32 实验时不需要它：

```bash
python download_datasets.py pathmnist224
```

不访问网络，只检查已有文件：

```bash
python download_datasets.py --check-only
```

数据文件保存到 `data/<DatasetName>/`。下载器会检查 MD5 和官方 train/validation/test 样本数。`data/` 整个目录已被 Git 忽略。

## 快速检查

下面两项检查都可以在 CPU 上运行，且不会加载真实数据：

```bash
python -m unittest tests.test_online_microbatch
python tests/smoke_pathmnist224.py
```

只查看任务计划，不启动实验：

```bash
python run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage all --jobs 1 --dry-run
```

## 运行 SAIL-DM

运行一个 32x32 完整方法实验：

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage all --jobs 1 --eval-reports-per-job 1
```

运行论文协议的 3 个蒸馏种子和每个合成集 5 次分类器评测：

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 2 43 --stage all --repeats 5 --jobs 1 --eval-reports-per-job 1
```

把多个 32x32 数据集加入同一任务队列：

```bash
python -u run_experiment.py --dataset pathmnist bloodmnist dermamnist organamnist --ipc 10 --seed 1 2 43 --stage all --jobs 1 --eval-reports-per-job 1
```

在 16 GB 显卡上运行原生 224x224 PathMNIST：

```bash
python -u run_experiment.py --dataset pathmnist224 --ipc 1 10 100 --seed 1 --stage all --jobs 1 --eval-reports-per-job 1
```

224x224 配置保留全部 4 个 P&E 视图和完整 loss。激活微批处理只改变显存占用，不改变数学目标。16 GB 显卡请使用单 job。

分开执行蒸馏与评测：

```bash
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage condense --jobs 1
python -u run_experiment.py --dataset bloodmnist --ipc 10 --seed 1 --stage evaluate --evaluation-architectures convnet --repeats 5 --jobs 1 --eval-reports-per-job 1
```

对同一条命令重新执行时，程序会续跑兼容的检查点，并跳过已完成且配置一致的结果。

## 评测协议

- 蒸馏阶段只使用官方训练集。
- 默认每 1,000 次蒸馏迭代在官方验证集上选择合成集。
- 测试集不参与合成图像或分类器检查点选择。
- 最终分类器以验证准确率选模；并列时依次用更低验证损失和更早 epoch 确定。
- JSON 会写入测试准确率、平衡准确率、macro-F1、每类召回率和混淆矩阵。
- 默认论文协议为 20,000 次蒸馏迭代、1,000 个分类器 epoch、3 个蒸馏种子（`1/2/43`），每个合成集进行 5 次分类器评测。

## 输出

所有运行产物写入 `outputs/`，并被 Git 忽略。完整方法的典型目录为：

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

`synthetic.pt` 是验证集选中的合成集，`checkpoint_last.pt` 用于中断恢复。不同数据集、IPC、蒸馏种子、评测架构和分类器 repeat 的结果会分开保存。

## 复现与资源注意事项

- 单张 16 GB 显卡建议 `--jobs 1`。增加 job 只改变并发度，不改变方法，但可能导致显存溢出。
- 不能把不同 P&E 因子的结果当作相同有效训练集大小直接对比；存储 IPC 与展开 patch 数必须分开报告。
- 默认聚簇为只使用训练集的 pixel-PCA K-means，验证集与测试集不参与聚簇。
- 自适应类内 `K_c` 使用 half-up 四舍五入，并同时受类样本数、IPC 和 `max_clusters_per_class` 约束。
- 224x224 配置使用更深的 ConvNet 与更小微批，但保留全部三个创新点。

## 扩展到新医学数据集

在 `configs/datasets.yaml` 中添加一个条目即可。数据加载器支持：

- MedMNIST 兼容 NPZ；
- 按 train/validation/test 和类目录组织的文件夹；
- CSV、JSON 或 JSONL 清单；
- 常规栅格图像、NumPy 数组、Torch 张量以及可选 DICOM。

每个数据集需要明确配置稳定的类别名称、图像尺寸/通道、归一化、官方或明确构建的数据划分，以及支持的 IPC。

## 常见问题

**CUDA 显存溢出。** 设置 `--jobs 1`、保留 `--eval-reports-per-job 1`，关闭其他 GPU 任务，并使用配置好的微批处理。不要为了省显存减少 P&E 或静默丢弃合成视图。

**Windows worker/DLL 警告。** 使用统一 `run_experiment.py` 入口。`runtime_compat.py` 会在导入 PyTorch/scikit-learn 前配置支持的 Windows 运行环境。

**国内下载被阻断或机械盘读取慢。** 可在另一台电脑下载官方 NPZ，复制到对应的 `data/<DatasetName>/` 后运行 `python download_datasets.py <key> --check-only`。对 224x224 NPZ，优先放到 SSD。

**实验被中断。** 重新执行完全相同的命令。当方法、IPC、种子或关键配置发生变化时，程序不会把不兼容检查点当作同一实验续跑。

## 引用

论文正式发表后会补充完整出版信息。在此之前，请按论文题目
**SAIL-DM: Support-Adaptive Intra-Class Local Distribution Matching for
Transferable Medical Dataset Condensation** 引用本仓库，并在复现记录中写明所使用的
commit hash。

## 许可证

SAIL-DM 使用 [MIT License](LICENSE) 开源。
