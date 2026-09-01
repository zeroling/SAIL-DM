"""Unified runner for the complete CACDM method.

The public entry point always enables adaptive pixel-PCA clustering with
centre-to-edge P&E initialization, cluster-size-weighted feature-mean
matching, and controlled radial/diagonal-spread matching.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from runtime_compat import configure_runtime

configure_runtime()

import importlib  # noqa: E402

from Core.config import (  # noqa: E402
    list_datasets,
    load_config,
    output_root,
)
from Core.io_utils import atomic_write_json, lock_is_abandoned, read_json  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
ALL_ARCHITECTURES = ("convnet", "resnet18", "vgg11", "alexnet")
_CANCEL_EVENT = threading.Event()
_ACTIVE_CHILDREN: set[subprocess.Popen] = set()
_ACTIVE_CHILDREN_LOCK = threading.Lock()
_CHILD_CPU_THREADS = 1
_CHILD_LOADER_WORKERS = 2


class ExperimentCancelled(RuntimeError):
    """Raised inside worker threads after the parent receives Ctrl+C."""


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(int(process.pid)),
                    "/T",
                    "/F",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def _cancel_all_children() -> None:
    _CANCEL_EVENT.set()
    with _ACTIVE_CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN)
    for process in children:
        _terminate_process_tree(process)


def _handle_parent_termination(signum: int, _frame: Any) -> None:
    """SIGTERM 也必须先回收独立 session 的内部 worker。"""

    _cancel_all_children()
    raise KeyboardInterrupt(f"received signal {signum}")

def _positive_ipcs(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        ipc = int(value)
        if ipc <= 0:
            raise ValueError("IPC must be a positive integer")
        if ipc not in result:
            result.append(ipc)
    return result


def _nonnegative_seeds(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        seed = int(value)
        if seed < 0:
            raise ValueError("Condensation seeds must be non-negative")
        if seed not in result:
            result.append(seed)
    if not result:
        raise ValueError("At least one condensation seed is required")
    return result


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid resolved configuration: {path}")
    return payload


def _evaluation_paths(
    config: dict[str, Any],
    ipc: int,
    condensation_seed: int,
    snapshot_iteration: int | None,
    evaluation_epochs: int | None,
    evaluation_split: str = "test",
) -> tuple[Path | None, Path | None]:
    """Resolve an optional historical snapshot and isolated result root."""

    evaluation_split = str(evaluation_split).strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split!r}")
    selection_mode = str(
        config["condensation"]["evaluation"].get(
            "model_selection", "best-val"
        )
    ).strip().lower()
    if selection_mode not in {"best-val", "last"}:
        raise ValueError(f"Unsupported model selection: {selection_mode!r}")

    run_root = (
        output_root(config)
        / f"ipc_{int(ipc)}"
        / f"condense_seed_{int(condensation_seed)}"
    )
    synthetic_path = None
    if snapshot_iteration is not None:
        synthetic_path = (
            run_root
            / "synthetic_snapshots"
            / f"synthetic_iteration_{int(snapshot_iteration):06d}.pt"
        )
    if snapshot_iteration is None and evaluation_epochs is None:
        return (
            synthetic_path,
            (
                run_root / "evaluation_last_epoch"
                if evaluation_split == "test" and selection_mode == "last"
                else None
                if evaluation_split == "test"
                else run_root / "evaluation_validation"
            ),
        )
    snapshot_label = (
        f"iteration_{int(snapshot_iteration):06d}"
        if snapshot_iteration is not None
        else "selected_snapshot"
    )
    epoch_label = (
        f"epochs_{int(evaluation_epochs)}"
        if evaluation_epochs is not None
        else "protocol_epochs"
    )
    return (
        synthetic_path,
        run_root
        / (
            "evaluation_overrides_last_epoch"
            if evaluation_split == "test" and selection_mode == "last"
            else "evaluation_overrides"
            if evaluation_split == "test"
            else "evaluation_overrides_validation"
        )
        / snapshot_label
        / epoch_label,
    )


def _architectures_for_ipc(
    config: dict[str, Any], ipc: int
) -> list[str]:
    """Resolve the default architecture protocol for one IPC."""

    evaluation = config["condensation"]["evaluation"]
    schedule = evaluation.get("architectures_by_ipc", {})
    selected = schedule.get(int(ipc), schedule.get(str(int(ipc))))
    values = selected if selected is not None else evaluation["architectures"]
    return [str(value).lower() for value in values]


def _worker(args: argparse.Namespace) -> int:
    config = _load_snapshot(args.resolved_config)
    loader_workers = max(
        0, int(os.environ.get("INNOVATION_LOADER_WORKERS", "2"))
    )
    if os.name == "nt":
        # Windows spawn 每个 worker 都会重新导入 PyTorch；本地诊断保持单进程。
        config["project"]["windows_num_workers"] = 0
        effective_loader_workers = 0
    else:
        config["project"]["num_workers"] = loader_workers
        effective_loader_workers = loader_workers
    import torch

    cpu_threads = max(
        1, int(os.environ.get("INNOVATION_CPU_THREADS", "1"))
    )
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(
        "runtime: device=%s cpu_threads=%d interop_threads=%d "
        "loader_workers=%d"
        % (
            "cuda" if torch.cuda.is_available() else "cpu",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            effective_loader_workers,
        ),
        flush=True,
    )
    if args.worker == "condense":
        module = importlib.import_module("Pipeline.Stages.condense")
        module.run_condensation(
            config,
            condensation_seed=args.worker_condensation_seed,
            ipc=args.worker_ipc,
        )
    else:
        from Pipeline.evaluate import run_evaluation

        synthetic_path, evaluation_root = _evaluation_paths(
            config,
            args.worker_ipc,
            args.worker_condensation_seed,
            args.worker_snapshot_iteration,
            args.worker_evaluation_epochs,
            args.worker_evaluation_split,
        )
        run_evaluation(
            config,
            ipc=args.worker_ipc,
            architecture=args.worker_architecture,
            repeat=args.worker_repeat,
            condensation_seed=args.worker_condensation_seed,
            epoch_limit=args.worker_evaluation_epochs,
            synthetic_path=synthetic_path,
            evaluation_root=evaluation_root,
            evaluation_split=args.worker_evaluation_split,
            evaluation_interval_epochs=args.worker_evaluation_interval,
        )
    return 0


def _run_child(
    snapshot: Path,
    task: str,
    ipc: int,
    condensation_seed: int = 0,
    architecture: str = "convnet",
    repeat: int = 0,
    snapshot_iteration: int | None = None,
    evaluation_epochs: int | None = None,
    evaluation_split: str = "test",
    evaluation_interval: int | None = None,
    gpu_id: str | None = None,
) -> None:
    if _CANCEL_EVENT.is_set():
        raise ExperimentCancelled("实验已取消")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        task,
        "--resolved-config",
        str(snapshot),
        "--worker-ipc",
        str(int(ipc)),
        "--worker-condensation-seed",
        str(int(condensation_seed)),
        "--worker-architecture",
        str(architecture),
        "--worker-repeat",
        str(int(repeat)),
        "--worker-evaluation-split",
        str(evaluation_split),
    ]
    if snapshot_iteration is not None:
        command.extend(
            ["--worker-snapshot-iteration", str(int(snapshot_iteration))]
        )
    if evaluation_epochs is not None:
        command.extend(
            ["--worker-evaluation-epochs", str(int(evaluation_epochs))]
        )
    if evaluation_interval is not None:
        command.extend(
            ["--worker-evaluation-interval", str(int(evaluation_interval))]
        )
    child_env = os.environ.copy()
    child_env["INNOVATION_CPU_THREADS"] = str(_CHILD_CPU_THREADS)
    child_env["INNOVATION_LOADER_WORKERS"] = str(_CHILD_LOADER_WORKERS)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        child_env[variable] = str(_CHILD_CPU_THREADS)
    if gpu_id is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpu_tag = f"[gpu{gpu_id}] " if gpu_id is not None else ""
    print("\n>>>", gpu_tag + subprocess.list2cmdline(command), flush=True)
    # 内部 worker 的唯一实时通道是父进程 stdout。在 Linux 上
    # 经历 Ctrl+C/恢复后，父进程继承的 stderr 偶尔可能已失效；
    # 显式合并后，logging 和 traceback 都不再依赖该描述符。
    popen_kwargs: dict[str, Any] = {"stderr": subprocess.STDOUT}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(command, env=child_env, **popen_kwargs)
    with _ACTIVE_CHILDREN_LOCK:
        _ACTIVE_CHILDREN.add(process)
    try:
        while True:
            if _CANCEL_EVENT.is_set():
                _terminate_process_tree(process)
                raise ExperimentCancelled("实验已取消")
            try:
                return_code = process.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if return_code != 0:
            if _CANCEL_EVENT.is_set():
                raise ExperimentCancelled("实验已取消")
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        with _ACTIVE_CHILDREN_LOCK:
            _ACTIVE_CHILDREN.discard(process)


@contextmanager
def _summary_lock(path: Path):
    """串行化多 seed 进程对数据集级 summary.json 的重新聚合。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = lock_is_abandoned(path, 7200.0)
            except FileNotFoundError:
                continue
            if stale:
                path.unlink(missing_ok=True)
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _write_summary_unlocked(
    config: dict[str, Any],
    ipcs: list[int],
    condensation_seeds: list[int],
    snapshot_iteration: int | None = None,
    evaluation_epochs: int | None = None,
    evaluation_split: str = "test",
) -> Path:
    root = output_root(config)
    requested_ipcs = {int(ipc) for ipc in ipcs}
    requested_seeds = {int(seed) for seed in condensation_seeds}
    discovered_ipcs = {
        int(path.name.removeprefix("ipc_"))
        for path in root.glob("ipc_*")
        if path.is_dir() and path.name.removeprefix("ipc_").isdigit()
    }
    ipcs = sorted(discovered_ipcs | requested_ipcs)
    condensation_seeds_by_ipc: dict[int, list[int]] = {}
    for ipc in ipcs:
        discovered = {
            int(path.name.removeprefix("condense_seed_"))
            for path in (root / f"ipc_{int(ipc)}").glob("condense_seed_*")
            if path.is_dir()
            and path.name.removeprefix("condense_seed_").isdigit()
        }
        if ipc in requested_ipcs:
            discovered |= requested_seeds
        condensation_seeds_by_ipc[ipc] = sorted(discovered)
    condensation_seeds = sorted(
        {seed for seeds in condensation_seeds_by_ipc.values() for seed in seeds}
    )
    records: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    architectures_by_ipc: dict[int, list[str]] = {}
    for ipc in ipcs:
        for condensation_seed in condensation_seeds_by_ipc[ipc]:
            _, override_root = _evaluation_paths(
                config,
                ipc,
                condensation_seed,
                snapshot_iteration,
                evaluation_epochs,
                evaluation_split,
            )
            evaluation_root = override_root or (
                root
                / f"ipc_{ipc}"
                / f"condense_seed_{int(condensation_seed)}"
                / "evaluation"
            )
            for result_path in evaluation_root.glob(
                "*/repeat_*/result.json"
            ):
                result = read_json(result_path)
                if isinstance(result, dict):
                    result.setdefault(
                        "condensation_seed", int(condensation_seed)
                    )
                    records.append(result)
        configured_architectures = _architectures_for_ipc(config, int(ipc))
        recorded_architectures = sorted(
            {
                str(item.get("architecture", "")).strip().lower()
                for item in records
                if int(item.get("ipc", -1)) == int(ipc)
                and str(item.get("architecture", "")).strip()
            }
        )
        architectures_by_ipc[ipc] = list(dict.fromkeys(
            [*configured_architectures, *recorded_architectures]
        ))
        for architecture in architectures_by_ipc[ipc]:
            selected = [
                item
                for item in records
                if int(item.get("ipc", -1)) == int(ipc)
                and str(item.get("architecture", "")).lower()
                == str(architecture).lower()
            ]
            accuracies = [
                float(
                    item.get(
                        "evaluation_metrics", item.get("test_metrics", {})
                    )["accuracy"]
                )
                for item in selected
            ]
            if accuracies:
                key = f"ipc_{ipc}/{str(architecture).lower()}"
                seed_means = []
                per_seed: dict[str, dict[str, float | int]] = {}
                for condensation_seed in condensation_seeds_by_ipc[ipc]:
                    seed_accuracies = [
                        float(
                            item.get(
                                "evaluation_metrics",
                                item.get("test_metrics", {}),
                            )["accuracy"]
                        )
                        for item in selected
                        if int(item.get("condensation_seed", -1))
                        == int(condensation_seed)
                    ]
                    if not seed_accuracies:
                        continue
                    seed_mean = statistics.fmean(seed_accuracies)
                    seed_means.append(seed_mean)
                    per_seed[str(int(condensation_seed))] = {
                        "count": len(seed_accuracies),
                        "accuracy_mean": seed_mean,
                        "accuracy_std": (
                            statistics.pstdev(seed_accuracies)
                            if len(seed_accuracies) > 1
                            else 0.0
                        ),
                    }
                aggregates[key] = {
                    "count": len(accuracies),
                    "accuracy_mean": statistics.fmean(accuracies),
                    "accuracy_std": (
                        statistics.pstdev(accuracies)
                        if len(accuracies) > 1
                        else 0.0
                    ),
                    "condensation_seed_count": len(seed_means),
                    "condensation_accuracy_mean": statistics.fmean(
                        seed_means
                    ),
                    "condensation_accuracy_std": (
                        statistics.pstdev(seed_means)
                        if len(seed_means) > 1
                        else 0.0
                    ),
                    "per_condensation_seed": per_seed,
                }
    selection_mode = str(
        config["condensation"]["evaluation"].get(
            "model_selection", "best-val"
        )
    ).strip().lower()
    if snapshot_iteration is None and evaluation_epochs is None:
        path = root / (
            "summary_last_epoch.json"
            if evaluation_split == "test" and selection_mode == "last"
            else "summary.json"
            if evaluation_split == "test"
            else "summary_validation.json"
        )
    else:
        snapshot_label = (
            f"iteration_{int(snapshot_iteration):06d}"
            if snapshot_iteration is not None
            else "selected_snapshot"
        )
        epoch_label = (
            f"epochs_{int(evaluation_epochs)}"
            if evaluation_epochs is not None
            else "protocol_epochs"
        )
        path = (
            root
            / (
                "evaluation_overrides_last_epoch"
                if evaluation_split == "test" and selection_mode == "last"
                else "evaluation_overrides"
                if evaluation_split == "test"
                else "evaluation_overrides_validation"
            )
            / snapshot_label
            / epoch_label
            / "summary.json"
        )
    atomic_write_json(
        {
            "dataset": config["_runtime"]["dataset"],
            "ipcs": ipcs,
            "condensation_runs_per_ipc": max(
                (len(seeds) for seeds in condensation_seeds_by_ipc.values()),
                default=0,
            ),
            "condensation_run_counts_by_ipc": {
                str(int(ipc)): len(condensation_seeds_by_ipc[ipc])
                for ipc in ipcs
            },
            "condensation_seeds": list(condensation_seeds),
            "condensation_seeds_by_ipc": {
                str(int(ipc)): condensation_seeds_by_ipc[ipc]
                for ipc in ipcs
            },
            "evaluation_repeats": int(
                config["condensation"]["evaluation"]["repeats"]
            ),
            "architectures_by_ipc": {
                str(int(ipc)): architectures_by_ipc[ipc]
                for ipc in ipcs
            },
            "snapshot_iteration_override": snapshot_iteration,
            "evaluation_epochs_override": evaluation_epochs,
            "evaluation_split": evaluation_split,
            "aggregates": aggregates,
            "records": records,
        },
        path,
    )
    return path


def _write_summary(
    config: dict[str, Any],
    ipcs: list[int],
    condensation_seeds: list[int],
    snapshot_iteration: int | None = None,
    evaluation_epochs: int | None = None,
    evaluation_split: str = "test",
) -> Path:
    root = output_root(config)
    with _summary_lock(root / ".summary.lock"):
        return _write_summary_unlocked(
            config,
            ipcs,
            condensation_seeds,
            snapshot_iteration,
            evaluation_epochs,
            evaluation_split,
        )


def _visible_gpu_ids() -> list[str]:
    """父进程可见的 GPU 标识列表（尊重外层 CUDA_VISIBLE_DEVICES）。

    返回值直接用于子进程的 CUDA_VISIBLE_DEVICES，因此外层若已限定
    卡子集，子进程会继承同一标识，嵌套绑卡不会错位。
    """

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and str(raw).strip():
        tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
        if tokens:
            return tokens
    try:
        import torch

        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def _preflight_device(allow_cpu: bool) -> None:
    import torch

    if torch.cuda.is_available():
        print(
            f"GPU x{torch.cuda.device_count()}: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )
        return
    if allow_cpu:
        print("警告：CUDA 不可见，将在 CPU 上运行（仅适合冒烟检查）。", flush=True)
        return
    raise RuntimeError(
        "CUDA 不可见。请确认 GPU 驱动与 PyTorch CUDA 版本；"
        "仅做 CPU 冒烟检查时可加 --allow-cpu。"
    )


def main(argv: list[str] | None = None) -> int:
    _CANCEL_EVENT.clear()
    parser = argparse.ArgumentParser(
        description="CACDM 完整方法统一实验入口"
    )
    parser.add_argument(
        "--resume-all",
        action="store_true",
        help=(
            "一键续跑固定任务矩阵：先排 PathMNIST IPC=100 seeds 1/2/43，"
            "再排其余三个数据集的10个配置；所有任务共享 --jobs 并发池"
        ),
    )
    parser.add_argument(
        "--resume-remaining",
        action="store_true",
        help=(
            "合并两台服务器结果后的最小补跑矩阵：Path IPC=100 seed 43；"
            "Blood IPC=50/100；Derma IPC=1/10/50；OrganA IPC=1/10/50"
        ),
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        nargs="+",
        help="一个或多个数据集（共享全局并行任务队列）",
    )
    parser.add_argument(
        "--ipc",
        nargs="+",
        type=int,
        metavar="N",
        help="覆盖数据集的默认 IPC 列表",
    )
    parser.add_argument(
        "--seed",
        "--seeds",
        dest="seeds",
        nargs="+",
        type=int,
        metavar="SEED",
        default=None,
        help="任意多个蒸馏种子（默认 1 2 43）",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "condense", "evaluate"),
        default="all",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="蒸馏迭代轮数覆盖（默认取配置：20000）",
    )
    parser.add_argument(
        "--jobs",
        "--job",
        dest="jobs",
        type=int,
        default=None,
        help=(
            "总并发数，均摊到检测到的各卡（如双卡 --jobs 10 = 每卡 5）；"
            "不传时自动取『GPU 数 × --jobs-per-gpu』"
        ),
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=4,
        help=(
            "每张 GPU 的并发实验数（实测单 job 约 1.7GB 显存，"
            "10GB 卡 4 个安全，24GB 卡可加到 6）"
        ),
    )
    parser.add_argument(
        "--eval-reports-per-job",
        type=int,
        default=5,
        help=(
            "每个基础 job 可同时运行的独立评估 report 数（默认5）；"
            "蒸馏占满一个基础 job，单个评估 report 只占其 1/5 容量"
        ),
    )
    parser.add_argument(
        "--cpu-threads-per-job",
        type=int,
        default=1,
        help="每个实验的 PyTorch/OpenMP/BLAS CPU 线程数（默认 1）",
    )
    parser.add_argument(
        "--loader-workers-per-job",
        type=int,
        default=2,
        help="每个实验在验证/评测阶段的数据加载进程数（默认 2）",
    )
    parser.add_argument(
        "--snapshot-iteration",
        type=int,
        help=(
            "Evaluate a saved synthetic snapshot, for example iteration 20000; "
            "results are isolated from the default evaluation"
        ),
    )
    parser.add_argument(
        "--evaluation-epochs",
        type=int,
        help="Override classifier evaluation epochs, for example 1000",
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="test",
        help="Score the trained classifier on validation or test data",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        help=(
            "Record an evaluation learning-curve point every N classifier "
            "epochs; intended for diagnostics, not test-set model selection"
        ),
    )
    parser.add_argument(
        "--evaluation-model-selection",
        choices=("best-val", "last"),
        default="best-val",
        help=(
            "Choose the classifier checkpoint used for test: best-val "
            "checks validation every configured interval; last uses the "
            "final training epoch without validation checks"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="最终评估重复次数覆盖（协议默认：5）",
    )
    parser.add_argument(
        "--evaluation-repeat",
        type=int,
        help=(
            "只运行指定的最终评估 repeat（0-based）；"
            "供外层全局队列按 report 并行调度"
        ),
    )
    parser.add_argument(
        "--evaluation-architectures",
        nargs="+",
        choices=list(ALL_ARCHITECTURES),
        help="评估架构覆盖（默认四个架构全跑）",
    )
    parser.add_argument(
        "--disable-online-evaluation",
        action="store_true",
        help=(
            "Disable in-training validation/best selection; synthetic.pt is "
            "then written only once from the final iteration."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="覆盖输出根目录",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="单迭代 + 单 epoch 冒烟检查（独立输出目录）",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="允许无 CUDA 运行（仅限诊断）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的实验计划，不启动任何实验",
    )
    parser.add_argument(
        "--worker",
        choices=("condense", "evaluate"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resolved-config", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-ipc", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-condensation-seed",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-architecture",
        default="convnet",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-repeat", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-snapshot-iteration",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-evaluation-epochs",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-evaluation-split",
        choices=("val", "test"),
        default="test",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-evaluation-interval",
        type=int,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)
    if os.name != "nt" and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_parent_termination)
    if args.resume_all and args.resume_remaining:
        parser.error("--resume-all 与 --resume-remaining 不能同时使用")
    if args.resume_all or args.resume_remaining:
        if args.datasets is not None or args.ipc is not None:
            parser.error(
                "一键续跑模式已固定 datasets/ipc，不要再传这些参数"
            )
        if args.seeds is not None:
            parser.error("一键续跑模式已固定各数据集种子，不要再传 --seed")
        if args.smoke:
            parser.error("一键续跑模式不能与 --smoke 同时使用")
        args.datasets = (
            ["pathmnist", "bloodmnist", "dermamnist", "organamnist"]
            if args.resume_all
            else ["organamnist", "bloodmnist", "pathmnist", "dermamnist"]
        )
        args.seeds = [1, 2, 43]
        if args.iterations is None:
            args.iterations = 20000
    elif args.seeds is None:
        args.seeds = [1, 2, 43]
    if args.snapshot_iteration is not None:
        if args.snapshot_iteration <= 0:
            parser.error("--snapshot-iteration must be positive")
        if args.stage == "condense":
            parser.error(
                "--snapshot-iteration is only valid for evaluate/all stages"
            )
    if args.evaluation_repeat is not None:
        if args.evaluation_repeat < 0:
            parser.error("--evaluation-repeat must be non-negative")
        if args.stage == "condense":
            parser.error(
                "--evaluation-repeat is only valid for evaluate/all stages"
            )
    if args.evaluation_epochs is not None:
        if args.evaluation_epochs <= 0:
            parser.error("--evaluation-epochs must be positive")
        if args.stage == "condense":
            parser.error(
                "--evaluation-epochs is only valid for evaluate/all stages"
            )
    if args.evaluation_interval is not None:
        if args.evaluation_interval <= 0:
            parser.error("--evaluation-interval must be positive")
        if args.stage == "condense":
            parser.error(
                "--evaluation-interval is only valid for evaluate/all stages"
            )
    if not args.datasets:
        parser.error("必须通过 --datasets 指定至少一个数据集")
    if args.jobs is not None and args.jobs <= 0:
        parser.error("--jobs 必须为正整数")
    if args.jobs_per_gpu <= 0:
        parser.error("--jobs-per-gpu 必须为正整数")
    if args.eval_reports_per_job <= 0:
        parser.error("--eval-reports-per-job 必须为正整数")
    if args.cpu_threads_per_job <= 0:
        parser.error("--cpu-threads-per-job 必须为正整数")
    if args.loader_workers_per_job < 0:
        parser.error("--loader-workers-per-job 不能为负数")
    global _CHILD_CPU_THREADS, _CHILD_LOADER_WORKERS
    _CHILD_CPU_THREADS = int(args.cpu_threads_per_job)
    _CHILD_LOADER_WORKERS = int(args.loader_workers_per_job)

    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    available = list_datasets(config_path)
    datasets: list[str] = []
    for name in args.datasets:
        key = str(name).strip().lower()
        if key not in available:
            raise ValueError(
                f"配置不含数据集 {name!r}；"
                f"可用：{', '.join(available)}"
            )
        if key not in datasets:
            datasets.append(key)

    print(
        "CACDM 完整协议：自适应聚类初始化 + "
        "簇规模加权均值匹配 + 受控簇内离散度匹配"
    )
    print(f"配置：{config_path}　入口：Pipeline.Stages.condense")
    print(f"数据集（共享全局任务队列）：{', '.join(datasets)}")

    if not args.dry_run and not args.smoke:
        _preflight_device(args.allow_cpu)

    dataset_plans: list[dict[str, Any]] = []
    for dataset in datasets:
        overrides: dict[str, Any] = {}
        if args.output_root is not None:
            overrides.setdefault("project", {})["output_root"] = str(
                args.output_root
            )
        else:
            overrides.setdefault("project", {})["output_root"] = str(
                Path("outputs") / "cacdm"
            )
        if args.iterations is not None:
            overrides.setdefault("condensation", {}).setdefault(
                "idm", {}
            )["iterations"] = int(args.iterations)

        condensation_override = overrides.setdefault("condensation", {})
        condensation_override.setdefault("evaluation", {})[
            "model_selection"
        ] = str(args.evaluation_model_selection)
        if args.disable_online_evaluation:
            condensation_override.setdefault("online_evaluation", {}).update(
                {
                    "interval_iterations": 0,
                    "select_best_by_accuracy": False,
                }
            )
        else:
            # Validate every 1,000 condensation iterations and keep the best
            # validation snapshot. The test set is never consulted.
            condensation_override.setdefault("online_evaluation", {})[
                "select_best_by_accuracy"
            ] = True
        cluster_override = condensation_override.setdefault(
            "cluster_matching", {}
        )
        cluster_override.update(
            {
                "initialization_enabled": True,
                "matching_enabled": True,
                "descriptor_mode": "pixel_pca",
            }
        )
        condensation_override.setdefault(
            "representative_sampling", {}
        )["enabled"] = False
        condensation_override.setdefault("cluster_spread", {}).update(
            {
                "enabled": True,
                "mixer_mode": "full",
                "target_gradient_fraction": 0.15,
            }
        )
        condensation_override["experiment_name"] = "cacdm"

        if args.smoke:
            condensation_override["experiment_name"] = "cacdm_smoke"

        config = load_config(
            config_path, dataset=dataset, overrides=overrides
        )
        if args.resume_remaining:
            ipcs = {
                "pathmnist": [100],
                "bloodmnist": [50, 100],
                "dermamnist": [1, 10, 50],
                "organamnist": [1, 10, 50],
            }[dataset]
        elif args.resume_all and dataset == "pathmnist":
            ipcs = [100]
        else:
            ipcs = _positive_ipcs(
                args.ipc
                if args.ipc is not None
                else list(config["data"]["default_ipcs"])
            )
        if args.smoke and args.ipc is None:
            ipcs = [ipcs[0]]
        if any(int(ipc) > 1000 for ipc in ipcs):
            raise ValueError("最终实验范围只支持 IPC<=1000")
        repeats = int(
            args.repeats
            if args.repeats is not None
            else config["condensation"]["evaluation"]["repeats"]
        )
        if repeats <= 0:
            raise ValueError("--repeats must be positive")
        if (
            args.evaluation_repeat is not None
            and int(args.evaluation_repeat) >= repeats
        ):
            raise ValueError(
                f"--evaluation-repeat={args.evaluation_repeat} exceeds "
                f"configured repeats={repeats}"
            )
        config["condensation"]["evaluation"]["repeats"] = repeats
        architectures = list(
            args.evaluation_architectures
            if args.evaluation_architectures is not None
            else config["condensation"]["evaluation"]["architectures"]
        )
        config["condensation"]["evaluation"]["architectures"] = architectures
        if args.evaluation_architectures is not None:
            config["condensation"]["evaluation"][
                "architectures_by_ipc"
            ] = {}
        condensation_seeds = (
            [43]
            if args.resume_remaining and dataset == "pathmnist"
            else _nonnegative_seeds(list(args.seeds))
        )

        if args.smoke:
            config["_smoke"] = True
            config["_smoke_samples"] = 64
            if args.output_root is None:
                config["project"]["output_root"] = str(
                    Path("outputs")
                    / "smoke"
                    / "cacdm"
                )
            config["condensation"]["idm"].update(
                {
                    "iterations": 1,
                    "model_train_steps": 1,
                    "net_num": 4,
                    "net_num_by_ipc": {
                        int(ipc): 4 for ipc in ipcs
                    },
                    "batch_real_by_ipc": {
                        # Every P&E source target gets a real sample.
                        int(ipc): max(32, 4 * int(ipc))
                        for ipc in ipcs
                    },
                    "checkpoint_interval_iterations": 1,
                    "synthetic_snapshot_interval_iterations": 1,
                    "preview_interval_iterations": 0,
                }
            )
            config["condensation"]["online_evaluation"].update(
                {
                    "interval_iterations": (
                        0 if args.disable_online_evaluation else 1
                    ),
                    "select_best_by_accuracy": bool(
                        not args.disable_online_evaluation
                    ),
                    "epochs": 1,
                }
            )
            config["condensation"]["evaluation"].update(
                {
                    "architectures": ["convnet"],
                    "architectures_by_ipc": {},
                    "repeats": 1,
                    "epochs": 1,
                    "checkpoint_interval_epochs": 1,
                }
            )
            architectures = ["convnet"]
            repeats = 1

        if args.dry_run:
            print(
                f"[dry-run] dataset={dataset} ipc={ipcs} seeds="
                f"{condensation_seeds} stage={args.stage} "
                "architectures_by_ipc={"
                + ", ".join(
                    f"{ipc}:{_architectures_for_ipc(config, ipc)}"
                    for ipc in ipcs
                )
                + f"}} repeats={repeats} "
                f"snapshot_iteration={args.snapshot_iteration} "
                f"evaluation_epochs={args.evaluation_epochs} "
                f"evaluation_split={args.evaluation_split} "
                f"evaluation_repeat={args.evaluation_repeat} "
                f"evaluation_interval={args.evaluation_interval} "
                f"evaluation_model_selection={args.evaluation_model_selection} "
                f"online_evaluation_interval="
                f"{config['condensation']['online_evaluation']['interval_iterations']} "
                f"output={output_root(config)}"
            )
            continue

        root = output_root(config)
        root.mkdir(parents=True, exist_ok=True)
        # 外层统一队列会同时启动同一数据集的多个 IPC/seed。若它们都写
        # dataset 根目录下同一个 resolved_config.json，即使内容相同也会
        # 争用临时文件。按任务身份保存配置，既消除共享写点，也方便追溯。
        ipc_label = "-".join(str(int(value)) for value in ipcs)
        seed_label = "-".join(
            str(int(value)) for value in condensation_seeds
        )
        architecture_label = "-".join(
            str(value).lower() for value in architectures
        )
        repeat_label = (
            "all"
            if args.evaluation_repeat is None
            else str(int(args.evaluation_repeat))
        )
        snapshot = (
            root
            / "resolved_configs"
            / (
                f"stage_{args.stage}__ipc_{ipc_label}__"
                f"seeds_{seed_label}__arch_{architecture_label}__"
                f"repeat_{repeat_label}.json"
            )
        )
        atomic_write_json(config, snapshot)
        print(
            f"dataset={dataset} ipc={ipcs} "
            f"stage={args.stage} output={root}"
        )

        dataset_plans.append(
            {
                "dataset": str(dataset),
                "config": config,
                "snapshot": snapshot,
                "ipcs": list(ipcs),
                "condensation_seeds": list(condensation_seeds),
                "architectures": list(architectures),
                "repeats": int(repeats),
            }
        )

    if args.dry_run:
        print("\ndry-run 结束，未启动任何实验。")
        return 0

    def condensation_progress(
        plan: dict[str, Any], ipc: int, seed: int
    ) -> int:
        run_root = (
            output_root(plan["config"])
            / f"ipc_{int(ipc)}"
            / f"condense_seed_{int(seed)}"
        )
        completed = 0
        summary = read_json(run_root / "summary.json")
        if isinstance(summary, dict):
            completed = int(summary.get("iterations", 0))
        log_path = run_root / "train.log"
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            logged = [
                int(value)
                for value in re.findall(r"\biter=(\d+)/\d+", text)
            ]
            if logged:
                completed = max(completed, logged[-1])
        return completed

    def condensation_finished(
        plan: dict[str, Any], ipc: int, seed: int, target: int
    ) -> bool:
        run_root = (
            output_root(plan["config"])
            / f"ipc_{int(ipc)}"
            / f"condense_seed_{int(seed)}"
        )
        summary = read_json(run_root / "summary.json")
        return (
            isinstance(summary, dict)
            and int(summary.get("iterations", 0)) >= int(target)
            and (run_root / "synthetic.pt").is_file()
        )

    def missing_evaluations(
        plan: dict[str, Any], ipc: int, seed: int
    ) -> list[tuple[str, int]]:
        if args.stage == "condense":
            return []
        config = plan["config"]
        evaluation = config["condensation"]["evaluation"]
        epochs = int(
            args.evaluation_epochs
            if args.evaluation_epochs is not None
            else evaluation.get("epochs_by_ipc", {}).get(
                str(int(ipc)), evaluation["epochs"]
            )
        )
        updates = epochs + int(
            bool(evaluation.get("include_epoch_zero_update", False))
        )
        selection_mode = str(
            evaluation.get("model_selection", "best-val")
        ).strip().lower()
        interval = (
            int(evaluation.get("model_selection_interval_epochs", 10))
            if args.evaluation_split == "test"
            and selection_mode == "best-val"
            else 0
        )
        protocol = (
            "best_validation_epoch_v1"
            if args.evaluation_split == "test"
            and selection_mode == "best-val"
            else "last_epoch_v1"
            if args.evaluation_split == "test"
            else "final_epoch_v1"
        )
        synthetic_path, evaluation_root_override = _evaluation_paths(
            config,
            ipc,
            seed,
            args.snapshot_iteration,
            args.evaluation_epochs,
            args.evaluation_split,
        )
        run_root = (
            output_root(config)
            / f"ipc_{int(ipc)}"
            / f"condense_seed_{int(seed)}"
        )
        expected_synthetic = synthetic_path or run_root / "synthetic.pt"
        expected_synthetic_iteration: int | None = None
        expected_condensation_iterations: int | None = None
        if expected_synthetic.is_file():
            import torch

            synthetic_payload = torch.load(
                expected_synthetic,
                map_location="cpu",
                weights_only=False,
            )
            expected_synthetic_iteration = int(
                synthetic_payload.get("iteration", -1)
            )
            expected_condensation_iterations = int(
                synthetic_payload.get(
                    "optimization_iterations",
                    expected_synthetic_iteration,
                )
            )
            del synthetic_payload
        evaluation_root = evaluation_root_override or run_root / "evaluation"
        missing: list[tuple[str, int]] = []
        selected_repeats = (
            (int(args.evaluation_repeat),)
            if args.evaluation_repeat is not None
            else range(int(plan["repeats"]))
        )
        for architecture in _architectures_for_ipc(config, int(ipc)):
            for repeat in selected_repeats:
                result = read_json(
                    evaluation_root
                    / str(architecture).lower()
                    / f"repeat_{repeat}"
                    / "result.json"
                )
                complete = (
                    isinstance(result, dict)
                    and str(result.get("evaluation_split", "test")).lower()
                    == args.evaluation_split
                    and int(result.get("epochs", 0)) >= epochs
                    and int(result.get("training_updates", 0)) >= updates
                    and str(result.get("model_selection_protocol", ""))
                    == protocol
                    and int(
                        result.get("model_selection_interval_epochs", -1)
                    )
                    == interval
                    and (
                        expected_synthetic_iteration is None
                        or int(result.get("synthetic_iteration", -2))
                        == expected_synthetic_iteration
                    )
                    and (
                        expected_condensation_iterations is None
                        or int(result.get("condensation_iterations", -2))
                        == expected_condensation_iterations
                    )
                    and Path(str(result.get("synthetic_dataset", ""))).resolve()
                    == expected_synthetic.resolve()
                )
                if not complete:
                    missing.append((str(architecture), repeat))
        return missing

    def run_condensation(
        plan: dict[str, Any], ipc: int, condensation_seed: int,
        gpu_id: str | None,
    ) -> None:
        _run_child(
            Path(plan["snapshot"]),
            "condense",
            ipc,
            condensation_seed=condensation_seed,
            gpu_id=gpu_id,
        )

    def run_evaluation_report(
        plan: dict[str, Any],
        ipc: int,
        condensation_seed: int,
        architecture: str,
        repeat: int,
        gpu_id: str | None,
    ) -> None:
        _run_child(
            Path(plan["snapshot"]),
            "evaluate",
            ipc,
            condensation_seed=condensation_seed,
            architecture=architecture,
            repeat=repeat,
            snapshot_iteration=args.snapshot_iteration,
            evaluation_epochs=args.evaluation_epochs,
            evaluation_split=args.evaluation_split,
            evaluation_interval=args.evaluation_interval,
            gpu_id=gpu_id,
        )

    def evaluation_specs(
        plan: dict[str, Any], ipc: int, condensation_seed: int
    ) -> list[dict[str, Any]]:
        """Split every missing architecture/repeat report into one GPU task."""

        return [
            {
                "kind": "evaluate",
                "plan": plan,
                "ipc": int(ipc),
                "seed": int(condensation_seed),
                "architecture": str(architecture),
                "repeat": int(repeat),
            }
            for architecture, repeat in missing_evaluations(
                plan, int(ipc), int(condensation_seed)
            )
        ]

    ordered_tasks = [
        (index, plan, int(ipc), int(seed))
        for index, (plan, ipc, seed) in enumerate(
            (
                (plan, ipc, seed)
                for plan in dataset_plans
                for ipc in sorted(int(value) for value in plan["ipcs"])
                for seed in plan["condensation_seeds"]
            )
        )
    ]
    evaluation_tasks: list[tuple[int, dict[str, Any]]] = []
    partial_condensation_tasks = []
    new_condensation_tasks = []
    for index, plan, ipc, seed in ordered_tasks:
        target = int(plan["config"]["condensation"]["idm"]["iterations"])
        progress = condensation_progress(plan, ipc, seed)
        condensation_complete = condensation_finished(
            plan, ipc, seed, target
        )
        reports = evaluation_specs(plan, ipc, seed)
        task = {
            "kind": "condense",
            "plan": plan,
            "ipc": int(ipc),
            "seed": int(seed),
        }
        if args.stage == "evaluate":
            evaluation_tasks.extend((index, report) for report in reports)
        elif args.stage == "condense":
            if not condensation_complete:
                destination = (
                    partial_condensation_tasks
                    if progress > 0
                    else new_condensation_tasks
                )
                destination.append((progress, index, task))
        elif condensation_complete:
            evaluation_tasks.extend((index, report) for report in reports)
        else:
            destination = (
                partial_condensation_tasks
                if progress > 0
                else new_condensation_tasks
            )
            destination.append((progress, index, task))
    partial_condensation_tasks.sort(key=lambda item: (-item[0], item[1]))
    new_condensation_tasks.sort(key=lambda item: item[1])
    all_task_specs = (
        [task for _index, task in evaluation_tasks]
        + [task for _progress, _index, task in partial_condensation_tasks]
        + [task for _progress, _index, task in new_condensation_tasks]
    )
    if args.resume_all:
        # Stable partition: all three PathMNIST IPC=100 seeds occupy the head
        # of the global queue. Seeds 1/2 resume; seed 43 starts from scratch.
        # Any extra --jobs slots pull work from the remaining datasets.
        all_task_specs.sort(
            key=lambda task: (
                0 if task["plan"]["dataset"] == "pathmnist" else 1
            )
        )
        print(
            "一键续跑队列：PathMNIST IPC=100 seeds 1/2/43 优先；"
            "空余并发槽立即运行其余三个数据集的10个配置",
            flush=True,
        )
    elif args.resume_remaining:
        # The only unfinished PathMNIST task (IPC=100, seed=43) must always
        # occupy the first slot.  Remaining slots continue with the most
        # advanced checkpoints, then new tasks.
        all_task_specs.sort(
            key=lambda task: (
                0 if task["plan"]["dataset"] == "pathmnist" else 1
            )
        )
        print(
            "最小补跑队列：PathMNIST IPC=100 seed 43 固定占第一个槽；"
            "其余槽优先恢复进度最高的断点，且不调度已完成配置",
            flush=True,
        )
    print(
        "任务优先级：缺失评估 report > 未完成蒸馏（迭代轮次从高到低）"
        " > 全新蒸馏；每个架构/repeat 独立并行，完整结果自动跳过",
        flush=True,
    )
    visible_gpus = _visible_gpu_ids()
    gpu_count = len(visible_gpus)
    slots_per_gpu = int(args.jobs_per_gpu)
    if gpu_count > 0:
        if args.jobs is not None:
            total_jobs = int(args.jobs)
            busiest = (total_jobs + gpu_count - 1) // gpu_count
            if busiest > slots_per_gpu:
                print(
                    f"注意：--jobs={total_jobs} 折合最忙的卡跑 {busiest} 个并发，"
                    f"高于默认安全值 {slots_per_gpu}/卡，请确认显存余量",
                    flush=True,
                )
        else:
            total_jobs = gpu_count * slots_per_gpu
        pool_ids: list[str | None] = [
            visible_gpus[i % gpu_count] for i in range(total_jobs)
        ]
    else:
        total_jobs = int(args.jobs) if args.jobs is not None else 1
        pool_ids = [None] * total_jobs
    report_factor = int(args.eval_reports_per_job)
    total_capacity_units = int(total_jobs) * report_factor
    gpu_capacity_units = {
        gpu_id: pool_ids.count(gpu_id) * report_factor
        for gpu_id in visible_gpus
    }
    gpu_used_units = {gpu_id: 0 for gpu_id in visible_gpus}
    used_capacity_units = 0
    initial_condensations = sum(
        task["kind"] == "condense" for task in all_task_specs
    )
    initial_reports = sum(
        task["kind"] == "evaluate" for task in all_task_specs
    )
    print(
        f"GPU 调度：可见卡 {visible_gpus if visible_gpus else '无'}，"
        f"每卡并发 {slots_per_gpu}，总并发 {total_jobs}；"
        f"每 job CPU 线程 {_CHILD_CPU_THREADS}、loader 进程 "
        f"{_CHILD_LOADER_WORKERS}；启动 {initial_condensations} 个蒸馏任务"
        f" + {initial_reports} 个现成评估 report；"
        f"蒸馏并发上限 {total_jobs}，评估并发上限 "
        f"{total_jobs}×{report_factor}={total_capacity_units}；"
        "新蒸馏完成后立即拆分并行评估",
        flush=True,
    )
    executor = ThreadPoolExecutor(max_workers=total_capacity_units)
    pending = list(all_task_specs)
    futures: dict[Any, dict[str, Any]] = {}

    def task_units(task: dict[str, Any]) -> int:
        return report_factor if task["kind"] == "condense" else 1

    def reserve_task(task: dict[str, Any]) -> tuple[bool, str | None, int]:
        nonlocal used_capacity_units
        units = task_units(task)
        if used_capacity_units + units > total_capacity_units:
            return False, None, units
        if not visible_gpus:
            used_capacity_units += units
            return True, None, units
        candidates = [
            gpu_id
            for gpu_id in visible_gpus
            if gpu_used_units[gpu_id] + units <= gpu_capacity_units[gpu_id]
        ]
        if not candidates:
            return False, None, units
        gpu_id = min(
            candidates,
            key=lambda item: (
                gpu_used_units[item] / max(1, gpu_capacity_units[item]),
                visible_gpus.index(item),
            ),
        )
        gpu_used_units[gpu_id] += units
        used_capacity_units += units
        return True, gpu_id, units

    def release_task(task: dict[str, Any]) -> None:
        nonlocal used_capacity_units
        units = int(task["_capacity_units"])
        gpu_id = task["_gpu_id"]
        used_capacity_units -= units
        if gpu_id is not None:
            gpu_used_units[gpu_id] -= units

    def submit_available() -> None:
        while pending and not _CANCEL_EVENT.is_set():
            selected_index = None
            gpu_id = None
            units = 0
            for index, candidate in enumerate(pending):
                reserved, candidate_gpu, candidate_units = reserve_task(candidate)
                if reserved:
                    selected_index = index
                    gpu_id = candidate_gpu
                    units = candidate_units
                    break
            if selected_index is None:
                break
            task = pending.pop(selected_index)
            task["_gpu_id"] = gpu_id
            task["_capacity_units"] = units
            if task["kind"] == "condense":
                future = executor.submit(
                    run_condensation,
                    task["plan"],
                    task["ipc"],
                    task["seed"],
                    gpu_id,
                )
            else:
                future = executor.submit(
                    run_evaluation_report,
                    task["plan"],
                    task["ipc"],
                    task["seed"],
                    task["architecture"],
                    task["repeat"],
                    gpu_id,
                )
            futures[future] = task

    try:
        submit_available()
        while futures:
            completed_futures, _unfinished = wait(
                tuple(futures), return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                task = futures.pop(future)
                release_task(task)
                try:
                    future.result()
                except CancelledError:
                    if not _CANCEL_EVENT.is_set():
                        raise
                except ExperimentCancelled:
                    if not _CANCEL_EVENT.is_set():
                        raise
                except Exception as error:
                    suffix = (
                        f" architecture={task['architecture']}"
                        f" repeat={task['repeat']}"
                        if task["kind"] == "evaluate"
                        else ""
                    )
                    raise RuntimeError(
                        f"实验失败：stage={task['kind']} "
                        f"dataset={task['plan']['dataset']} "
                        f"ipc={task['ipc']} seed={task['seed']}{suffix}"
                    ) from error
                if task["kind"] == "condense" and args.stage == "all":
                    reports = evaluation_specs(
                        task["plan"], task["ipc"], task["seed"]
                    )
                    # Newly unblocked reports have priority over fresh
                    # condensation, while preserving architecture/repeat order.
                    pending[0:0] = reports
                    if reports:
                        print(
                            "蒸馏完成，已加入独立评估 report："
                            f"dataset={task['plan']['dataset']} "
                            f"ipc={task['ipc']} seed={task['seed']} "
                            f"reports={len(reports)}",
                            flush=True,
                        )
            submit_available()
    except KeyboardInterrupt:
        print(
            "\n收到 Ctrl+C：正在取消全部排队任务并终止活跃子进程……",
            flush=True,
        )
        _cancel_all_children()
        pending.clear()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        print("全部实验进程已停止；再次运行同一命令可从断点恢复。", flush=True)
        return 130
    except Exception:
        _cancel_all_children()
        pending.clear()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    summaries: list[Path] = []
    for plan in dataset_plans:
        summary = _write_summary(
            plan["config"],
            plan["ipcs"],
            plan["condensation_seeds"],
            snapshot_iteration=args.snapshot_iteration,
            evaluation_epochs=args.evaluation_epochs,
            evaluation_split=args.evaluation_split,
        )
        summaries.append(summary)
        print(f"\n数据集 {plan['dataset']} 完成：{summary}")

    for summary in summaries:
        print(f"Completed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
