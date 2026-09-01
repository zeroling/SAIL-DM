"""Build one human-readable report from all validation and test artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Mapping

from Core.config import output_root
from Core.io_utils import lock_is_abandoned


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _exclusive_lock(path: Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Serialize report refreshes across parallel worker processes."""

    deadline = time.monotonic() + float(timeout_seconds)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = lock_is_abandoned(path, 300.0)
            except FileNotFoundError:
                continue
            if stale:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for report lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _metric(metrics: Mapping[str, Any] | None, name: str) -> float | None:
    if not metrics or name not in metrics:
        return None
    return float(metrics[name])


def _collect_validation(method_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in method_root.glob(
        "*/ipc_*/condense_seed_*/online_evaluation.json"
    ):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        try:
            dataset = path.parents[2].name
            ipc = int(path.parents[1].name.removeprefix("ipc_"))
            seed = int(path.parent.name.removeprefix("condense_seed_"))
        except ValueError:
            continue
        selection = payload.get("selection") or {}
        selected_iteration = selection.get("iteration")
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        valid_records = [
            record
            for record in records
            if isinstance(record, dict)
            and int(record.get("iteration", -1)) > 0
            and int(record.get("iteration", -1)) % 1000 == 0
        ]
        best_iteration = None
        if valid_records:
            best_iteration = int(
                min(
                    valid_records,
                    key=lambda record: (
                        -float(record.get("accuracy", float("-inf"))),
                        float(record.get("loss", float("inf"))),
                        int(record["iteration"]),
                    ),
                )["iteration"]
            )
        for record in valid_records:
            iteration = int(record["iteration"])
            rows.append(
                {
                    "dataset": dataset,
                    "ipc": ipc,
                    "condensation_seed": seed,
                    "iteration": iteration,
                    "accuracy": float(record["accuracy"]),
                    "loss": float(record["loss"]),
                    "evaluation_epochs": int(record.get("epochs", 0)),
                    "seconds": float(record.get("seconds", 0.0)),
                    "best_so_far_or_final": iteration
                    == int(selected_iteration or best_iteration or -1),
                    "selected_final": iteration
                    == int(selected_iteration or -1),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            int(row["ipc"]),
            int(row["condensation_seed"]),
            int(row["iteration"]),
        ),
    )


def _collect_evaluations(method_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in method_root.glob(
        "*/ipc_*/condense_seed_*/evaluation/*/repeat_*/result.json"
    ):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        evaluation = payload.get(
            "evaluation_metrics", payload.get("test_metrics", {})
        )
        selection = payload.get("model_selection") or {}
        validation = selection.get("best") or {}
        rows.append(
            {
                "dataset": str(payload.get("dataset", path.parents[5].name)),
                "ipc": int(payload.get("ipc", 0)),
                "condensation_seed": int(payload.get("condensation_seed", 0)),
                "architecture": str(payload.get("architecture", path.parents[1].name)),
                "repeat": int(payload.get("repeat", 0)),
                "synthetic_iteration": int(payload.get("synthetic_iteration", -1)),
                "condensation_iterations": int(
                    payload.get("condensation_iterations", -1)
                ),
                "classifier_validation_epoch": (
                    int(
                        validation.get(
                            "completed_update", validation.get("epoch_index")
                        )
                    )
                    if "completed_update" in validation
                    or "epoch_index" in validation
                    else None
                ),
                "classifier_validation_accuracy": _metric(validation, "accuracy"),
                "classifier_validation_loss": _metric(validation, "loss"),
                "test_accuracy": _metric(evaluation, "accuracy"),
                "test_loss": _metric(evaluation, "loss"),
                "test_balanced_accuracy": _metric(
                    evaluation, "balanced_accuracy"
                ),
                "test_macro_f1": _metric(evaluation, "macro_f1"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            int(row["ipc"]),
            int(row["condensation_seed"]),
            str(row["architecture"]),
            int(row["repeat"]),
        ),
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * float(value):.3f}%"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _markdown(
    validation: list[dict[str, Any]], evaluations: list[dict[str, Any]]
) -> str:
    lines = [
        "# 实验结果总览",
        "",
        "> 本文件由训练程序自动更新。第一层每1000轮验证并仅保留一个best合成图；第二层使用验证集选择分类器权重；测试集不参与选择。",
        "",
        "## 每1000轮蒸馏验证结果（仅保存best合成图）",
        "",
    ]
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in validation:
        key = (
            str(row["dataset"]),
            int(row["ipc"]),
            int(row["condensation_seed"]),
        )
        grouped.setdefault(key, []).append(row)
    if not grouped:
        lines.append("尚无验证结果。")
    for (dataset, ipc, seed), rows in grouped.items():
        lines.extend(
            [
                "",
                f"### {dataset} · IPC={ipc} · seed={seed}",
                "",
                "| 迭代 | Val Acc | Val Loss | 快评耗时 | 状态 |",
                "|---:|---:|---:|---:|:---|",
            ]
        )
        for row in rows:
            status = "最终选中" if row["selected_final"] else (
                "当前最优" if row["best_so_far_or_final"] else ""
            )
            lines.append(
                f"| {int(row['iteration'])} | {_percent(row['accuracy'])} | "
                f"{_number(row['loss'])} | {float(row['seconds']):.2f}s | {status} |"
            )
    lines.extend(["", "## 每次最终评估结果", ""])
    if not evaluations:
        lines.append("尚无最终评估结果。")
    else:
        lines.extend(
            [
                "| 数据集 | IPC | 蒸馏种子 | 架构 | Repeat | 合成图迭代 | 分类器Val轮次 | 分类器Val Acc | Test Acc | Balanced Acc | Macro-F1 | Test Loss |",
                "|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in evaluations:
            validation_epoch = row["classifier_validation_epoch"]
            lines.append(
                f"| {row['dataset']} | {row['ipc']} | {row['condensation_seed']} | "
                f"{row['architecture']} | {row['repeat']} | {row['synthetic_iteration']} | "
                f"{validation_epoch if validation_epoch is not None else '-'} | "
                f"{_percent(row['classifier_validation_accuracy'])} | "
                f"{_percent(row['test_accuracy'])} | "
                f"{_percent(row['test_balanced_accuracy'])} | "
                f"{_percent(row['test_macro_f1'])} | {_number(row['test_loss'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def refresh_results_report(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Refresh the method-wide JSON and Markdown reports from source artifacts."""

    method_root = output_root(config).parent
    method_root.mkdir(parents=True, exist_ok=True)
    lock_path = method_root / ".results_overview.lock"
    with _exclusive_lock(lock_path):
        validation = _collect_validation(method_root)
        evaluations = _collect_evaluations(method_root)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_root": str(method_root.resolve()),
            "selection_protocol": {
                "synthetic_dataset": (
                    "validate every 1000 iterations and atomically retain only "
                    "the maximum-validation-accuracy synthetic.pt"
                ),
                "classifier_checkpoint": "maximum validation accuracy every 10 epochs",
                "test_set_used_for_selection": False,
            },
            "validation_snapshots": validation,
            "evaluations": evaluations,
        }
        json_path = method_root / "results_overview.json"
        markdown_path = method_root / "results_overview.md"
        _atomic_write(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        markdown = _markdown(validation, evaluations)
        continuation_path = method_root / "RESULTS_CONTINUATION.md"
        if continuation_path.is_file():
            continuation = continuation_path.read_text(encoding="utf-8").strip()
            if continuation:
                markdown = markdown.rstrip() + "\n\n" + continuation + "\n"
        _atomic_write(markdown_path, markdown)
    return json_path, markdown_path
