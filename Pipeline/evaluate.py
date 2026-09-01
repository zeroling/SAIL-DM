"""HoP-TM-style final evaluation on four fresh architectures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import Subset

from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_rng_state,
)
from Core.config import output_root
from Core.data import build_loader, unpack_batch
from Core.experiment_runtime import cleanup_memory, cuda_peak_megabytes
from Core.io_utils import atomic_write_json, read_json
from Core.logging_utils import get_stage_logger
from Core.results_report import refresh_results_report
from Core.run_context import autocast_context, resolve_device
from Core.seed import seed_everything
from Net.Classification import build_evaluation_model
from Net.Condensation.idm_official import (
    ParamDiffAug,
    diff_augment,
    partition_and_expand,
)
from Pipeline.data import TensorImageDataset, experiment_bundle


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(
    config: Mapping[str, Any],
    num_classes: int,
    architecture: str,
):
    return build_evaluation_model(
        architecture,
        int(config["data"]["image"]["channels"]),
        int(num_classes),
        config["data"]["image"]["size"],
        convnet_depth=int(
            config["condensation"]["idm"]["network_depth"]
        ),
    )


def _preflight_batch(
    config: Mapping[str, Any],
    num_classes: int,
    requested: int,
    training: bool,
    architecture: str,
) -> tuple[int, float]:
    device = resolve_device(config)
    if device.type != "cuda":
        return int(requested), 0.0
    settings = config["condensation"]
    minimum = max(1, int(settings["memory"].get("retry_minimum", 1)))
    limit = (
        torch.cuda.get_device_properties(device).total_memory
        * float(settings["memory"]["max_reserved_fraction"])
        / (1024**2)
    )
    batch = max(minimum, int(requested))
    failed_upper: int | None = None
    best_batch: int | None = None
    best_peak = 0.0
    channels = int(config["data"]["image"]["channels"])
    height, width = map(int, config["data"]["image"]["size"])
    for _ in range(12):
        model = None
        optimizer = None
        images = None
        logits = None
        loss = None
        targets = None
        fits = False
        peak = 0.0
        try:
            cleanup_memory()
            torch.cuda.reset_peak_memory_stats(device)
            model = _model(
                config, num_classes, architecture
            ).to(device)
            images = torch.rand(
                batch, channels, height, width, device=device
            )
            if training:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                targets = torch.randint(
                    0, num_classes, (batch,), device=device
                )
                images = diff_augment(
                    images,
                    str(settings["idm"]["dsa_strategy"]),
                    seed=0,
                    param=ParamDiffAug(),
                )
                with autocast_context(config, device):
                    logits = model(images)
                    loss = F.cross_entropy(logits.float(), targets)
                loss.backward()
                optimizer.step()
            else:
                with torch.inference_mode(), autocast_context(config, device):
                    logits = model(images)
            peak = cuda_peak_megabytes()
            fits = peak <= limit
        except torch.OutOfMemoryError:
            fits = False
        finally:
            del model, optimizer, images, logits, loss, targets
            cleanup_memory()
        if fits:
            best_batch = int(batch)
            best_peak = float(peak)
            # 请求值本身已经安全，没有更大的候选需要搜索。
            if failed_upper is None:
                return best_batch, best_peak
            gap = failed_upper - best_batch
            if gap <= max(1, best_batch // 32):
                return best_batch, best_peak
            batch = (best_batch + failed_upper) // 2
        else:
            failed_upper = int(batch)
            if best_batch is None:
                if batch <= minimum:
                    raise torch.OutOfMemoryError(
                        f"最小评估 batch={minimum} 仍然显存不足"
                    )
                batch = max(minimum, batch // 2)
            else:
                gap = failed_upper - best_batch
                if gap <= max(1, best_batch // 32):
                    return best_batch, best_peak
                batch = (best_batch + failed_upper) // 2
    if best_batch is None:
        raise torch.OutOfMemoryError("未找到可用的评估 batch")
    return best_batch, best_peak


def _evaluation_epochs(
    evaluation: Mapping[str, Any], ipc: int
) -> int:
    overrides = evaluation.get("epochs_by_ipc", {})
    epochs = overrides.get(str(int(ipc)), overrides.get(int(ipc)))
    return int(evaluation["epochs"] if epochs is None else epochs)


def _evaluation_training_updates(
    evaluation: Mapping[str, Any], ipc: int
) -> int:
    epochs = _evaluation_epochs(evaluation, ipc)
    return epochs + int(bool(evaluation.get("include_epoch_zero_update", False)))


def _validation_is_better(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None
) -> bool:
    """Compare validation checkpoints without consulting the test split."""

    if incumbent is None:
        return True
    candidate_accuracy = float(candidate["accuracy"])
    incumbent_accuracy = float(incumbent["accuracy"])
    if candidate_accuracy != incumbent_accuracy:
        return candidate_accuracy > incumbent_accuracy
    candidate_loss = float(candidate["loss"])
    incumbent_loss = float(incumbent["loss"])
    if candidate_loss != incumbent_loss:
        return candidate_loss < incumbent_loss
    return int(candidate["completed_update"]) < int(
        incumbent["completed_update"]
    )


def _new_evaluation_optimizer(
    model: torch.nn.Module,
    evaluation: Mapping[str, Any],
    learning_rate: float,
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=float(learning_rate),
        momentum=float(evaluation["momentum"]),
        weight_decay=float(evaluation["weight_decay"]),
    )


def _train_epoch(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augmentation: ParamDiffAug,
    collect_metrics: bool = True,
) -> dict[str, float] | None:
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    metric_parts: list[tuple[torch.Tensor, int]] = []
    strategy = str(config["condensation"]["idm"]["dsa_strategy"])
    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        images = diff_augment(
            images,
            strategy,
            param=augmentation,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config, device):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        optimizer.step()
        if collect_metrics:
            amount = int(labels.numel())
            metric_parts.append(
                (
                    torch.stack(
                        (
                            loss.detach().double(),
                            (logits.detach().argmax(1) == labels)
                            .sum()
                            .double(),
                        )
                    ),
                    amount,
                )
            )
            count += amount
    if not collect_metrics:
        return None
    if metric_parts:
        values = torch.stack([part for part, _amount in metric_parts]).cpu().tolist()
        for (loss_value, correct_value), (_part, amount) in zip(
            values, metric_parts, strict=True
        ):
            total_loss += float(loss_value) * int(amount)
            correct += int(correct_value)
    return {
        "loss": total_loss / max(1, count),
        "accuracy": correct / max(1, count),
    }


@torch.inference_mode()
def _test(
    config: Mapping[str, Any],
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    label_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    loss_parts: list[tuple[torch.Tensor, int]] = []
    total_loss = 0.0
    count = 0
    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with autocast_context(config, device):
            logits = model(images)
        amount = int(labels.numel())
        loss_parts.append(
            (F.cross_entropy(logits.float(), labels).detach(), amount)
        )
        count += amount
        label_parts.append(labels.detach())
        prediction_parts.append(logits.argmax(1).detach())
    for loss_value, (_loss, amount) in zip(
        torch.stack([loss for loss, _amount in loss_parts]).cpu().tolist(),
        loss_parts,
        strict=True,
    ):
        total_loss += float(loss_value) * int(amount)
    labels_all = torch.cat(label_parts).cpu().tolist()
    predictions = torch.cat(prediction_parts).cpu().tolist()
    class_ids = list(range(int(num_classes)))
    matrix = confusion_matrix(labels_all, predictions, labels=class_ids)
    recalls = []
    for class_id in class_ids:
        tp = int(matrix[class_id, class_id])
        fn = int(matrix[class_id, :].sum()) - tp
        recalls.append(tp / max(1, tp + fn))
    return {
        "loss": total_loss / max(1, count),
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions,
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def run_evaluation(
    config: Mapping[str, Any],
    ipc: int,
    architecture: str,
    repeat: int = 0,
    condensation_seed: int = 0,
    epoch_limit: int | None = None,
    synthetic_path: str | Path | None = None,
    evaluation_root: str | Path | None = None,
    evaluation_split: str = "test",
    evaluation_interval_epochs: int | None = None,
) -> dict[str, Any]:
    evaluation_split = str(evaluation_split).strip().lower()
    if evaluation_split not in {"val", "test"}:
        raise ValueError(
            "evaluation_split must be either 'val' or 'test', got "
            f"{evaluation_split!r}"
        )
    if (
        evaluation_interval_epochs is not None
        and int(evaluation_interval_epochs) <= 0
    ):
        raise ValueError("evaluation_interval_epochs must be positive")
    root = (
        output_root(config)
        / f"ipc_{int(ipc)}"
        / f"condense_seed_{int(condensation_seed)}"
    )
    synthetic_path = (
        Path(synthetic_path)
        if synthetic_path is not None
        else root / "synthetic.pt"
    )
    if not synthetic_path.is_file():
        raise FileNotFoundError(
            f"Synthetic dataset does not exist: {synthetic_path}"
        )
    payload = torch.load(
        synthetic_path, map_location="cpu", weights_only=False
    )
    synthetic_sha256 = _file_sha256(synthetic_path)
    images, labels = partition_and_expand(
        payload["images"],
        payload["labels"],
        int(payload.get("partition_expansion", 1)),
    )
    train_dataset = TensorImageDataset(images, labels)
    architecture = str(architecture).strip().lower()
    evaluation_settings = config["condensation"]["evaluation"]
    supported = {
        str(value).lower()
        for value in evaluation_settings["architectures"]
    }
    for values in evaluation_settings.get("architectures_by_ipc", {}).values():
        supported.update(str(value).lower() for value in values)
    if architecture not in supported:
        raise ValueError(
            f"评估架构 {architecture!r} 不在配置列表中"
        )
    evaluation_root = (
        Path(evaluation_root)
        if evaluation_root is not None
        else root / "evaluation"
    )
    directory = (
        evaluation_root
        / architecture
        / f"repeat_{int(repeat)}"
    )
    result_path = directory / "result.json"
    settings = config["condensation"]
    evaluation = settings["evaluation"]
    selection_mode = str(
        evaluation.get("model_selection", "best-val")
    ).strip().lower()
    if selection_mode not in {"best-val", "last"}:
        raise ValueError("evaluation.model_selection must be best-val or last")
    use_best_validation = evaluation_split == "test" and selection_mode == "best-val"
    model_selection_interval = (
        int(evaluation.get("model_selection_interval_epochs", 10))
        if use_best_validation
        else 0
    )
    if use_best_validation and model_selection_interval <= 0:
        raise ValueError(
            "evaluation.model_selection_interval_epochs must be positive"
        )
    model_selection_protocol = (
        "best_validation_epoch_v1" if use_best_validation
        else "last_epoch_v1" if evaluation_split == "test"
        else "final_epoch_v1"
    )
    epochs = int(
        epoch_limit
        if epoch_limit is not None
        else _evaluation_epochs(evaluation, ipc)
    )
    training_updates = (
        int(epochs) + int(bool(evaluation.get("include_epoch_zero_update", False)))
        if epoch_limit is not None
        else _evaluation_training_updates(evaluation, ipc)
    )
    synthetic_iteration = int(payload.get("iteration", -1))
    condensation_iterations = int(
        payload.get("optimization_iterations", synthetic_iteration)
    )
    existing = read_json(result_path)
    if (
        existing
        and str(existing.get("evaluation_split", "test")).lower()
        == evaluation_split
        and int(existing.get("epochs", 0)) >= epochs
        and int(existing.get("training_updates", 0)) >= training_updates
        and int(existing.get("synthetic_iteration", -2))
        == synthetic_iteration
        and int(
            existing.get(
                "condensation_iterations",
                existing.get("synthetic_iteration", -2),
            )
        )
        == condensation_iterations
        and (
            str(existing.get("synthetic_sha256", "")) == synthetic_sha256
            or (
                not existing.get("synthetic_sha256")
                and str(existing.get("synthetic_dataset", ""))
                and Path(str(existing["synthetic_dataset"])).resolve()
                == synthetic_path.resolve()
            )
        )
        and str(existing.get("model_selection_protocol", ""))
        == model_selection_protocol
        and int(existing.get("model_selection_interval_epochs", -1))
        == model_selection_interval
    ):
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    bundle = experiment_bundle(config)
    evaluation_dataset = (
        bundle.val if evaluation_split == "val" else bundle.test
    )
    selection_dataset = bundle.val if use_best_validation else None
    if bool(config.get("_smoke", False)):
        limit = int(config.get("_smoke_samples", 64))
        train_dataset = Subset(
            train_dataset, range(min(len(train_dataset), limit))
        )
        evaluation_dataset = Subset(
            evaluation_dataset,
            range(min(len(evaluation_dataset), limit)),
        )
        if selection_dataset is not None:
            selection_dataset = Subset(
                selection_dataset,
                range(min(len(selection_dataset), limit)),
            )
    seed = (
        int(config["project"]["seed"])
        + int(ipc) * 100000
        + int(repeat) * 1009
        + 7000000
    )
    device = resolve_device(config)
    seed_everything(
        seed, bool(config["project"].get("deterministic", False))
    )
    train_batch, train_peak = _preflight_batch(
        config,
        bundle.num_classes,
        min(
            int(settings["memory"]["evaluation_batch"]),
            int(evaluation["batch_size"]),
            len(train_dataset),
        ),
        training=True,
        architecture=architecture,
    )
    evaluation_batch, evaluation_peak = _preflight_batch(
        config,
        bundle.num_classes,
        min(
            int(settings["memory"]["inference_batch"]),
            len(evaluation_dataset),
        ),
        training=False,
        architecture=architecture,
    )
    train_loader = build_loader(
        train_dataset,
        config,
        train=True,
        batch_size=train_batch,
        # TensorImageDataset 已全部在 CPU 内存中；启动 worker
        # 只会在每个小 epoch 引入 IPC/调度开销。
        num_workers=0,
    )
    evaluation_loader = build_loader(
        evaluation_dataset,
        config,
        train=False,
        batch_size=evaluation_batch,
    )
    selection_loader = (
        build_loader(
            selection_dataset,
            config,
            train=False,
            batch_size=min(evaluation_batch, len(selection_dataset)),
        )
        if selection_dataset is not None
        else None
    )
    model = _model(
        config, bundle.num_classes, architecture
    ).to(device)
    initial_learning_rate = float(evaluation["learning_rate"])
    optimizer = _new_evaluation_optimizer(
        model,
        evaluation,
        initial_learning_rate,
    )
    completed = 0
    train_metrics: dict[str, float] = {}
    best_validation: dict[str, Any] | None = None
    best_checkpoint_path = directory / "checkpoint_best_val.pt"
    checkpoint = find_latest_checkpoint(directory)
    if checkpoint is not None:
        state = load_checkpoint(checkpoint, device)
        compatible = (
            int(state.get("synthetic_iteration", -2))
            == synthetic_iteration
            and int(state.get("condensation_iterations", -2))
            == condensation_iterations
            and str(state.get("synthetic_sha256", synthetic_sha256))
            == synthetic_sha256
            and int(state.get("requested_epochs", -1)) == epochs
            and str(state.get("model_selection_protocol", ""))
            == model_selection_protocol
            and int(state.get("model_selection_interval_epochs", -1))
            == model_selection_interval
        )
        best_state = None
        if compatible and use_best_validation:
            if best_checkpoint_path.is_file():
                best_state = load_checkpoint(best_checkpoint_path, device)
                compatible = (
                    int(best_state.get("synthetic_iteration", -2))
                    == synthetic_iteration
                    and int(best_state.get("condensation_iterations", -2))
                    == condensation_iterations
                    and str(
                        best_state.get("synthetic_sha256", synthetic_sha256)
                    )
                    == synthetic_sha256
                    and int(best_state.get("requested_epochs", -1))
                    == epochs
                    and str(best_state.get("model_selection_protocol", ""))
                    == model_selection_protocol
                    and int(
                        best_state.get(
                            "model_selection_interval_epochs", -1
                        )
                    )
                    == model_selection_interval
                )
            else:
                compatible = False
        if compatible:
            model.load_state_dict(
                model_state_from_checkpoint(state), strict=True
            )
            optimizer.load_state_dict(state["optimizer"])
            restore_rng_state(state.get("rng_state"))
            completed = int(state.get("epoch", 0))
            train_metrics = dict(state.get("train_metrics", {}))
            if best_state is not None:
                best_validation = dict(best_state["validation_metrics"])
    logger = get_stage_logger(
        (
            f"evaluation_{config['_runtime']['dataset']}_"
            f"{ipc}_{architecture}_{repeat}"
        ),
        directory,
    )
    curve_path = directory / "evaluation_curve.json"
    curve_payload = read_json(curve_path) or {}
    curve_is_compatible = (
        int(curve_payload.get("synthetic_iteration", -2))
        == synthetic_iteration
        and int(curve_payload.get("condensation_iterations", -2))
        == condensation_iterations
        and str(curve_payload.get("synthetic_sha256", synthetic_sha256))
        == synthetic_sha256
        and int(curve_payload.get("requested_epochs", -1)) == epochs
        and str(curve_payload.get("evaluation_split", "")).lower()
        == evaluation_split
        and int(curve_payload.get("interval_epochs", -1))
        == int(evaluation_interval_epochs or -1)
    )
    curve_points = (
        list(curve_payload.get("points", []))
        if curve_is_compatible
        else []
    )
    recorded_curve_updates = {
        int(point["completed_update"])
        for point in curve_points
        if "completed_update" in point
    }
    training_augmentation = ParamDiffAug()
    for update_index in range(completed, training_updates):
        completed_update = int(update_index + 1)
        checkpoint_due = (
            completed_update
            % int(evaluation["checkpoint_interval_epochs"])
            == 0
            or completed_update == training_updates
        )
        log_due = (
            completed_update == 1
            or completed_update % int(evaluation["log_interval_epochs"]) == 0
            or completed_update == training_updates
        )
        selection_due = (
            selection_loader is not None
            and (
                completed_update % model_selection_interval == 0
                or completed_update == training_updates
            )
        )
        curve_due = (
            evaluation_interval_epochs is not None
            and completed_update % int(evaluation_interval_epochs) == 0
            and completed_update not in recorded_curve_updates
        )
        current_train_metrics = _train_epoch(
            config,
            model,
            train_loader,
            optimizer,
            device,
            training_augmentation,
            collect_metrics=(
                checkpoint_due or log_due or selection_due or curve_due
            ),
        )
        if current_train_metrics is not None:
            train_metrics = current_train_metrics
        # The public HoP-TM evaluator recreates SGD after the update whose
        # zero-based epoch index is epochs//2 + 1, resetting momentum too.
        if int(update_index) == int(epochs) // 2 + 1:
            optimizer = _new_evaluation_optimizer(
                model,
                evaluation,
                initial_learning_rate * 0.1,
            )
        if checkpoint_due:
            atomic_torch_save(
                {
                    "epoch": completed_update,
                    "requested_epochs": int(epochs),
                    "synthetic_iteration": int(synthetic_iteration),
                    "condensation_iterations": int(
                        condensation_iterations
                    ),
                    "synthetic_sha256": synthetic_sha256,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "model_selection_protocol": model_selection_protocol,
                    "model_selection_interval_epochs": (
                        model_selection_interval
                    ),
                    "rng_state": capture_rng_state(),
                },
                directory / "checkpoint_last.pt",
            )
        if log_due:
            logger.info(
                "ipc=%d architecture=%s repeat=%d epoch=%d/%d "
                "loss=%.6f acc=%.4f",
                ipc,
                architecture,
                repeat,
                update_index,
                epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
            )
        if selection_due:
            validation_metrics = _test(
                config,
                model,
                selection_loader,
                device,
                bundle.num_classes,
            )
            validation_candidate = {
                **validation_metrics,
                "completed_update": completed_update,
                "epoch_index": int(update_index),
            }
            selected = _validation_is_better(
                validation_candidate, best_validation
            )
            if selected:
                best_validation = validation_candidate
                atomic_torch_save(
                    {
                        "epoch": completed_update,
                        "requested_epochs": int(epochs),
                        "synthetic_iteration": int(synthetic_iteration),
                        "condensation_iterations": int(
                            condensation_iterations
                        ),
                        "synthetic_sha256": synthetic_sha256,
                        "model": model.state_dict(),
                        "train_metrics": dict(train_metrics),
                        "validation_metrics": dict(best_validation),
                        "model_selection_protocol": (
                            model_selection_protocol
                        ),
                        "model_selection_interval_epochs": (
                            model_selection_interval
                        ),
                    },
                    best_checkpoint_path,
                )
                atomic_write_json(
                    {
                        "protocol": model_selection_protocol,
                        "split": "val",
                        "interval_epochs": model_selection_interval,
                        "best": best_validation,
                    },
                    directory / "model_selection.json",
                )
            logger.info(
                "validation selection ipc=%d epoch=%d/%d loss=%.6f "
                "acc=%.4f selected=%s",
                ipc,
                update_index,
                epochs,
                validation_metrics["loss"],
                validation_metrics["accuracy"],
                selected,
            )
        if curve_due:
            periodic_metrics = _test(
                config,
                model,
                evaluation_loader,
                device,
                bundle.num_classes,
            )
            curve_points.append(
                {
                    "completed_update": completed_update,
                    "epoch_index": int(update_index),
                    "train_metrics": dict(train_metrics),
                    "evaluation_metrics": periodic_metrics,
                }
            )
            recorded_curve_updates.add(completed_update)
            atomic_write_json(
                {
                    "dataset": str(config["_runtime"]["dataset"]),
                    "ipc": int(ipc),
                    "condensation_seed": int(condensation_seed),
                    "architecture": architecture,
                    "repeat": int(repeat),
                    "seed": int(seed),
                    "synthetic_iteration": synthetic_iteration,
                    "condensation_iterations": condensation_iterations,
                    "synthetic_sha256": synthetic_sha256,
                    "requested_epochs": int(epochs),
                    "interval_epochs": int(evaluation_interval_epochs),
                    "evaluation_split": evaluation_split,
                    "points": curve_points,
                },
                curve_path,
            )
            logger.info(
                "periodic %s ipc=%d epoch=%d/%d loss=%.6f "
                "acc=%.4f balanced_acc=%.4f macro_f1=%.4f",
                evaluation_split,
                ipc,
                update_index,
                epochs,
                periodic_metrics["loss"],
                periodic_metrics["accuracy"],
                periodic_metrics["balanced_accuracy"],
                periodic_metrics["macro_f1"],
            )
    final_train_metrics = dict(train_metrics)
    selected_train_metrics = dict(train_metrics)
    if use_best_validation:
        if best_validation is None or not best_checkpoint_path.is_file():
            raise RuntimeError(
                "best-validation classifier checkpoint was not created"
            )
        best_state = load_checkpoint(best_checkpoint_path, device)
        model.load_state_dict(
            model_state_from_checkpoint(best_state), strict=True
        )
        best_validation = dict(best_state["validation_metrics"])
        selected_train_metrics = dict(best_state.get("train_metrics", {}))
    evaluation_metrics = _test(
        config, model, evaluation_loader, device, bundle.num_classes
    )
    result = {
        "dataset": str(config["_runtime"]["dataset"]),
        "ipc": int(ipc),
        "condensation_seed": int(condensation_seed),
        "architecture": architecture,
        "repeat": int(repeat),
        "seed": int(seed),
        "epochs": int(epochs),
        "training_updates": int(training_updates),
        "synthetic_dataset": str(synthetic_path.resolve()),
        "synthetic_sha256": synthetic_sha256,
        "synthetic_iteration": synthetic_iteration,
        "condensation_iterations": condensation_iterations,
        "snapshot_selection": payload.get("selection"),
        "train_metrics": selected_train_metrics,
        "final_epoch_train_metrics": final_train_metrics,
        "model_selection_protocol": model_selection_protocol,
        "model_selection_interval_epochs": model_selection_interval,
        "model_selection": (
            {
                "split": "val",
                "metric": "accuracy",
                "mode": "maximum",
                "tie_breakers": ["lower_loss", "earlier_epoch"],
                "best": best_validation,
            }
            if use_best_validation
            else None
        ),
        "evaluation_split": evaluation_split,
        "evaluation_metrics": evaluation_metrics,
        f"{evaluation_split}_metrics": evaluation_metrics,
        "memory": {
            "train_batch": int(train_batch),
            "evaluation_batch": int(evaluation_batch),
            "train_preflight_peak_mib": float(train_peak),
            "evaluation_preflight_peak_mib": float(evaluation_peak),
            "train_preflight_fraction": float(
                train_peak
                / max(
                    1.0,
                    torch.cuda.get_device_properties(device).total_memory
                    / (1024**2),
                )
            )
            if device.type == "cuda"
            else 0.0,
            "evaluation_preflight_fraction": float(
                evaluation_peak
                / max(
                    1.0,
                    torch.cuda.get_device_properties(device).total_memory
                    / (1024**2),
                )
            )
            if device.type == "cuda"
            else 0.0,
        },
    }
    if evaluation_split == "test":
        # Preserve the public result schema used by existing reports.
        result["test_metrics"] = evaluation_metrics
    atomic_write_json(result, result_path)
    try:
        refresh_results_report(config)
    except Exception as error:
        logger.warning("results overview refresh failed: %s", error)
    del model, optimizer, train_loader, evaluation_loader, selection_loader
    cleanup_memory()
    return result
