"""Regression checks for memory-safe online evaluation; no real data loaded."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from Core.config import load_config
from Net.Condensation.idm_official import build_idm_convnet, diff_augment, ParamDiffAug
from Pipeline.data import TensorImageDataset
from Pipeline.Stages.condense import _online_classifier_step, _quick_evaluate

torch.set_num_threads(1)


def tiny_model():
    return nn.Sequential(
        nn.Conv2d(3, 6, 3, padding=1), nn.GroupNorm(3, 6),
        nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(6, 3),
    )


class OnlineMicrobatchTests(unittest.TestCase):
    def test_gradients_and_momentum_match_full_batch(self):
        torch.manual_seed(12)
        full = tiny_model()
        chunked = copy.deepcopy(full)
        options = dict(lr=0.01, momentum=0.9, weight_decay=0.0005)
        full_optimizer = torch.optim.SGD(full.parameters(), **options)
        chunk_optimizer = torch.optim.SGD(chunked.parameters(), **options)
        config = {"project": {"amp": False}}
        calls = []
        hook = chunked.register_forward_pre_hook(lambda model, args: calls.append(len(args[0])))
        for count in (11, 7):
            images = torch.randn(count, 3, 8, 8)
            labels = torch.arange(count) % 3
            # Independent reference: the previous unsplit SGD update.
            full_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(full(images), labels).backward()
            full_optimizer.step()
            _online_classifier_step(
                config, chunked, chunk_optimizer, images, labels, 4, torch.device("cpu"),
            )
            for reference, actual in zip(full.parameters(), chunked.parameters()):
                torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(
                    full_optimizer.state[reference]["momentum_buffer"],
                    chunk_optimizer.state[actual]["momentum_buffer"], rtol=1e-5, atol=1e-7,
                )
        hook.remove()
        self.assertEqual(calls, [4, 4, 3, 4, 3])

    def test_quick_evaluation_preserves_pe_steps_and_rng(self):
        config = load_config(dataset="pathmnist224")
        config["project"].update(device="cpu", amp=False, pin_memory=False)
        config["data"]["image"]["size"] = [32, 32]
        config["condensation"]["idm"]["network_depth"] = 3
        config["condensation"]["online_evaluation"].update(
            epochs=1, batch_size=7, train_microbatch=3, evaluation_batch_size=2,
        )
        # Three stored canvases -> twelve P&E training views, not three.
        images = torch.randn(3, 3, 32, 32)
        labels = torch.arange(3)
        val = TensorImageDataset(torch.randn(5, 3, 32, 32), torch.arange(5) % 3)
        rng = torch.get_rng_state().clone()
        augmented_counts = []
        real_augment = diff_augment

        def track_augment(images, *args, **kwargs):
            augmented_counts.append(len(images))
            return real_augment(images, *args, **kwargs)

        with patch("Pipeline.Stages.condense.diff_augment", side_effect=track_augment):
            result = _quick_evaluate(
                config, images, labels, val, 3, 1000, 17, 100, 2, torch.device("cpu"),
            )
        self.assertEqual(augmented_counts, [7, 5, 7, 5])
        self.assertEqual(result["training_updates"], 2)
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(result["logical_batch_size"], 7)
        self.assertEqual(result["train_microbatch"], 3)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(0 <= result["accuracy"] <= 1)

    def test_only_224_has_microbatch_override(self):
        high = load_config(dataset="pathmnist224")["condensation"]
        low = load_config(dataset="pathmnist")["condensation"]
        self.assertEqual(high["online_evaluation"]["train_microbatch"], 8)
        self.assertNotIn("train_microbatch", low["online_evaluation"])
        self.assertEqual(high["online_evaluation"]["batch_size"], 128)
        self.assertEqual(high["online_evaluation"]["epochs"], 1000)
        for ipc in (1, 10, 100):
            self.assertEqual(high["idm"]["partition_expansion_by_ipc"][ipc], 2)
            self.assertEqual(high["idm"]["image_learning_rate_by_ipc"][ipc], 0.2)


def cuda_smoke():
    """Exercise the old logical batch128 at native 224 with only 8 activations."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA test requested without CUDA")
    config = load_config(dataset="pathmnist224")
    device = torch.device("cuda")
    model = build_idm_convnet(3, 9, (224, 224), depth=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    calls = []
    hook = model.register_forward_pre_hook(lambda model, args: calls.append(len(args[0])))
    torch.cuda.reset_peak_memory_stats()
    for count in (128, 16):
        images = torch.randn(count, 3, 224, 224, device=device)
        labels = torch.arange(count, device=device) % 9
        images = diff_augment(
            images, config["condensation"]["idm"]["dsa_strategy"], param=ParamDiffAug(),
        )
        _online_classifier_step(config, model, optimizer, images, labels, 8, device)
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise AssertionError("Non-finite parameters after GPU step")
        del images, labels
    torch.cuda.synchronize()
    hook.remove()
    assert calls == [8] * 18, calls
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(), "logical_batches": [128, 16],
        "microbatch": 8, "optimizer_steps": 2,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }, indent=2))


if __name__ == "__main__":
    if "--cuda" in sys.argv:
        cuda_smoke()
    else:
        unittest.main()
