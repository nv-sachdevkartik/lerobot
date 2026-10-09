# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Checkpoint failures must never silently produce an untrained evaluation policy."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


@pytest.fixture
def tiny_checkpoint(monkeypatch, tmp_path):
    # Replace only the multi-billion-parameter allocation and OpenPI conversion.
    # Exercise the production loader with real safetensors and load_state_dict.
    def tiny_init(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(2))

    monkeypatch.setattr(PI05Policy, "__init__", tiny_init)
    monkeypatch.setattr(PI05Policy, "_convert_openpi_state_dict", lambda self, state: state)
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([3.0, 7.0])}, str(path))
    resolver = Mock(return_value=str(path))
    monkeypatch.setattr("transformers.utils.cached_file", resolver)
    return path, resolver


def test_pretrained_forwards_explicit_hub_options_and_loads_weights(tiny_checkpoint):
    _, resolver = tiny_checkpoint
    options = {
        "cache_dir": Path("custom-cache"),
        "force_download": True,
        "resume_download": True,
        "proxies": {"https": "https://proxy.invalid"},
        "token": False,
        "revision": "a" * 40,
        "local_files_only": True,
    }
    policy = PI05Policy.from_pretrained("example/policy", config=SimpleNamespace(), **options)
    resolver.assert_called_once_with("example/policy", "model.safetensors", **options)
    torch.testing.assert_close(policy.weight, torch.tensor([3.0, 7.0]))


@pytest.mark.parametrize("failure", ["missing", "corrupt", "shape", "conversion"])
def test_pretrained_checkpoint_failure_raises(tiny_checkpoint, monkeypatch, failure):
    path, resolver = tiny_checkpoint
    if failure == "missing":
        resolver.side_effect = OSError("checkpoint unavailable")
    elif failure == "corrupt":
        path.write_bytes(b"not a safetensors file")
    elif failure == "shape":
        save_file({"weight": torch.ones(3)}, str(path))
    else:

        def bad_conversion(self, state):
            raise ValueError("unsupported checkpoint conversion")

        monkeypatch.setattr(PI05Policy, "_convert_openpi_state_dict", bad_conversion)
    with pytest.raises(RuntimeError, match="Could not load PI05 checkpoint"):
        PI05Policy.from_pretrained("example/policy", config=SimpleNamespace())


def test_pretrained_preserves_explicit_non_strict_loading(tiny_checkpoint):
    path, _ = tiny_checkpoint
    save_file({"weight": torch.tensor([3.0, 7.0]), "extra": torch.ones(1)}, str(path))
    policy = PI05Policy.from_pretrained("example/policy", config=SimpleNamespace(), strict=False)
    torch.testing.assert_close(policy.weight, torch.tensor([3.0, 7.0]))
