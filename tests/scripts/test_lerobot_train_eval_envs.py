#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the evaluation environments built during training."""

from typing import Any

import pytest

from lerobot.configs.default import EvalConfig
from lerobot.scripts import lerobot_train


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_make_eval_envs_forwards_trust_remote_code(monkeypatch, trust_remote_code: bool) -> None:
    """`trust_remote_code` must reach `make_env`, otherwise hub environments cannot be built."""
    captured: dict[str, Any] = {}

    def fake_make_env(env_cfg, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(lerobot_train, "make_env", fake_make_env)
    monkeypatch.setattr(lerobot_train, "close_envs", lambda envs: None)

    with lerobot_train._make_eval_envs(object(), EvalConfig(batch_size=1), trust_remote_code):
        pass

    assert captured["trust_remote_code"] is trust_remote_code
