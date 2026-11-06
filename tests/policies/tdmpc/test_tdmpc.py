#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Camera-key regressions adapted from upstream PR #4741 at bc3f17f65767f9cfaf6684ff46bdd7b1c42c0280."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.random_utils import set_seed
from tests.utils import DEVICE


def test_select_action_with_custom_image_key():
    """Default MPC accepts named cameras across replanning and reset."""
    set_seed(0)
    config = TDMPCConfig(device=DEVICE)
    camera_key = f"{OBS_IMAGES}.laptop"
    config.input_features = {
        camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    policy = TDMPCPolicy(config).to(DEVICE).eval()
    batch = {
        camera_key: torch.rand(2, 3, 84, 84, device=DEVICE),
        OBS_STATE: torch.rand(2, 6, device=DEVICE),
    }
    assert config.use_mpc
    for _ in range(config.n_action_repeats + 1):
        action = policy.select_action(batch)
        assert action.shape == (2, 6)
        assert torch.isfinite(action).all()
    assert policy._prev_mean is not None
    policy.reset()
    assert policy._prev_mean is None
    action = policy.select_action(batch)
    assert action.shape == (2, 6)
    assert torch.isfinite(action).all()


def test_forward_is_independent_of_image_key():
    """Both camera names consume the same augmented pixels under the same seed."""
    custom_key = f"{OBS_IMAGES}.laptop"

    def make_config(image_key):
        config = TDMPCConfig(device=DEVICE, max_random_shift_ratio=0.0476)
        config.input_features = {
            image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
        config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
        return config

    set_seed(0)
    policy_a = TDMPCPolicy(make_config(OBS_IMAGE)).to(DEVICE)
    policy_b = TDMPCPolicy(make_config(custom_key)).to(DEVICE)
    policy_b.load_state_dict(policy_a.state_dict())
    batch_size = 2
    horizon = policy_a.config.horizon
    image = torch.rand(batch_size, horizon + 1, 3, 84, 84, device=DEVICE)
    shared = {
        OBS_STATE: torch.rand(batch_size, horizon + 1, 6, device=DEVICE),
        ACTION: torch.rand(batch_size, horizon, 6, device=DEVICE),
        REWARD: torch.rand(batch_size, horizon, device=DEVICE),
        "index": torch.arange(batch_size, device=DEVICE),
        "observation.state_is_pad": torch.zeros(batch_size, horizon + 1, dtype=torch.bool, device=DEVICE),
        "action_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool, device=DEVICE),
        "next.reward_is_pad": torch.zeros(batch_size, horizon, dtype=torch.bool, device=DEVICE),
    }
    batch_a = {OBS_IMAGE: image, **shared}
    batch_b = {custom_key: image, **shared}
    set_seed(0)
    loss_a, _ = policy_a.forward(batch_a)
    set_seed(0)
    loss_b, _ = policy_b.forward(batch_b)
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
    torch.testing.assert_close(loss_a, loss_b)
