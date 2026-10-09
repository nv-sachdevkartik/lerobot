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

"""Episode metrics must exclude native vector autoreset transitions after first done."""

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("datasets")

from lerobot.policies import ACTConfig, PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts.lerobot_eval import eval_policy
from lerobot.utils.constants import OBS_STATE


class _ZeroPolicy(PreTrainedPolicy):
    config_class = ACTConfig
    name = "zero_eval_test"

    def __init__(self):
        super().__init__(ACTConfig(device="cpu"))

    def reset(self):
        pass

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch):
        raise NotImplementedError

    def predict_action_chunk(self, batch, **kwargs):
        return self.select_action(batch).unsqueeze(1)

    def select_action(self, batch, **kwargs):
        return torch.zeros(batch[OBS_STATE].shape[0], 1)


class _NativeAutoresetBatch(gym.vector.VectorEnv):
    """Slot zero starts a second episode while slot one is still in its first."""

    num_envs = 2
    _max_episode_steps = 3
    task = "test"
    task_description = "test native autoreset episode metrics"

    def __init__(self, terminal_reward, terminal_success):
        self.terminal_reward = terminal_reward
        self.terminal_success = terminal_success
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.tick = 0
        return {"agent_pos": np.zeros((2, 1), np.float32)}, {}

    def step(self, actions):
        assert actions.shape == (2, 1)
        self.tick += 1
        first_reward = self.terminal_reward if self.tick == 1 else 1000.0 * self.tick
        first_success = self.terminal_success if self.tick == 1 else True
        return (
            {"agent_pos": np.full((2, 1), self.tick, np.float32)},
            np.array([first_reward, self.tick], np.float32),
            np.array([True, self.tick == 3]),
            np.array([False, False]),
            {"final_info": {"is_success": np.array([first_success, self.tick == 3])}},
        )

    def call(self, name, *args, **kwargs):
        return (getattr(self, name),) * self.num_envs

    def get_attr(self, name):
        return self.call(name)


@pytest.mark.parametrize("terminal_reward,terminal_success", [(5.0, False), (-5.0, False), (5.0, True)])
def test_eval_policy_stops_metrics_at_each_slots_first_done(terminal_reward, terminal_success):
    env = _NativeAutoresetBatch(terminal_reward, terminal_success)
    pre, post = make_policy_processor_pipelines([], [])
    try:
        result = eval_policy(
            env,
            _ZeroPolicy(),
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=pre,
            postprocessor=post,
            n_episodes=2,
        )
    finally:
        env.close()

    first, second = result["per_episode"]
    assert first["sum_reward"] == terminal_reward
    assert first["success"] == terminal_success
    assert second["sum_reward"] == 6.0
    assert second["max_reward"] == 3.0
    assert second["success"]
    assert result["aggregated"]["avg_sum_reward"] == (terminal_reward + 6.0) / 2
    assert result["aggregated"]["n_success"] == 1 + int(terminal_success)
    # Invalid transitions must not add zero to an entirely negative episode's maximum.
    assert first["max_reward"] == terminal_reward
