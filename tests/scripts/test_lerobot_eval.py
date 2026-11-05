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

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("datasets")

from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.configs import AlohaEnvConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.flux3 import make_flux3_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts import lerobot_eval
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.policies.flux3.helpers import task_config

pytestmark = pytest.mark.dataset


class _TwoStepEnv(gym.Env):
    """Seed selects an episode with distinct images and measured robot positions."""

    task = "move"
    task_description = "move the robot"
    _max_episode_steps = 2
    observation_space = gym.spaces.Dict(
        {
            "pixels": gym.spaces.Dict({"top": gym.spaces.Box(0, 255, (64, 96, 3), np.uint8)}),
            "agent_pos": gym.spaces.Box(-10.0, 10.0, (6,), np.float32),
        }
    )
    action_space = gym.spaces.Box(-100.0, 100.0, (6,), np.float32)

    def _observation(self):
        return {
            "pixels": {"top": np.full((64, 96, 3), self.episode * 100 + self.tick, np.uint8)},
            "agent_pos": np.full(6, self.episode + self.tick * 0.1, np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = seed
        self.tick = 0
        return self._observation(), {}

    def step(self, action):
        self.tick += 1
        return self._observation(), 0.0, self.tick == self._max_episode_steps, False, {}


class _RecordingPolicy(nn.Module):
    def reset(self):
        self.inputs = []

    def select_action(self, observation):
        self.inputs.append(
            {key: value.clone() for key, value in observation.items() if torch.is_tensor(value)}
        )
        # A nonzero normalized delta updates the FLUX3 processor's command anchor each tick.
        return torch.full((observation[OBS_STATE].shape[0], 6), 0.25)


class _AccumulatingStep(ProcessorStep):
    """Independent pre/post state: resetting one pipeline cannot mask a missing reset of the other."""

    def __init__(self, key):
        self.key = key
        self.reset()

    def reset(self):
        self.offset = 0

    def __call__(self, transition):
        self.offset += 1
        result = transition.copy()
        if self.key == OBS_STATE:
            observation = dict(result[TransitionKey.OBSERVATION])
            observation[OBS_STATE] = observation[OBS_STATE] + self.offset
            result[TransitionKey.OBSERVATION] = observation
        else:
            result[TransitionKey.ACTION] = result[TransitionKey.ACTION] + self.offset
        return result

    def transform_features(self, features):
        return features


@pytest.mark.parametrize("pipeline_kind", ["flux3", "independent", "stateless"])
def test_rollout_starts_each_episode_with_fresh_processor_state(pipeline_kind):
    def make_processors():
        if pipeline_kind == "flux3":
            return make_flux3_pre_post_processors(task_config())
        if pipeline_kind == "independent":
            return make_policy_processor_pipelines(
                [_AccumulatingStep(OBS_STATE)], [_AccumulatingStep(ACTION)]
            )
        return make_policy_processor_pipelines([], [])

    def run(env, policy, pre, post, seed):
        result = rollout(
            env,
            policy,
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=pre,
            postprocessor=post,
            seeds=[seed],
        )
        return result[ACTION], policy.inputs

    env = gym.vector.SyncVectorEnv([_TwoStepEnv])
    try:
        policy = _RecordingPolicy()
        pre, post = make_processors()
        run(env, policy, pre, post, seed=0)
        # Episode A's last FLUX3 joint command is 1.0; B starts at a distinct measured position, 2.0.
        reused_actions, reused_inputs = run(env, policy, pre, post, seed=2)

        fresh_pre, fresh_post = make_processors()
        fresh_actions, fresh_inputs = run(env, _RecordingPolicy(), fresh_pre, fresh_post, seed=2)
    finally:
        env.close()

    # Compare every processed observation (including FLUX3 image/state/command histories)
    # and every absolute action sent to the environment, not just reset() call counts.
    assert len(reused_inputs) == len(fresh_inputs) == 2
    for reused, fresh in zip(reused_inputs, fresh_inputs, strict=True):
        assert reused.keys() == fresh.keys()
        for key in fresh:
            torch.testing.assert_close(reused[key], fresh[key], rtol=0, atol=0, msg=key)
    torch.testing.assert_close(reused_actions, fresh_actions, rtol=0, atol=0)
    if pipeline_kind == "flux3":
        # Delta 0.25 unnormalizes to 0.5 and must be added to B's initial measured position.
        torch.testing.assert_close(reused_actions[0, 0, :-1], torch.full((5,), 2.5), rtol=0, atol=0)


def test_rollout_reports_what_the_environment_measures():
    class _MeasuringEnv(_TwoStepEnv):
        def step(self, action):
            observation, reward, terminated, truncated, _ = super().step(action)
            return observation, reward, terminated, truncated, {"metrics": {"tick": float(self.tick)}}

    env = gym.vector.SyncVectorEnv([_MeasuringEnv])
    try:
        result = rollout(
            env,
            _RecordingPolicy(),
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=make_policy_processor_pipelines([], [])[0],
            postprocessor=make_policy_processor_pipelines([], [])[1],
            seeds=[0],
        )
    finally:
        env.close()
    assert result["metrics"]["tick"].tolist() == [[1.0, 2.0]]


def test_rollout_ends_at_the_step_cap_below_the_env_limit():
    class _LongEnv(_TwoStepEnv):
        _max_episode_steps = 500

    env = gym.vector.SyncVectorEnv([_LongEnv])
    try:
        result = rollout(
            env,
            _RecordingPolicy(),
            env_preprocessor=PolicyProcessorPipeline(steps=[]),
            env_postprocessor=PolicyProcessorPipeline(steps=[]),
            preprocessor=make_policy_processor_pipelines([], [])[0],
            postprocessor=make_policy_processor_pipelines([], [])[1],
            seeds=[0],
            max_episode_steps=3,
        )
    finally:
        env.close()
    assert result[ACTION].shape[1] == 3
    assert result["done"][0].tolist() == [False, False, True]


class _RuntimeOwnedEnv(AlohaEnvConfig):
    def evaluation_context(self, output_dir):
        return self.context(output_dir)


def _runtime_config(tmp_path, env):
    return EvalPipelineConfig(
        env=env,
        policy=ACTConfig(device="cpu"),
        output_dir=tmp_path,
        job_name="runtime-test",
        tf32=None,
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [("policy", "requires a policy"), ("device", "no device"), ("output", "output_dir is not set")],
)
def test_eval_main_validates_before_runtime_start(tmp_path, monkeypatch, field, message):
    env = _RuntimeOwnedEnv()
    cfg = _runtime_config(tmp_path, env)
    env.context = MagicMock(side_effect=AssertionError("Runtime started before validation"))
    if field == "policy":
        cfg.policy = None
    elif field == "device":
        cfg.policy.device = None
    else:
        cfg.output_dir = None

    with pytest.raises(ValueError, match=message):
        lerobot_eval.eval_main(cfg)

    env.context.assert_not_called()


def test_runtime_start_failure_prevents_evaluation(tmp_path, monkeypatch):
    env = _RuntimeOwnedEnv()

    @contextmanager
    def context(output_dir):
        raise RuntimeError("simulator startup failed")
        yield

    env.context = context
    setup = MagicMock()
    monkeypatch.setattr(lerobot_eval, "make_env", setup)
    with pytest.raises(RuntimeError, match="simulator startup failed"):
        lerobot_eval.eval_main(_runtime_config(tmp_path, env))
    setup.assert_not_called()


@pytest.mark.parametrize("provider", [False, True])
@pytest.mark.parametrize("failure", [None, "policy", "evaluation"])
def test_standard_evaluation_lifetime(tmp_path, monkeypatch, provider, failure):
    events = []
    output_dir = tmp_path / "results"
    env = _RuntimeOwnedEnv() if provider else AlohaEnvConfig()
    if provider:

        @contextmanager
        def context(path):
            assert path == output_dir
            events.append("start")
            try:
                yield
            finally:
                assert events[-1] == "close"
                assert (path / "eval_info.json").exists() is (failure is None)
                events.append("stop")

        env.context = context
    cfg = _runtime_config(output_dir, env)
    envs = {"default": {0: object()}}
    policy = MagicMock()
    policy.config = cfg.policy
    info = {"overall": {"pc_success_ci95": [0.0, 100.0], "pc_success": 0.0, "n_success": 0, "n_episodes": 1}}
    device = MagicMock(return_value=torch.device("cpu"))
    seed = MagicMock()

    def create_env(*args, **kwargs):
        if provider:
            assert events == ["start"]
        events.append("environment")
        return envs

    make_env = MagicMock(side_effect=create_env)
    make_policy = MagicMock(return_value=policy)
    if failure == "policy":
        make_policy.side_effect = RuntimeError("policy failed")
    evaluate = MagicMock(return_value=info)
    if failure == "evaluation":
        evaluate.side_effect = RuntimeError("evaluation failed")

    def close_env(*args):
        assert (output_dir / "eval_info.json").exists() is (failure is None)
        events.append("close")

    close = MagicMock(side_effect=close_env)
    monkeypatch.setattr(lerobot_eval, "get_safe_torch_device", device)
    monkeypatch.setattr(lerobot_eval, "set_seed", seed)
    monkeypatch.setattr(lerobot_eval, "make_env", make_env)
    monkeypatch.setattr(lerobot_eval, "make_policy", make_policy)
    monkeypatch.setattr(lerobot_eval, "make_pre_post_processors", lambda **kwargs: ("pre", "post"))
    monkeypatch.setattr(
        lerobot_eval, "make_env_pre_post_processors", lambda **kwargs: ("env_pre", "env_post")
    )
    monkeypatch.setattr(lerobot_eval, "eval_policy_all", evaluate)
    monkeypatch.setattr(lerobot_eval, "close_envs", close)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)

    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            lerobot_eval.eval_main(cfg)
        close.assert_called_once_with(envs)
        assert evaluate.call_count == (1 if failure == "evaluation" else 0)
        assert events[-1] == ("stop" if provider else "close")
        return

    lerobot_eval.eval_main(cfg)

    device.assert_called_once_with("cpu", log=True)
    seed.assert_called_once_with(cfg.seed)
    make_env.assert_called_once_with(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    make_policy.assert_called_once_with(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    assert evaluate.call_count == 1
    assert evaluate.call_args.kwargs["start_seed"] == cfg.seed
    assert evaluate.call_args.kwargs["preprocessor"] == "pre"
    close.assert_called_once_with(envs)
    assert json.loads((output_dir / "eval_info.json").read_text()) == info
    assert events[-1] == ("stop" if provider else "close")


def test_eval_metrics_exclude_steps_after_first_done(monkeypatch):
    policy = MagicMock(spec=PreTrainedPolicy)
    policy.config = SimpleNamespace(save_predicted_video=False)
    policy.training = False
    data = {
        "done": torch.tensor([[True, True, True], [False, True, True]]),
        "reward": torch.tensor([[-3.0, 99.0, 99.0], [-4.0, -2.0, 99.0]]),
        "success": torch.tensor([[False, True, True], [False, True, False]]),
    }
    monkeypatch.setattr(lerobot_eval, "rollout", lambda **kwargs: data)
    result = lerobot_eval.eval_policy(SimpleNamespace(num_envs=2), policy, None, None, None, None, 2)
    assert result["per_episode"][0]["sum_reward"] == -3.0
    assert result["per_episode"][0]["max_reward"] == -3.0
    assert result["per_episode"][0]["success"] is False
    assert result["per_episode"][1]["sum_reward"] == -6.0
    assert result["per_episode"][1]["max_reward"] == -2.0
    assert result["per_episode"][1]["success"] is True
