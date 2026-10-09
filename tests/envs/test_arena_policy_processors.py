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

"""Real processor-boundary checks without model weights or checkpoint tokenizer claims."""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

LANGUAGE_POLICIES = {
    "pi0": "text_tokenizer_name",
    "pi05": "text_tokenizer_name",
    "fineart_vla": "text_tokenizer_name",
    "smolvla": "vlm_model_name",
    "multi_task_dit": "text_encoder_name",
    "xvla": "tokenizer_name",
}
PROCESSOR_POLICIES = sorted(
    {"act", "diffusion", "tdmpc", "vqbet", "gaussian_actor", "evo1", "fastwam", "lingbot_va", "vla_jepa"}
    | LANGUAGE_POLICIES.keys()
)


@pytest.fixture
def local_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "move": 2, "left": 3, "right": 4}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    path = tmp_path / "clip-local-tokenizer"
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize(
    "policy_name,normalization",
    [
        (name, mode)
        for name in PROCESSOR_POLICIES
        for mode in (
            [NormalizationMode.MIN_MAX]
            if name == "tdmpc"
            else [NormalizationMode.MEAN_STD, NormalizationMode.QUANTILES]
        )
    ],
)
def test_arena_through_policy_processors_to_numpy_action(policy_name, normalization, request):
    camera = f"{OBS_IMAGES}.front"
    kwargs = {
        "device": "cpu",
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
            camera: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        "normalization_mapping": {
            "STATE": normalization,
            "ACTION": normalization,
            "VISUAL": NormalizationMode.IDENTITY,
        },
    }
    if policy_name in LANGUAGE_POLICIES:
        kwargs[LANGUAGE_POLICIES[policy_name]] = request.getfixturevalue("local_tokenizer")
    if policy_name == "fastwam":
        kwargs.update(proprio_dim=7, image_size=(224, 224))
    elif policy_name == "lingbot_va":
        kwargs["obs_cam_keys"] = [camera]
    elif policy_name == "xvla":
        kwargs["action_mode"] = "auto"
    elif policy_name == "fineart_vla":
        kwargs["enable_fast_action_loss"] = False
    config = make_policy_config(policy_name, **kwargs)
    config.validate_features()
    # Both mappings represent the same physical transform. Values are synthetic,
    # deliberately nonidentity statistics, never substituted into a real checkpoint.
    stats = {
        OBS_STATE: {
            "mean": torch.full((7,), 2.0),
            "std": torch.full((7,), 2.0),
            "q01": torch.zeros(7),
            "q99": torch.full((7,), 4.0),
            "min": torch.zeros(7),
            "max": torch.full((7,), 4.0),
        },
        ACTION: {
            "mean": torch.full((7,), 10.0),
            "std": torch.full((7,), 4.0),
            "q01": torch.full((7,), 6.0),
            "q99": torch.full((7,), 14.0),
            "min": torch.full((7,), 6.0),
            "max": torch.full((7,), 14.0),
        },
    }
    policy_pre, policy_post = make_pre_post_processors(config, dataset_stats=stats)
    env_config = IsaaclabArenaEnv(
        state_keys="joints",
        state_dim=7,
        action_dim=7,
        camera_keys="front",
        enable_cameras=True,
        camera_height=224,
        camera_width=224,
    )
    env_pre, env_post = make_env_pre_post_processors(env_config, config)
    observation = preprocess_observation(
        {
            "policy": {"joints": torch.full((2, 7), 3.0)},
            "camera_obs": {"front": torch.full((2, 224, 224, 3), 128, dtype=torch.uint8)},
        }
    )
    observation["task"] = ["move left", "move right"]
    processed = policy_pre(env_pre(observation))
    torch.testing.assert_close(processed[OBS_STATE][:, :7], torch.full((2, 7), 0.5))
    assert processed[camera].shape == (2, 3, 224, 224)
    assert torch.isfinite(processed[camera]).all()
    if policy_name != "xvla":  # XVLA additionally applies ImageNet image normalization.
        torch.testing.assert_close(processed[camera], torch.full((2, 3, 224, 224), 128 / 255))
    if policy_name in LANGUAGE_POLICIES:
        assert processed[OBS_LANGUAGE_TOKENS].shape[0] == 2
        assert processed[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool
        assert processed[OBS_LANGUAGE_ATTENTION_MASK].any(dim=-1).all()
        assert not torch.equal(processed[OBS_LANGUAGE_TOKENS][0], processed[OBS_LANGUAGE_TOKENS][1])

    action_dim = config.max_action_dim if policy_name == "evo1" else 7
    action = env_post({ACTION: policy_post(torch.full((2, action_dim), 0.5))})[ACTION]
    assert action.device.type == "cpu"
    np.testing.assert_allclose(action.numpy(), np.full((2, 7), 12.0), atol=1e-6)


@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_arena_flux3_synchronous_command_history_and_reset(representation):
    """Submitted commands, rather than measured state, anchor subsequent deltas."""
    camera = f"{OBS_IMAGES}.front"
    config = make_policy_config(
        "flux3",
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
            camera: PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        camera_layout="single",
        conditioning="history",
        n_obs_steps=2,
        history_snapshots=1,
        condition_on_past_actions=True,
        action_representation=representation,
        normalization_stats={name: {"q01": [-2.0] * 7, "q99": [2.0] * 7} for name in ("state", "action")},
    )
    config.validate_features()
    pre, post = make_pre_post_processors(config)
    env_pre, env_post = make_env_pre_post_processors(
        IsaaclabArenaEnv(
            state_keys="joints",
            state_dim=7,
            action_dim=7,
            camera_keys="front",
            enable_cameras=True,
            camera_height=16,
            camera_width=16,
        ),
        config,
    )

    def observe(state_value):
        observation = preprocess_observation(
            {
                "policy": {"joints": torch.full((2, 7), state_value)},
                "camera_obs": {"front": torch.full((2, 16, 16, 3), 128, dtype=torch.uint8)},
            }
        )
        observation["task"] = ["move left", "move right"]
        return pre(env_pre(observation))

    def submit():
        return env_post({ACTION: post(torch.full((2, 7), 0.25))})[ACTION].numpy()

    first = observe(0.0)
    assert first[OBS_STATE].shape == (2, 2, 7)
    assert first[camera].shape == (2, 2, 3, 16, 16)
    assert first["observation.past_actions"].shape == (2, 2, 7)
    np.testing.assert_allclose(submit(), 0.5, atol=1e-6)
    observe(0.1)  # The measured joint lagged behind the submitted command.
    np.testing.assert_allclose(submit(), 1.0 if representation == "delta" else 0.5, atol=1e-6)
    pre.reset()
    post.reset()
    observe(-1.0)
    np.testing.assert_allclose(submit(), -0.5 if representation == "delta" else 0.5, atol=1e-6)
