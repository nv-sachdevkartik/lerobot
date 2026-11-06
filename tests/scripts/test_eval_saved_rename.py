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

"""The evaluator admits the camera names its saved processors actually produce."""

from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    DeviceProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.scripts import lerobot_eval
from tests.fixtures.toy_env import ToyEnvConfig

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("override", [None, {}, {"observation.image": "observation.images.override"}])
def test_saved_rename_used_for_admission_and_policy(tmp_path, monkeypatch, override):
    saved = {"observation.image": "observation.images.saved"}
    expected = saved if override is None else override
    target = expected.get("observation.image", "observation.image")
    pre, post = make_policy_processor_pipelines(
        [RenameObservationsProcessorStep(saved), DeviceProcessorStep(device="cpu")], []
    )
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    policy_cfg = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            target: PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        pretrained_path=tmp_path,
    )
    cfg = EvalPipelineConfig(env=ToyEnvConfig(), policy=policy_cfg, output_dir=tmp_path, rename_map=override)
    env = cfg.env.create_env()
    close = Mock(wraps=env.close)
    monkeypatch.setattr(env, "close", close)
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *a, **k: {"toy": {0: env}})
    admitted = []

    def stop_before_weights(**kwargs):
        admitted.append(kwargs["rename_map"])
        raise RuntimeError("stop before weights")

    monkeypatch.setattr(lerobot_eval, "make_policy", stop_before_weights)
    with pytest.raises(RuntimeError, match="stop before weights"):
        lerobot_eval.eval_main.__wrapped__(cfg)
    assert admitted == [expected]
    close.assert_called_once()


def test_sequential_saved_renames_match_pipeline_output():
    pre, _ = make_policy_processor_pipelines(
        [
            RenameObservationsProcessorStep({"observation.images.raw": "observation.images.middle"}),
            RenameObservationsProcessorStep({"observation.images.middle": "observation.images.final"}),
        ],
        [],
    )
    mapping = lerobot_eval._processor_rename_map(pre)
    image = torch.ones(1, 3, 8, 8)
    output = pre({"observation.images.raw": image})
    assert mapping["observation.images.raw"] == "observation.images.final"
    assert torch.equal(output[mapping["observation.images.raw"]], image)


def test_processor_load_error_closes_environment(tmp_path, monkeypatch):
    cfg = EvalPipelineConfig(
        env=ToyEnvConfig(), policy=ACTConfig(device="cpu", pretrained_path=tmp_path), output_dir=tmp_path
    )
    env = cfg.env.create_env()
    close = Mock(wraps=env.close)
    monkeypatch.setattr(env, "close", close)
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *a, **k: {"toy": {0: env}})
    monkeypatch.setattr(
        lerobot_eval, "make_pre_post_processors", Mock(side_effect=ValueError("bad processor"))
    )
    with pytest.raises(ValueError, match="bad processor"):
        lerobot_eval.eval_main.__wrapped__(cfg)
    close.assert_called_once()


@pytest.mark.parametrize("rename", [None, {"observation.image": "observation.images.front"}])
def test_scratch_policy_infers_features_before_building_processors(tmp_path, monkeypatch, rename):
    cfg = EvalPipelineConfig(
        env=ToyEnvConfig(),
        policy=ACTConfig(
            device="cpu",
            pretrained_backbone_weights=None,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
        ),
        output_dir=tmp_path,
        rename_map=rename,
    )
    env = cfg.env.create_env()
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *a, **k: {"toy": {0: env}})
    target = (rename or {}).get("observation.image", "observation.image")

    def inspect_before_rollout(**kwargs):
        assert target in kwargs["policy"].config.input_features
        normalizer = next(s for s in kwargs["preprocessor"].steps if isinstance(s, NormalizerProcessorStep))
        unnormalizer = next(
            s for s in kwargs["postprocessor"].steps if isinstance(s, UnnormalizerProcessorStep)
        )
        assert {"observation.state", target} <= normalizer.features.keys()
        assert "action" in unnormalizer.features
        assert lerobot_eval._processor_rename_map(kwargs["preprocessor"]) == (rename or {})
        raise RuntimeError("scratch setup complete")

    monkeypatch.setattr(lerobot_eval, "eval_policy_all", inspect_before_rollout)
    with pytest.raises(RuntimeError, match="scratch setup complete"):
        lerobot_eval.eval_main.__wrapped__(cfg)
