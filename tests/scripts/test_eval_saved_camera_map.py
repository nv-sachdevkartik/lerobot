"""Keep saved camera identities; distinct feature names require an explicit admission map."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import AlohaEnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline, RenameObservationsProcessorStep
from lerobot.scripts import lerobot_eval
from lerobot.sim.types import EnvDescriptor
from lerobot.utils.feature_spec import FeatureSpec

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("with_descriptor", [False, True], ids=["native", "served"])
@pytest.mark.parametrize("same_names", [False, True], ids=["distinct-names", "permutation"])
@pytest.mark.parametrize("explicit", [False, True], ids=["saved", "cli"])
def test_eval_loads_saved_camera_map(tmp_path, monkeypatch, with_descriptor, same_names, explicit):
    left, right = "observation.images.left", "observation.images.right"
    sources = (left, right) if same_names else ("observation.images.z", "observation.images.a")
    saved_map = dict(zip(sources, (right, left), strict=True))
    cli_map = dict(zip(sources, (left, right), strict=True)) if explicit else {}
    checkpoint = tmp_path / "checkpoint"
    PolicyProcessorPipeline(
        [RenameObservationsProcessorStep(rename_map=saved_map), DeviceProcessorStep(device="cpu")]
    ).save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    PolicyProcessorPipeline([DeviceProcessorStep(device="cpu")]).save_pretrained(
        checkpoint, config_filename="policy_postprocessor.json"
    )
    saved_bytes = (checkpoint / "policy_preprocessor.json").read_bytes()
    cfg = EvalPipelineConfig(
        env=AlohaEnvConfig(),
        policy=ACTConfig(
            device="cpu",
            input_features={key: PolicyFeature(FeatureType.VISUAL, (3, 2, 3)) for key in (left, right)},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (1,))},
        ),
        rename_map=cli_map,
        output_dir=tmp_path / "results",
        job_name="camera-map-test",
        tf32=None,
    )
    cfg.policy.pretrained_path = checkpoint
    descriptor = (
        EnvDescriptor(
            sim_type="test",
            semantics="test",
            control_mode="position",
            fps=30,
            tasks={"test": [0]},
            max_episode_steps=2,
            features=tuple(FeatureSpec(key, (2, 3, 3), "uint8", "rgb", semantics="rgb") for key in sources),
            action_feature=FeatureSpec("action", (1,), "float32", names=("joint",), semantics="radians"),
        )
        if with_descriptor
        else None
    )
    world = SimpleNamespace(descriptor=descriptor)
    monkeypatch.setattr(lerobot_eval, "make_env", lambda *args, **kwargs: {"test": {0: world}})
    close = MagicMock()
    monkeypatch.setattr(lerobot_eval, "close_envs", close)
    monkeypatch.setattr(lerobot_eval, "make_env_pre_post_processors", lambda **kwargs: (None, None))
    policy = MagicMock(config=cfg.policy)

    def construct_policy(**kwargs):
        # Preserve the real factory's feature check while avoiding model weights.
        features = {
            kwargs["rename_map"].get(key, key): PolicyFeature(FeatureType.VISUAL, (3, 2, 3))
            for key in sources
        }
        validate_visual_features_consistency(kwargs["cfg"], features)
        return policy

    make_policy = MagicMock(side_effect=construct_policy)
    monkeypatch.setattr(lerobot_eval, "make_policy", make_policy)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)

    def evaluate(**kwargs):
        batch = {
            key: torch.full((1, 3, 2, 3), value, dtype=torch.uint8)
            for key, value in zip(sources, (17, 231), strict=True)
        }
        processed = kwargs["preprocessor"](batch)
        assert [processed[key][0, 0, 0, 0].item() for key in (left, right)] == (
            [17, 231] if explicit else [231, 17]
        )
        return {
            "overall": {"pc_success_ci95": [0.0, 100.0], "pc_success": 0.0, "n_success": 0, "n_episodes": 1}
        }

    evaluate = MagicMock(side_effect=evaluate)
    monkeypatch.setattr(lerobot_eval, "eval_policy_all", evaluate)
    if not same_names and not explicit:
        # Existing admission cannot infer distinct names from a saved pipeline. It must fail,
        # rather than silently start a rollout using a guessed camera order.
        with pytest.raises(ValueError):
            lerobot_eval.eval_main(cfg)
        evaluate.assert_not_called()
        assert make_policy.call_count == (0 if with_descriptor else 1)
    else:
        lerobot_eval.eval_main(cfg)
        make_policy.assert_called_once_with(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cli_map)
        evaluate.assert_called_once()
    close.assert_called_once()
    assert cfg.rename_map == cli_map
    assert (checkpoint / "policy_preprocessor.json").read_bytes() == saved_bytes
