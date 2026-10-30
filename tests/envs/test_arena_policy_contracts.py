"""Configuration/routing coverage only; these tests do not load policy weights or Isaac Sim."""

from __future__ import annotations

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import IsaaclabArenaEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies import make_policy_config
from lerobot.processor import IsaaclabArenaProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# Derived from the registry rather than frozen, so a new built-in is covered automatically instead of
# breaking an unrelated inventory assertion. A policy whose defaults cannot express the synthetic
# contract below needs an entry in the per-policy keyword arguments in the test that follows.
POLICIES = {
    name
    for name, cls in PreTrainedConfig.get_known_choices().items()
    if cls.__module__.startswith("lerobot.policies.")
}


@pytest.mark.parametrize("policy_name", sorted(POLICIES))
def test_policy_feature_contract_and_arena_processor_routing(policy_name):
    # A synthetic single RGB view and seven-dimensional state/action contract.
    # Equal state/action widths allow Flux3; this is not the 54/36 GR1 contract.
    kwargs = {
        "device": "cpu",
        "input_features": {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
            f"{OBS_IMAGES}.front": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
    }
    if policy_name == "fineart_vla":
        kwargs["enable_fast_action_loss"] = False
    elif policy_name == "fastwam":
        kwargs.update(proprio_dim=7, image_size=(224, 224))
    elif policy_name == "flux3":
        kwargs["camera_layout"] = "single"
    elif policy_name == "lingbot_va":
        kwargs["obs_cam_keys"] = [f"{OBS_IMAGES}.front"]
    elif policy_name == "longwam":
        kwargs.update(camera_keys=(f"{OBS_IMAGES}.front",), proprio_dim=7)
    elif policy_name == "lingbot_vla_v2":
        # Map the synthetic 7-dim contract onto one canonical slot; the default mapping
        # spans the policy's 55 canonical dims, which this contract cannot express.
        from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import SlotMapping

        kwargs.update(
            max_state_dim=7,
            max_action_dim=7,
            state_slots={"arm.position": SlotMapping(origin_keys=[{OBS_STATE: {"start": 0, "end": 7}}])},
            action_slots={"arm.position": SlotMapping(origin_keys=[{ACTION: {"start": 0, "end": 7}}])},
        )
    elif policy_name == "wall_oss_05":
        kwargs["camera_key_mapping"] = {f"{OBS_IMAGES}.front": "face_view"}
    elif policy_name == "xvla":
        kwargs["action_mode"] = "auto"
    elif policy_name == "eo1":
        # Keep config/routing tests offline; no pretrained VLM is instantiated.
        kwargs["vlm_config"] = {"model_type": "qwen2_5_vl"}
    policy_cfg = make_policy_config(policy_name, **kwargs)
    policy_cfg.validate_features()

    pre, post = make_env_pre_post_processors(
        IsaaclabArenaEnv(
            state_dim=7,
            action_dim=7,
            camera_keys="front",
            camera_height=224,
            camera_width=224,
            enable_cameras=True,
        ),
        policy_cfg=policy_cfg,
    )
    assert isinstance(pre.steps[0], IsaaclabArenaProcessorStep)
    assert post.steps == []
    assert policy_cfg.output_features[ACTION].shape == (7,)


@pytest.mark.parametrize("policy_name", ["wall_x", "dm05", "lawam"])
def test_legacy_gr1_contract_exceeds_default_policy_dimensions(policy_name):
    cfg = make_policy_config(
        policy_name,
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (54,)),
            f"{OBS_IMAGES}.front": PolicyFeature(FeatureType.VISUAL, (3, 512, 512)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (36,))},
    )
    with pytest.raises(ValueError, match="dimension|width"):
        cfg.validate_features()
