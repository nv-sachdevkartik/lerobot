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
import abc
import argparse
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
import torch
from draccus.cfgparsing import parse_string
from huggingface_hub import constants as hub_constants, hf_hub_download, snapshot_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.optim import LRSchedulerConfig, OptimizerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.utils.hub import HubMixin

from .types import FeatureType, PolicyFeature

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
    """

    n_obs_steps: int = 1

    # `input_features` can be set to None/null in order to infer those values from the dataset.
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # Parameter storage dtype. None uses PyTorch's default when constructing a policy.
    dtype: torch.dtype | None = None  # e.g. torch.float32, torch.bfloat16
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    # Whether the policy employed PEFT for training.
    use_peft: bool = False

    push_to_hub: bool = True  # TODO: rename; shadows HubMixin.push_to_hub()
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None
    # Optional Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
    pretrained_revision: str | None = None

    def __post_init__(self) -> None:
        if self.dtype is not None and not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"config.dtype must be a torch.dtype or None, got {self.dtype!r}.")

        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list[int] | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list[int] | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list[int] | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        # Encode against the base class so draccus includes the choice "type" key,
        # which `from_pretrained` needs to resolve the concrete subclass.
        with open(save_directory / CONFIG_NAME, "w") as f:
            json.dump(draccus.encode(self, PreTrainedConfig), f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        # Resolve the standard CLI revision before fetching config.json, rather
        # than applying it only after a potentially different revision was read.
        revision_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        revision_parser.add_argument("--pretrained_revision", default=argparse.SUPPRESS)
        revision_parser.add_argument("--embodiment_tag")
        source_options = revision_parser.parse_known_args(cli_overrides)[0]
        if hasattr(source_options, "pretrained_revision"):
            # Match the JSON/Optional[str] decoding used by the full config
            # parser: null means None, while a quoted "null" names a revision.
            with draccus.config_type("json"):
                revision = draccus.decode(str | None, parse_string(source_options.pretrained_revision))
        model_id = str(pretrained_name_or_path)
        if Path(model_id).expanduser().is_dir():
            model_id = str(Path(model_id).expanduser())
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            config = json.load(f)

        # Resolve the concrete config subclass from the serialized "type" tag, then parse
        # the config (with CLI overrides) directly for that class. The "type" key is
        # stripped because draccus only consumes it when parsing the registry base class.
        policy_type = config.pop("type", None)
        raw_groot = False
        if policy_type is None:
            from lerobot.policies.groot.configuration_groot import (
                GROOT_N1_7,
                _infer_groot_model_version_from_config,
                raw_groot_config_defaults,
            )

            # A shared Cosmos backbone name alone is not a GR00T architecture
            # declaration. Only explicit native version/type/architecture tags
            # may opt an otherwise untyped config into this loader.
            native_identity = {key: value for key, value in config.items() if key != "model_name"}
            if _infer_groot_model_version_from_config(native_identity) != GROOT_N1_7:
                raise ValueError(f"Missing 'type' field in {CONFIG_NAME} of {model_id}")
            if not Path(model_id).is_dir():
                # hf_hub_download returns a snapshot path, whose directory is
                # the immutable commit. Use that same commit for all weights
                # and sidecars, even if the requested revision was a branch.
                snapshot = Path(config_file).parent
                commit = snapshot.name
                if (
                    snapshot.parent.name != "snapshots"
                    or len(commit) != 40
                    or any(char not in "0123456789abcdef" for char in commit)
                ):
                    raise ValueError(f"Cannot determine immutable Hub snapshot from {config_file}")
                model_id = snapshot_download(
                    repo_id=model_id,
                    revision=commit,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    token=token,
                    local_files_only=local_files_only or hub_constants.HF_HUB_OFFLINE,
                    **({"proxies": proxies} if proxies is not None else {}),
                )
                revision = commit
            model_id = str(Path(model_id).expanduser().resolve())
            config = raw_groot_config_defaults(model_id, source_options.embodiment_tag)
            policy_type = "groot"
            raw_groot = True
        try:
            config_cls = cls.get_choice_class(policy_type)
        except Exception as e:
            raise ValueError(
                f"Policy type '{policy_type}' (from {CONFIG_NAME} of {model_id}) is not registered. "
                f"Available policy types: {cls.get_known_choices()}"
            ) from e

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        with draccus.config_type("json"):
            config = draccus.parse(config_cls, config_file, args=cli_overrides)
        if raw_groot and Path(config.base_model_path).expanduser().resolve() != Path(model_id):
            raise ValueError("A raw --policy.path cannot be combined with a different base_model_path")

        # The serialized config.json does not carry the source path, so parsing
        # yields pretrained_path=None. Backfill it: downstream code (e.g.
        # make_policy / make_pre_post_processors) silently builds an untrained
        # processor pipeline without it, instead of loading the pretrained stats.
        config.pretrained_path = Path(model_id)
        config.pretrained_revision = revision
        return config
