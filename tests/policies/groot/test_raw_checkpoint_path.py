"""CPU native checkpoint loading contracts: no downloads or model allocation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from draccus.utils import ParsingError

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.groot.configuration_groot import GrootConfig, resolve_groot_processor_directory
from tests.policies.groot.test_groot_n1_7 import _write_raw_n1_7_libero_checkpoint


def raw_checkpoint(path, nested=True):
    _write_raw_n1_7_libero_checkpoint(path)
    if nested:
        (path / "processor").mkdir()
        for name in ("processor_config.json", "statistics.json", "embodiment_id.json"):
            (path / name).rename(path / "processor" / name)
    return path


@pytest.mark.parametrize("nested", [False, True])
def test_native_path_infers_metadata_and_builds_processors(tmp_path, nested):
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained

    path = raw_checkpoint(tmp_path / "raw", nested)
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*.json")}
    cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu"])
    assert isinstance(cfg, GrootConfig)
    assert cfg.embodiment_tag == "libero_sim"
    assert cfg.chunk_size == 16 and cfg.n_action_steps == 8
    assert cfg.pretrained_path == path and cfg.base_model_path == str(path)
    pre, post = make_groot_pre_post_processors_from_pretrained(cfg, str(path))
    assert pre.steps and post.steps
    assert before == {p.relative_to(path): p.read_bytes() for p in path.rglob("*.json")}


def test_native_cli_overrides_validated(tmp_path):
    path = raw_checkpoint(tmp_path / "raw")
    cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu", "--n_action_steps=4"])
    assert cfg.n_action_steps == 4
    with pytest.raises(ParsingError) as error:
        PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu", "--n_action_steps=99"])
    assert "n_action_steps" in str(error.value.__cause__)
    with pytest.raises(ValueError, match="different base_model_path"):
        PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu", "--base_model_path=other"])


def test_unknown_untyped_config_still_rejected(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"unknown"}')
    with pytest.raises(ValueError, match="Missing 'type'"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_conflicting_sidecar_layout_rejected(tmp_path):
    path = raw_checkpoint(tmp_path / "raw")
    (path / "processor_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_groot_processor_directory(path)


def test_hub_raw_resolves_config_commit_once_and_keeps_path_in_eval(tmp_path, monkeypatch):
    import lerobot.configs.policies as module
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    commit = "a" * 40
    (tmp_path / "snapshots").mkdir()
    path = raw_checkpoint(tmp_path / "snapshots" / commit)
    download = Mock(return_value=str(path / "config.json"))
    snapshot = Mock(return_value=str(path))
    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(parser, "get_path_arg", lambda _: "org/native")
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=old"])
    monkeypatch.setattr(
        parser, "get_cli_overrides", lambda _: ["--pretrained_revision", "release", "--device=cpu"]
    )
    cfg = EvalPipelineConfig(env=None).policy
    assert download.call_args.kwargs["revision"] == "release"
    assert snapshot.call_count == 1
    assert snapshot.call_args.kwargs["revision"] == commit
    assert cfg.pretrained_path == path and cfg.base_model_path == str(path)
    assert cfg.pretrained_revision == commit


def test_single_file_raw_model_not_lerobot_state_dict(tmp_path, monkeypatch):
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.pretrained import PreTrainedPolicy

    path = raw_checkpoint(tmp_path / "raw")
    (path / "model.safetensors").write_bytes(b"not loaded in CPU contract")
    cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu"])
    monkeypatch.setattr(
        GrootPolicy, "__init__", lambda self, config: object.__setattr__(self, "config", config)
    )
    monkeypatch.setattr(GrootPolicy, "eval", lambda self: self)
    parent = Mock(side_effect=AssertionError("must not use LeRobot state dict loader"))
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", parent)
    assert GrootPolicy.from_pretrained(path, config=cfg).config.base_model_path == str(path)
    parent.assert_not_called()


def test_processor_revision_forwarded_and_serialized(monkeypatch):
    from lerobot.policies.groot import processor_groot as module

    loaders = []
    for name in ("AutoTokenizer", "Qwen2VLImageProcessor", "Qwen3VLVideoProcessor"):
        loader = Mock(return_value=SimpleNamespace(chat_template="template"))
        monkeypatch.setattr(module, name, SimpleNamespace(from_pretrained=loader))
        loaders.append(loader)
    monkeypatch.setattr(module, "Qwen3VLProcessor", lambda **kwargs: SimpleNamespace(**kwargs))
    step = module.GrootN17VLMEncodeStep(model_name="org/pinned", model_revision="commit")
    assert step.proc.tokenizer.padding_side == "left"
    for loader in loaders:
        loader.assert_called_once_with("org/pinned", revision="commit", trust_remote_code=True)
    restored = module.GrootN17VLMEncodeStep(**step.get_config())
    assert restored.model_revision == "commit"
    assert module.GrootN17VLMEncodeStep().model_revision is None


def test_cosmos_name_alone_is_not_raw_groot_identity(tmp_path):
    (tmp_path / "config.json").write_text('{"model_name":"nvidia/Cosmos-Reason2-2B"}')
    with pytest.raises(ValueError, match="Missing 'type'"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_typed_groot_retains_saved_config_and_single_file_loader(tmp_path, monkeypatch):
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.pretrained import PreTrainedPolicy

    GrootConfig(device="cpu", chunk_size=12, n_action_steps=6).save_pretrained(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"mock state dict")
    cfg = PreTrainedConfig.from_pretrained(tmp_path)
    assert cfg.chunk_size == 12 and cfg.n_action_steps == 6
    parent = Mock(return_value="typed-policy")
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", parent)
    assert GrootPolicy.from_pretrained(tmp_path, config=cfg) == "typed-policy"
    parent.assert_called_once()


def test_raw_hub_requires_snapshot_commit(tmp_path, monkeypatch):
    import lerobot.configs.policies as module

    path = raw_checkpoint(tmp_path / "raw")
    monkeypatch.setattr(module, "hf_hub_download", Mock(return_value=str(path / "config.json")))
    snapshot = Mock()
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    with pytest.raises(ValueError, match="immutable Hub snapshot"):
        PreTrainedConfig.from_pretrained("org/native")
    snapshot.assert_not_called()


def test_explicit_embodiment_selects_its_horizon_before_cli_validation(tmp_path):
    import json

    path = raw_checkpoint(tmp_path / "raw")
    processor_path = path / "processor" / "processor_config.json"
    content = json.loads(processor_path.read_text())
    content["processor_kwargs"]["modality_configs"]["other_robot"] = {
        "action": {"delta_indices": list(range(20))}
    }
    processor_path.write_text(json.dumps(content))
    cfg = PreTrainedConfig.from_pretrained(
        path, cli_overrides=["--device=cpu", "--embodiment_tag=other_robot"]
    )
    assert cfg.embodiment_tag == "other_robot"
    assert cfg.chunk_size == 20 and cfg.n_action_steps == 20
    assert cfg.action_decode_transform is None


def test_hub_snapshot_strict_signature_omits_unset_proxies(tmp_path, monkeypatch):
    import lerobot.configs.policies as module

    commit = "b" * 40
    (tmp_path / "snapshots").mkdir()
    path = raw_checkpoint(tmp_path / "snapshots" / commit)
    calls = []

    # Deliberately no **kwargs: the new snapshot call needs no unset legacy option.
    def snapshot(*, repo_id, revision, cache_dir, force_download, token, local_files_only):
        calls.append((repo_id, revision))
        return str(path)

    monkeypatch.setattr(module, "hf_hub_download", Mock(return_value=str(path / "config.json")))
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    cfg = PreTrainedConfig.from_pretrained("org/native", revision="release", cli_overrides=["--device=cpu"])
    assert cfg.pretrained_path == path
    assert calls == [("org/native", commit)]


@pytest.mark.parametrize(
    "offline,explicit,expected", [(True, False, True), (False, True, True), (False, False, False)]
)
def test_raw_snapshot_honors_hub_offline_mode(tmp_path, monkeypatch, offline, explicit, expected):
    import lerobot.configs.policies as module

    commit = "c" * 40
    (tmp_path / "snapshots").mkdir()
    path = raw_checkpoint(tmp_path / "snapshots" / commit)
    monkeypatch.setattr(module.hub_constants, "HF_HUB_OFFLINE", offline)
    monkeypatch.setattr(module, "hf_hub_download", Mock(return_value=str(path / "config.json")))
    snapshot = Mock(return_value=str(path))
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    cfg = PreTrainedConfig.from_pretrained(
        "org/native", local_files_only=explicit, cli_overrides=["--device=cpu"]
    )
    assert cfg.pretrained_path == path
    assert snapshot.call_args.kwargs["local_files_only"] is expected
    assert snapshot.call_args.kwargs["revision"] == commit
