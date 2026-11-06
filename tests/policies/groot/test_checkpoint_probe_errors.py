# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""A failed checkpoint lookup must never select a different model loading path."""

from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from lerobot.policies.groot import modeling_groot
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.pretrained import PreTrainedPolicy


def _lookup_error(error_type):
    if issubclass(error_type, HfHubHTTPError):
        status = 403 if issubclass(error_type, GatedRepoError) else 503
        if issubclass(error_type, (RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError)):
            status = 404
        response = httpx.Response(status, request=httpx.Request("HEAD", "https://huggingface.co/test"))
        return error_type("checkpoint lookup failed", response=response)
    return error_type("checkpoint lookup failed")


@pytest.mark.parametrize("local_files_only", [False, True])
@pytest.mark.parametrize(
    "error_type",
    [
        ConnectionError,
        TimeoutError,
        HfHubHTTPError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        KeyboardInterrupt,
        SystemExit,
    ],
)
def test_unknown_checkpoint_layout_preserves_lookup_error(
    monkeypatch, tmp_path, error_type, local_files_only
):
    error = _lookup_error(error_type)
    lookup = Mock(side_effect=error)
    construct = Mock(side_effect=AssertionError("lookup failure selected base model"))
    monkeypatch.setattr(modeling_groot, "hf_hub_download", lookup)
    monkeypatch.setattr(GrootPolicy, "__init__", construct)
    with pytest.raises(error_type) as caught:
        GrootPolicy.from_pretrained(
            "test/GR00T-N1.7",
            local_files_only=local_files_only,
            revision="pinned",
            cache_dir=tmp_path,
            token="test-token",
            force_download=True,
        )
    assert caught.value is error
    construct.assert_not_called()
    assert lookup.call_args.kwargs == {
        "repo_id": "test/GR00T-N1.7",
        "filename": "model.safetensors",
        "revision": "pinned",
        "cache_dir": tmp_path,
        "token": "test-token",
        "local_files_only": local_files_only,
        "force_download": False,
    }


@pytest.mark.parametrize("local", [False, True])
def test_confirmed_single_file_checkpoint_uses_standard_loader(monkeypatch, tmp_path, local):
    checkpoint = tmp_path if local else "test/GR00T-N1.7"
    if local:
        (tmp_path / "model.safetensors").write_bytes(b"present")
    lookup = Mock(return_value="/cache/model.safetensors")
    loaded = object()
    parent = Mock(return_value=loaded)
    monkeypatch.setattr(modeling_groot, "hf_hub_download", lookup)
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", parent)
    assert GrootPolicy.from_pretrained(checkpoint, revision="pinned", strict=True) is loaded
    assert parent.call_args.kwargs["pretrained_name_or_path"] == checkpoint
    assert parent.call_args.kwargs["revision"] == "pinned"
    assert parent.call_args.kwargs["strict"] is True
    assert lookup.call_count == (0 if local else 1)


@pytest.mark.parametrize("local", [False, True])
def test_confirmed_single_file_absence_selects_base_model(monkeypatch, tmp_path, local):
    checkpoint = tmp_path if local else "test/GR00T-N1.7"
    lookup = Mock(side_effect=_lookup_error(RemoteEntryNotFoundError))
    selected = RuntimeError("base model construction reached")
    construct = Mock(side_effect=selected)
    monkeypatch.setattr(modeling_groot, "hf_hub_download", lookup)
    monkeypatch.setattr(GrootPolicy, "__init__", construct)
    with pytest.raises(RuntimeError) as caught:
        GrootPolicy.from_pretrained(checkpoint)
    assert caught.value is selected
    assert construct.call_args.args[0].base_model_path == str(checkpoint)
    assert lookup.call_count == (0 if local else 1)


@pytest.mark.parametrize("cached_layout", ["single_file", "absent", "unknown"])
def test_real_offline_hub_cache_selects_only_confirmed_layout(monkeypatch, tmp_path, cached_layout):
    # Exercise the actual public Hub cache resolver and hf_hub_download(local_files_only=True).
    revision = "a" * 40
    repo = tmp_path / "models--test--GR00T-N1.7"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text(revision)
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if cached_layout == "single_file":
        (snapshot / "model.safetensors").write_bytes(b"present")
    elif cached_layout == "absent":
        missing = repo / ".no_exist" / revision
        missing.mkdir(parents=True)
        (missing / "model.safetensors").touch()

    selected = RuntimeError("base model construction reached")
    construct = Mock(side_effect=selected)
    parent = Mock(return_value=object())
    monkeypatch.setattr(GrootPolicy, "__init__", construct)
    monkeypatch.setattr(PreTrainedPolicy, "from_pretrained", parent)
    kwargs = {"cache_dir": tmp_path, "local_files_only": True}
    if cached_layout == "single_file":
        assert GrootPolicy.from_pretrained("test/GR00T-N1.7", **kwargs) is parent.return_value
        construct.assert_not_called()
    elif cached_layout == "absent":
        with pytest.raises(RuntimeError) as caught:
            GrootPolicy.from_pretrained("test/GR00T-N1.7", **kwargs)
        assert caught.value is selected
        parent.assert_not_called()
    else:
        with pytest.raises(LocalEntryNotFoundError):
            GrootPolicy.from_pretrained("test/GR00T-N1.7", **kwargs)
        construct.assert_not_called()
        parent.assert_not_called()


@pytest.mark.parametrize("offline", [False, True])
def test_negative_cache_cannot_hide_online_lookup_failure(monkeypatch, offline):
    error = LocalEntryNotFoundError("metadata request failed")
    monkeypatch.setattr(modeling_groot, "hf_hub_download", Mock(side_effect=error))
    monkeypatch.setattr(modeling_groot.constants, "HF_HUB_OFFLINE", offline)
    cache = Mock(return_value=modeling_groot._CACHED_NO_EXIST)
    monkeypatch.setattr(modeling_groot, "try_to_load_from_cache", cache)
    selected = RuntimeError("base model construction reached")
    construct = Mock(side_effect=selected)
    monkeypatch.setattr(GrootPolicy, "__init__", construct)
    with pytest.raises(RuntimeError if offline else LocalEntryNotFoundError) as caught:
        GrootPolicy.from_pretrained("test/GR00T-N1.7", revision="pinned")
    assert caught.value is (selected if offline else error)
    assert cache.call_count == int(offline)
    assert construct.call_count == int(offline)
