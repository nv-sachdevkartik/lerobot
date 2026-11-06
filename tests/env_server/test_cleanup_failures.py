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

"""Cleanup errors must reach callers without abandoning other owned resources."""

import asyncio
from queue import Empty
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.env_server import EnvClient, EnvServer, EnvServerConfig, RemoteVectorEnv, server as server_module
from lerobot.envs.utils import close_envs
from lerobot.sim import SimSettings
from lerobot.transport.protocol import ErrorCode, ProtocolError
from tests.fixtures.toy_env import ToyEnvConfig


def _raiser(error, events, name):
    def close():
        events.append(name)
        if error is not None:
            raise error

    return close


@pytest.mark.parametrize("through_eval", [False, True])
def test_live_close_failure_reaches_caller_and_server_remains_available(
    sim_server, monkeypatch, through_eval
):
    server, endpoint = sim_server
    client = EnvClient(endpoint)
    env = RemoteVectorEnv(client)
    env.reset()
    session_id = client.session_id
    session = server.sessions[session_id]
    original_close = session.env.close

    def fail_close():
        original_close()
        raise RuntimeError("confirmed simulator close failure")

    monkeypatch.setattr(session.env, "close", fail_close)
    with pytest.raises(ProtocolError, match="confirmed simulator close failure") as error:
        close_envs({"toy": {0: env}}) if through_eval else client.close()
    assert error.value.code is ErrorCode.EXECUTION
    assert session_id not in server.sessions
    assert client.session_id == ""
    other = EnvClient(endpoint)
    try:
        other.open(1)
    finally:
        other.close()


@pytest.mark.parametrize("cancel_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_session_cancellation_survives_later_cleanup_errors(cancel_type, caplog):
    events = []
    cancel = cancel_type("cancel close")
    session = server_module._Session.__new__(server_module._Session)
    session.env = SimpleNamespace(close=_raiser(cancel, events, "env"))
    session.presence = SimpleNamespace(close=_raiser(RuntimeError("presence failed"), events, "presence"))
    session.token = SimpleNamespace(undeclare=_raiser(RuntimeError("token failed"), events, "token"))
    with pytest.raises(cancel_type) as error:
        session.close()
    assert error.value is cancel
    assert events == ["env", "presence", "token"]
    assert "presence failed" in caplog.text and "token failed" in caplog.text


def test_close_envs_attempts_nested_collections_and_propagates_errors(caplog):
    events = []
    failure = RuntimeError("first environment failed")
    failing = SimpleNamespace(close=_raiser(failure, events, "first"))
    healthy = SimpleNamespace(close=_raiser(None, events, "second"))
    with pytest.raises(RuntimeError) as error:
        close_envs({"suite": [failing, {"nested": healthy}]})
    assert error.value is failure
    assert events == ["first", "second"]
    assert "first environment failed" in caplog.text


def test_close_envs_cancellation_still_attempts_remaining_resources():
    events = []
    cancel = KeyboardInterrupt("cancel envs")
    with pytest.raises(KeyboardInterrupt) as error:
        close_envs(
            [
                SimpleNamespace(close=_raiser(cancel, events, "first")),
                SimpleNamespace(close=_raiser(RuntimeError("second failed"), events, "second")),
                SimpleNamespace(close=_raiser(None, events, "third")),
            ]
        )
    assert error.value is cancel
    assert events == ["first", "second", "third"]


@pytest.mark.parametrize(
    "query_error",
    [ProtocolError(ErrorCode.EXECUTION, "remote close failed"), KeyboardInterrupt("cancel query")],
)
def test_client_reports_close_error_and_clears_state_when_transport_close_fails(query_error):
    client = EnvClient("tcp/127.0.0.1:1")
    client.session_id, client.instance_id = "session", "instance"
    client.deployment = "toy"
    client.num_envs, client.generation = 3, 2
    client.descriptor = object()
    client._query = MagicMock(side_effect=query_error)
    client.transport = MagicMock()
    client.transport.close.side_effect = RuntimeError("transport close failed")
    expected = KeyboardInterrupt if isinstance(query_error, KeyboardInterrupt) else RuntimeError
    with pytest.raises(expected) as error:
        client.close()
    assert not client.session_id and not client.instance_id and client.descriptor is None
    assert client.num_envs == 0 and client.generation == 0 and not client.failed
    client.transport.close.assert_called_once()
    if isinstance(query_error, KeyboardInterrupt):
        assert error.value is query_error
    else:
        assert error.value.__cause__ is not None
        assert query_error in error.value.__cause__.exceptions


def test_reaper_keeps_reaping_after_an_ordinary_close_failure(caplog):
    server = EnvServer(EnvServerConfig(sim=SimSettings(**ToyEnvConfig().sim_settings()), deployment="toy"))
    events = []
    for name, error in [("first", RuntimeError("reaped close failed")), ("second", None)]:
        presence = MagicMock(dropped=1)
        presence.get.side_effect = Empty
        server.sessions[name] = SimpleNamespace(
            presence=presence, opened_at=0, presence_established=True, close=_raiser(error, events, name)
        )
    server._reap_sessions()
    assert events == ["first", "second"] and server.sessions == {}
    assert "reaped close failed" in caplog.text


@pytest.mark.parametrize("target", ["collection", "client"])
def test_cleanup_does_not_replace_an_active_body_cancellation(target):
    events = []
    cancel = KeyboardInterrupt("body cancelled")
    failure = RuntimeError("cleanup failed")
    if target == "collection":

        def cleanup():
            close_envs([SimpleNamespace(close=_raiser(failure, events, "env"))])
    else:
        client = EnvClient("tcp/127.0.0.1:1")
        client.transport = MagicMock()
        client.transport.close.side_effect = failure
        cleanup = client.close
    with pytest.raises(KeyboardInterrupt) as error:
        try:
            raise cancel
        finally:
            cleanup()
    assert error.value is cancel


@pytest.mark.parametrize("provider_failure", [False, True])
@pytest.mark.parametrize("cancel_in_body", [False, True])
def test_server_cancellation_survives_all_cleanup_and_provider_errors(
    monkeypatch, caplog, provider_failure, cancel_in_body
):
    from contextlib import contextmanager

    from tests.fixtures import toy_env

    events = []
    cancel = KeyboardInterrupt("server cancelled")

    @contextmanager
    def resource_context(*args, **kwargs):
        try:
            yield
        finally:
            events.append("provider")
            if provider_failure:
                raise RuntimeError("provider failed")

    monkeypatch.setattr(toy_env, "resource_context", resource_context, raising=False)
    server = EnvServer(EnvServerConfig(sim=SimSettings(**ToyEnvConfig().sim_settings()), deployment="toy"))
    server.transport = MagicMock()
    server.transport.close.side_effect = _raiser(RuntimeError("transport failed"), events, "transport")

    def serve():
        server.sessions = {
            "first": SimpleNamespace(
                close=_raiser(RuntimeError("first failed") if cancel_in_body else cancel, events, "first")
            ),
            "second": SimpleNamespace(close=_raiser(RuntimeError("second failed"), events, "second")),
        }
        server._startup_env = SimpleNamespace(
            close=_raiser(RuntimeError("startup failed"), events, "startup")
        )
        if cancel_in_body:
            raise cancel

    monkeypatch.setattr(server, "_serve", serve)
    with pytest.raises(KeyboardInterrupt) as error:
        server.serve()
    assert error.value is cancel
    assert events == ["first", "second", "startup", "transport", "provider"]
    assert server.sessions == {} and server._startup_env is None
    assert (
        "second failed" in caplog.text
        and "startup failed" in caplog.text
        and "transport failed" in caplog.text
    )


@pytest.mark.parametrize(
    "absence",
    [
        TimeoutError("no reply"),
        ConnectionError("connection lost"),
        ProtocolError(ErrorCode.STALE, "Unknown simulator session"),
    ],
)
def test_client_close_warns_for_an_absent_session_and_clears_state(absence, caplog):
    client = EnvClient("tcp/127.0.0.1:1")
    client.session_id, client.instance_id, client.deployment = "session", "instance", "toy"
    client.num_envs, client.generation = 3, 2
    client.descriptor = object()
    client._query = MagicMock(side_effect=absence)
    client.transport = MagicMock()
    client.close()
    client.transport.close.assert_called_once()
    assert not client.session_id and not client.instance_id and client.descriptor is None
    assert client.num_envs == 0 and client.generation == 0 and not client.failed
    assert "Could not close the env server session cleanly" in caplog.text
