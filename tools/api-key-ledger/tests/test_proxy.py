import json
import logging
import socket
import sqlite3
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from key_ledger.config import Budget, Config, Credential
from key_ledger.ledger import Ledger
from key_ledger.proxy import _cooldown, create_app

BASE = "http://127.0.0.1:8787"
local_client = partial(TestClient, base_url=BASE, client=("127.0.0.1", 50001))
PROMPT = "unique prompt text for privacy verification"
COMPLETION = "unique completion text for privacy verification"
OPENAI_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "prompt_tokens_details": {"cached_tokens": 30},
}
ANTHROPIC_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_read_input_tokens": 30,
    "cache_creation_input_tokens": 10,
}


class MockStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.ReadError("untrusted transport exception detail")


def sse(payload: dict[str, Any], end: bytes = b"\n\n") -> bytes:
    return b"data: " + json.dumps(payload).encode() + end


@pytest.mark.parametrize("dialect", ["openai", "anthropic"])
@pytest.mark.parametrize("streaming", [False, True])
def test_protocols_and_accounting(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    dialect: str,
    streaming: bool,
) -> None:
    openai = dialect == "openai"
    path = "/v1/chat/completions" if openai else "/p/messages/v1/v1/messages"
    usage_data = OPENAI_USAGE if openai else ANTHROPIC_USAGE
    request_body = {
        "model": "model-alias",
        "stream": streaming,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    observed: list[httpx.Request] = []
    if openai:
        frames = (
            sse({"model": "mock-model", "choices": [{"delta": {"content": COMPLETION}}]})
            + sse({"model": "mock-model", "usage": usage_data}, b"\r\n\r\n")
            + b"data: [DONE]\r\n\r\n"
        )
    else:
        frames = (
            sse(
                {
                    "type": "message_start",
                    "message": {
                        "model": "mock-model",
                        "usage": {**usage_data, "output_tokens": 2},
                    },
                }
            )
            + sse({"type": "content_block_delta", "delta": {"text": COMPLETION}})
            + sse({"type": "message_delta", "usage": {"output_tokens": 20}})
            + sse({"type": "message_stop"})
        )

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if streaming:
            chunks = [frames[index : index + 1] for index in range(len(frames))]
            return httpx.Response(
                200, stream=MockStream(chunks), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200, json={"model": "mock-model", "usage": usage_data, "content": COMPLETION}
        )

    headers = {
        "x-ledger-sticky-key": "case-42",
        "x-ledger-tag": "pytest",
        "authorization": "Bearer placeholder",
        "anthropic-beta": "test-beta",
    }
    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        result = client.post(path, json=request_body, headers=headers)
        assert result.status_code == 200
        assert (result.content == frames) if streaming else result.json()["content"] == COMPLETION
        (event,) = client.get("/api/events").json()["events"]
        assert event["model"] == "mock-model"
        assert event["sticky_key"] != "case-42"
        assert event["tag"] == "pytest"
        assert event["outcome"] == "success"
        assert event["input_tokens"] == (100 if openai else 140)
        assert event["output_tokens"] == 20
        assert event["cached_input_tokens"] == 30
        assert event["cache_write_tokens"] == (0 if openai else 10)
        assert event["cost"] == ("0.000235" if openai else "0.000325")
        assert event["price_snapshot"]["cached_input_per_1m"] == "0.5"
        (stats,) = client.get("/api/stats", params={"by": "tag"}).json()["rows"]
        assert stats["group"] == "pytest"
        assert stats["cost"] == event["cost"]
        assert client.get("/health").json() == {"ok": True}
    (sent,) = observed
    actual = json.loads(sent.content)
    assert actual["messages"] == request_body["messages"]
    assert ("stream_options" in actual) == (openai and streaming)
    assert "x-ledger-tag" not in sent.headers and "x-ledger-sticky-key" not in sent.headers
    if openai:
        assert sent.url.path == "/gateway/v1/chat/completions"
        assert sent.headers["authorization"] == f"Bearer {credentials['chat', 'a'].value}"
    else:
        assert sent.url.path == "/v1/messages"
        assert sent.headers["x-api-key"] == credentials["messages", "c"].value
        assert sent.headers["anthropic-beta"] == "test-beta"
        assert "authorization" not in sent.headers


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_failover_repins_and_logs_each_attempt(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    status: int,
) -> None:
    observed = []
    first = f"Bearer {credentials['chat', 'a'].value}"

    def upstream(request: httpx.Request) -> httpx.Response:
        seen = request.headers["authorization"]
        observed.append(seen)
        if seen == first:
            return httpx.Response(status, text=PROMPT, headers={"retry-after": "190"})
        return httpx.Response(200, json={"model": "mock-model", "usage": OPENAI_USAGE})

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        for _ in range(2):
            result = client.post(
                "/v1/chat/completions",
                json={"model": "mock-model"},
                headers={"x-ledger-sticky-key": "same-case"},
            )
            assert result.status_code == 200
        slots = {row["key_alias"]: row for row in client.get("/api/keys").json()["keys"]}
        assert slots["a"]["status"] == ("cooldown" if status == 429 else "quarantined")
        assert bool(slots["a"]["cooldown_until"]) == (status == 429)
        entries = list(reversed(client.get("/api/events").json()["events"]))
        assert [row["http_status"] for row in entries] == [status, 200, 200]
        assert [row["key_alias"] for row in entries] == ["a", "b", "b"]
        assert len({row["sticky_key"] for row in entries}) == 1
    assert observed[1] == observed[2] != first
    assert _cooldown("190", 60) == 190
    assert _cooldown("Fri, 01 Jan 2100 00:00:00 GMT", 60) > 60
    assert _cooldown("malformed", 60) == 60


def test_pool_exhaustion_and_budget_transition(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
) -> None:
    pool = config.pools["chat"]
    pool.keys[0].budget = Budget(total="0.000235")
    observed: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"]
        observed.append(key)
        if key == f"Bearer {credentials['chat', 'b'].value}":
            return httpx.Response(401)
        return httpx.Response(200, json={"model": "mock-model", "usage": OPENAI_USAGE})

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        assert client.post("/v1/chat/completions", json={"model": "mock-model"}).status_code == 200
        for _ in range(2):
            failed = client.post("/v1/chat/completions", json={"model": "mock-model"})
            assert failed.status_code == 503
            error = failed.json()["error"]
            assert error["code"] == "pool_exhausted"
            assert error["health"] == {"budget_exceeded": 1, "quarantined": 1}
        assert len(client.get("/api/events").json()["events"]) == 2
    assert len(observed) == 2


def test_midstream_partial_without_retry(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
) -> None:
    called = 0

    def upstream(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called += 1
        message = {
            "type": "message_start",
            "message": {
                "model": "mock-model",
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 30},
            },
        }
        return httpx.Response(
            200,
            stream=MockStream([sse(message)], fail=True),
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/messages", json={"model": "mock-model", "stream": True})
        assert response.status_code == 200
        (event,) = client.get("/api/events").json()["events"]
        assert event["outcome"] == "partial"
        assert event["http_status"] == 200
        assert event["input_tokens"] == 130
        assert event["output_tokens"] is None
        assert event["pricing_status"] == "partial"
        assert event["cost"] == "0.000215"
    assert called == 1


def test_allowlisted_origin_headers_and_no_persistence(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[httpx.Request] = []
    token = credentials["chat", "a"].value
    caller_key = "caller-api-token-093150"

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"model": token, "usage": OPENAI_USAGE, "text": COMPLETION},
            headers={"set-cookie": "private=upstream-secret-cookie"},
        )

    caplog.set_level(logging.INFO)
    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        for headers, expected in [
            ({"host": "foreign.example.test"}, 400),
            ({"origin": "http://localhost:8888"}, 403),
            ({"sec-fetch-site": "cross-site"}, 403),
            ({"content-type": "text/plain"}, 415),
        ]:
            assert (
                client.post("/v1/chat/completions", json={}, headers=headers).status_code
                == expected
            )
        assert client.get("/p/chat/v1/messages").status_code == 405
        assert client.post("/p/chat/v1/messages", json={}).status_code == 404
        assert client.post("/api/keys", json={}).status_code == 405
        assert not calls
        for label in (token, caller_key):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "mock-model", "messages": [{"role": "user", "content": PROMPT}]},
                headers={
                    "authorization": f"Bearer {caller_key}",
                    "x-api-key": caller_key,
                    "x-ledger-tag": label,
                    "x-ledger-sticky-key": label,
                    "cookie": "browser=private-cookie",
                    "x-forwarded-host": "foreign.example.test",
                },
            )
            assert response.status_code == 200
            assert "set-cookie" not in response.headers
        assert len(calls) == 2
        for request in calls:
            assert "cookie" not in request.headers and "x-api-key" not in request.headers
            assert "x-forwarded-host" not in request.headers
        api = "".join(client.get(url).text for url in ("/api/stats", "/api/events", "/api/keys"))
        assert client.get("/api/events?limit=-1").status_code == 400
        assert client.get("/dashboard").headers["content-security-policy"]
        assert client.get("/dashboard.js").status_code == 200
        assert client.get("/dashboard.css").status_code == 200
    with local_client(app, client=("192.0.2.5", 50001)) as remote:
        assert remote.get("/health").json()["error"]["code"] == "remote_client"

    with sqlite3.connect(config.database) as db:
        dump = "\n".join(db.iterdump())
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("DELETE FROM ledger_entries")
    for private in (token, caller_key, PROMPT, COMPLETION, "upstream-secret-cookie"):
        assert private not in dump and private not in api and private not in caplog.text
        for file in config.database.parent.glob("*.sqlite3*"):
            assert private.encode() not in Path(file).read_bytes()
    assert {event["pricing_status"] for event in Ledger(config).events()} == {"unpriced"}


def test_stream_options_can_be_disabled_and_redirects_are_never_followed(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
) -> None:
    config.pools["chat"].inject_stream_usage = False
    observed: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(307, headers={"location": "https://other.example.test/"})

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "mock-model", "stream": True})
        assert response.status_code == 502
        assert "location" not in response.headers
    (sent,) = observed
    assert "stream_options" not in sent.content.decode()


@pytest.mark.parametrize("placeholder", ["test", "4"])
def test_short_caller_key_preserves_model_tag_and_cost(
    config: Config, credentials: dict[tuple[str, str], Credential], placeholder: str
) -> None:
    model = "claude-4-7-sonnet-latest"
    tag = "pytest-4"
    pool = config.pools["chat"]
    pool.pricing[model] = pool.pricing["mock-model"]

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": json.loads(request.content)["model"], "usage": OPENAI_USAGE}
        )

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        for label, model_id in [(tag, model), (placeholder, placeholder)]:
            response = client.post(
                "/v1/chat/completions",
                json={"model": model_id},
                headers={"authorization": f"Bearer {placeholder}", "x-ledger-tag": label},
            )
            assert response.status_code == 200
        exact, preserved = client.get("/api/events").json()["events"]
    assert preserved["model"] == model and preserved["tag"] == tag
    assert preserved["pricing_status"] == "priced" and preserved["cost"] == "0.000235"
    assert exact["model"] is None and exact["tag"] is None


@pytest.mark.parametrize("status", [400, 404, 422, 500])
def test_upstream_error_keeps_diagnostics_without_storing_body(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    provider_key = credentials["chat", "a"].value
    caller_key = "caller-credential-12345"
    message = "invalid model diagnostic for this request"
    payload = {
        "error": {
            "code": status,
            "message": f"{message}: {provider_key}",
            "details": [{caller_key: [f"Bearer {caller_key}", "pytest"]}],
        },
    }
    # JSON escapes must be decoded before credential redaction.
    escaped_key = "".join(f"\\u{ord(char):04x}" for char in provider_key)
    encoded = json.dumps(payload).replace(provider_key, escaped_key).encode()
    calls: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            stream=MockStream([encoded[:20], encoded[20:]]),
            headers={"content-type": "application/json", "set-cookie": "discard-this"},
        )

    caplog.set_level(logging.INFO)
    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock-model"},
            headers={"authorization": f"Bearer {caller_key}", "x-api-key": "test"},
        )
        assert response.status_code == status
        error = response.json()["error"]
        assert error["code"] == status
        assert error["message"] == f"{message}: [redacted]"
        assert error["details"] == [{"[redacted]": ["Bearer [redacted]", "pytest"]}]
        assert "set-cookie" not in response.headers
        assert len(calls) == 1
        events = client.get("/api/events")
        (event,) = events.json()["events"]
        assert event["outcome"] == "upstream_error" and event["http_status"] == status

    with sqlite3.connect(config.database) as db:
        dump = "\n".join(db.iterdump())
    for private in (provider_key, caller_key, message):
        assert private not in events.text and private not in dump and private not in caplog.text


def test_html_upstream_error_is_redacted_plain_text(
    config: Config, credentials: dict[tuple[str, str], Credential]
) -> None:
    key = credentials["messages", "c"].value

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            text=f"\x1b<script>invalid schema {key}</script>",
            headers={"content-type": "text/html"},
        )

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        response = client.post("/v1/messages", json={"model": "mock-model"})
    assert response.status_code == 422
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.text == "<script>invalid schema [redacted]</script>"


@pytest.mark.parametrize(
    "body",
    [b"x" * (64 * 1024) + b"discarded tail", b"4 " * 9000],
    ids=["oversized-upstream", "redaction-expansion"],
)
def test_error_body_size_is_bounded_before_and_after_redaction(
    config: Config, credentials: dict[tuple[str, str], Credential], body: bytes
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, stream=MockStream([body[:30_000], body[30_000:]]))

    app = create_app(config, credentials, transport=httpx.MockTransport(upstream))
    with local_client(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock-model"},
            headers={"authorization": "Bearer 4"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "upstream_error"
    assert len(response.content) <= 64 * 1024


def test_upstream_client_uses_long_read_timeout_and_keepalive(
    config: Config,
    credentials: dict[tuple[str, str], Credential],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    app = create_app(config, credentials)
    with local_client(app):
        client = app.state.client
        assert client.timeout == httpx.Timeout(300.0, connect=30.0, pool=10.0)
        assert client._trust_env is False
        transport = client._transport
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        options = transport._pool._socket_options
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
        for name, seconds in (
            ("TCP_KEEPIDLE", 20),
            ("TCP_KEEPALIVE", 20),
            ("TCP_KEEPINTVL", 10),
            ("TCP_KEEPCNT", 6),
        ):
            option = getattr(socket, name, None)
            if option is not None:
                assert (socket.IPPROTO_TCP, int(option), seconds) in options


def test_upstream_read_timeout_returns_502_and_records_event(
    config: Config, credentials: dict[tuple[str, str], Credential]
) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow upstream", request=request)

    injected = httpx.MockTransport(upstream)
    app = create_app(config, credentials, transport=injected)
    with local_client(app) as client:
        assert app.state.client._transport is injected
        response = client.post("/v1/chat/completions", json={"model": "mock-model"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "upstream_error"
        (event,) = client.get("/api/events").json()["events"]
        assert event["outcome"] == "upstream_error"
