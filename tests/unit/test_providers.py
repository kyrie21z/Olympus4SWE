"""Unit tests for the vendor-neutral provider compatibility layer."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from olympus.providers.auth import (
    CredentialPool,
    EndpointTarget,
    PooledProvider,
    apply_url_rewrites,
    read_env_file,
    resolve_env_mapping,
)
from olympus.providers.dialects import (
    AnthropicMessagesDialect,
    OpenAIChatCompat,
    OpenAIChatDialect,
)
from olympus.providers.factory import (
    AnthropicProvider,
    OpenAICompatibleProvider,
    build_provider,
)
from olympus.providers.registry import ProviderRegistry
from olympus.providers.types import (
    ChatMessage,
    HttpStatusError,
    LLMProvider,
    ProviderError,
    ToolCall,
    ToolSpec,
)

_FORBIDDEN_PROVIDER_IMPORTS = (
    "olympus.core",
    "olympus.tools",
    "olympus.harness",
    "olympus.stages",
    "olympus.cli",
)

# Split so this guard test does not match its own source file.
_VENDOR_MARKERS = (
    "bai" + "lian",
    "dash" + "scope",
    "bai" + "lian_api_key",
    "qwen" + "3.8",
)


def _write_providers(path: Path, providers: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"providers": providers}), encoding="utf-8")
    return path


def _build(
    tmp_path: Path,
    *,
    environ: Mapping[str, str],
    providers: Any | None = None,
    **kwargs: Any,
) -> LLMProvider:
    project_path = tmp_path / ".olympus" / "providers.json"
    if providers is not None:
        _write_providers(project_path, providers)
    return build_provider(
        user_providers_path=None,
        project_providers_path=project_path,
        environ=environ,
        **kwargs,
    )


def test_base_module_reexports_neutral_contracts() -> None:
    from olympus.providers import base

    for name in (
        "ChatMessage",
        "LLMProvider",
        "LLMResponse",
        "MockLLMProvider",
        "ToolCall",
        "ToolSpec",
        "UsageStats",
        "canonical_tool_specs",
    ):
        assert hasattr(base, name), name
    assert not hasattr(base, "serialize_tool_schemas_openai")


def test_read_env_file_parses_dotenv_syntax(tmp_path: Path) -> None:
    env_path = tmp_path / "provider.env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                'export PRIMARY_KEY="sk-one"',
                "SECONDARY_KEY='sk-two'",
                "BROKEN_LINE",
                "",
                "EMPTY_VALUE=",
            ]
        ),
        encoding="utf-8",
    )
    parsed = read_env_file(env_path)
    assert parsed == {
        "PRIMARY_KEY": "sk-one",
        "SECONDARY_KEY": "sk-two",
        "EMPTY_VALUE": "",
    }
    assert read_env_file(tmp_path / "absent.env") == {}


def test_resolve_env_mapping_precedence(tmp_path: Path) -> None:
    provider_file = tmp_path / "provider.env"
    provider_file.write_text("SHARED=provider\nPROVIDER_ONLY=1\n", encoding="utf-8")
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / ".env").write_text("SHARED=cwd\nCWD_ONLY=1\n", encoding="utf-8")
    override_file = tmp_path / "override.env"
    override_file.write_text("SHARED=override\nOVERRIDE_ONLY=1\n", encoding="utf-8")

    layered = resolve_env_mapping(
        provider_env_file=provider_file,
        cwd=cwd,
        environ={"OLYMPUS_ENV_FILE": str(override_file)},
    )
    assert layered["SHARED"] == "override"
    assert layered["PROVIDER_ONLY"] == "1"
    assert layered["CWD_ONLY"] == "1"
    assert layered["OVERRIDE_ONLY"] == "1"

    local = resolve_env_mapping(provider_env_file=provider_file, cwd=cwd, environ={})
    assert local["SHARED"] == "cwd"

    process = resolve_env_mapping(
        provider_env_file=provider_file,
        cwd=cwd,
        environ={"SHARED": "process", "OLYMPUS_ENV_FILE": str(override_file)},
    )
    assert process["SHARED"] == "process"


def test_apply_url_rewrites_is_suffix_based() -> None:
    rewrites = {"/api/v1": "/compatible-mode/v1"}
    assert (
        apply_url_rewrites("https://host.example.com/api/v1/", rewrites)
        == "https://host.example.com/compatible-mode/v1"
    )
    assert (
        apply_url_rewrites("https://host.example.com/v1", rewrites)
        == "https://host.example.com/v1"
    )
    assert apply_url_rewrites("https://host.example.com/v1/", {}) == "https://host.example.com/v1"


def test_registry_loads_dict_and_list_formats(tmp_path: Path) -> None:
    user_path = tmp_path / "user.json"
    _write_providers(
        user_path,
        {
            "gateway": {
                "dialect": "openai-chat",
                "aliases": ["GW"],
                "default_model": "gateway-model",
                "endpoints": [
                    {
                        "base_url": "https://gateway.example.com/api/v1",
                        "api_key_env": "GATEWAY_API_KEY",
                        "url_suffix_rewrite": {"/api/v1": "/v1"},
                        "headers": {"X-Tenant": "alpha"},
                    }
                ],
            }
        },
    )
    project_path = tmp_path / "project.json"
    _write_providers(
        project_path,
        [
            {"id": "local", "default_model": "local-model", "base_url": "http://127.0.0.1:8000/v1"},
            {"id": "second", "default_model": "second-model", "default": True},
        ],
    )

    registry = ProviderRegistry.load(
        user_providers_path=user_path,
        project_providers_path=project_path,
        environ={},
    )
    spec = registry.get("gateway")
    assert spec is not None
    assert spec.default_model == "gateway-model"
    assert spec.endpoints[0].url_suffix_rewrite == {"/api/v1": "/v1"}
    assert registry.get("gw") is spec
    assert registry.list_ids() == ["anthropic", "gateway", "local", "openai", "second"]
    assert registry.choose_auto({}) == "second"


def test_registry_external_spec_overrides_builtin(tmp_path: Path) -> None:
    user_path = tmp_path / "user.json"
    _write_providers(
        user_path,
        [
            {
                "id": "OpenAI",
                "default_model": "proxy-model",
                "base_url": "https://proxy.example.com/v1",
                "api_key_env": "PROXY_API_KEY",
                "compat": {"max_tokens_field": "max_completion_tokens"},
            }
        ],
    )
    registry = ProviderRegistry.load(
        user_providers_path=user_path,
        project_providers_path=tmp_path / "missing.json",
        environ={},
    )
    spec = registry.get("openai")
    assert spec is not None
    assert spec.id == "openai"
    assert spec.default_model == "proxy-model"
    assert spec.compat == {"max_tokens_field": "max_completion_tokens"}
    assert registry.get("openai-compatible") is None


def test_registry_project_default_beats_user_default(tmp_path: Path) -> None:
    user_path = tmp_path / "user.json"
    project_path = tmp_path / "project.json"
    _write_providers(
        user_path,
        [{"id": "user-default", "default_model": "u", "api_key_env": "U_KEY", "default": True}],
    )
    _write_providers(
        project_path,
        [
            {
                "id": "project-default",
                "default_model": "p",
                "api_key_env": "P_KEY",
                "default": True,
            }
        ],
    )
    registry = ProviderRegistry.load(
        user_providers_path=user_path,
        project_providers_path=project_path,
        environ={},
    )
    assert registry.choose_auto({}) == "project-default"


def test_registry_invalid_spec_does_not_leak_secret_values(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    _write_providers(
        bad_path,
        [{"id": "broken", "default_model": "m", "unexpected": "sk-should-not-leak"}],
    )
    with pytest.raises(ProviderError) as excinfo:
        ProviderRegistry.load(
            user_providers_path=bad_path,
            project_providers_path=tmp_path / "missing.json",
            environ={},
        )
    assert "unexpected" in str(excinfo.value)
    assert "sk-should-not-leak" not in str(excinfo.value)


def test_build_provider_auto_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    providers = [
        {
            "id": "gateway",
            "default_model": "gateway-model",
            "base_url": "https://gateway.example.com/v1",
            "api_key_env": "GATEWAY_API_KEY",
        },
        {
            "id": "fallback",
            "default_model": "fallback-model",
            "base_url": "https://fallback.example.com/v1",
            "api_key_env": "FALLBACK_API_KEY",
            "default": True,
        },
    ]

    explicit = _build(
        tmp_path,
        environ={"OLYMPUS_PROVIDER": "anthropic", "OPENAI_API_KEY": "sk-openai"},
        providers=providers,
    )
    assert explicit.model_name == "claude-3-7-sonnet-latest"

    ad_hoc = _build(
        tmp_path,
        environ={
            "OLYMPUS_BASE_URL": "https://adhoc.example.com/v1",
            "OLYMPUS_API_KEY": "sk-adhoc",
        },
        providers=providers,
    )
    assert ad_hoc.model_name == "gpt-4o-mini"
    assert ad_hoc.pool.endpoints == (
        EndpointTarget(api_key="sk-adhoc", base_url="https://adhoc.example.com/v1"),
    )

    keyless_ad_hoc = _build(
        tmp_path / "keyless",
        environ={"OLYMPUS_BASE_URL": "http://127.0.0.1:11434/v1"},
        providers=providers,
    )
    assert keyless_ad_hoc.pool.endpoints == (
        EndpointTarget(api_key="", base_url="http://127.0.0.1:11434/v1"),
    )

    by_process_key = _build(
        tmp_path,
        environ={"GATEWAY_API_KEY": "sk-gateway"},
        providers=providers,
    )
    assert by_process_key.model_name == "gateway-model"
    assert by_process_key.pool.endpoints[0].base_url == "https://gateway.example.com/v1"

    by_default = _build(tmp_path, environ={}, providers=providers)
    assert by_default.model_name == "fallback-model"
    assert by_default.pool.endpoints == ()

    fallback = _build(tmp_path / "clean", environ={})
    assert fallback.model_name == "gpt-4o-mini"


def test_build_provider_ad_hoc_base_url_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    provider = build_provider(
        base_url="http://127.0.0.1:8080/v1/",
        api_key="sk-local",
        user_providers_path=None,
        project_providers_path=tmp_path / "missing.json",
        env_file=None,
        environ={},
    )
    assert provider.model_name == "gpt-4o-mini"
    assert provider.pool.endpoints == (
        EndpointTarget(api_key="sk-local", base_url="http://127.0.0.1:8080/v1"),
    )

    keyless = build_provider(
        base_url="http://127.0.0.1:11434/v1",
        user_providers_path=None,
        project_providers_path=tmp_path / "missing.json",
        environ={},
    )
    assert keyless.pool.endpoints == (
        EndpointTarget(api_key="", base_url="http://127.0.0.1:11434/v1"),
    )


def test_build_provider_applies_endpoint_url_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    providers = [
        {
            "id": "gateway",
            "default_model": "gateway-model",
            "endpoints": [
                {
                    "base_url": "https://gateway.example.com/api/v1",
                    "api_key_env": "GATEWAY_API_KEY",
                    "url_suffix_rewrite": {"/api/v1": "/compatible-mode/v1"},
                }
            ],
        }
    ]
    provider = _build(tmp_path, environ={"GATEWAY_API_KEY": "sk-key"}, providers=providers)
    assert provider.pool.endpoints[0].base_url == (
        "https://gateway.example.com/compatible-mode/v1"
    )


def test_build_provider_unknown_name_lists_registry_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        _build(tmp_path, environ={}, provider="does-not-exist")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "openai" in message
    assert "anthropic" in message


def test_build_provider_without_credentials_fails_on_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    provider = _build(tmp_path, environ={})
    with pytest.raises(ProviderError, match="No credentials"):
        provider.complete(messages=[ChatMessage(role="user", content="hi")])


def test_credential_pool_failover_classification() -> None:
    pool = CredentialPool([])
    assert pool.should_failover(HttpStatusError(401, "u", "")) is True
    assert pool.should_failover(HttpStatusError(402, "u", "")) is True
    assert pool.should_failover(HttpStatusError(403, "u", "")) is True
    assert pool.should_failover(HttpStatusError(429, "u", "")) is True
    assert pool.should_failover(HttpStatusError(400, "u", "Insufficient quota")) is True
    assert pool.should_failover(HttpStatusError(400, "u", "exceeded balance")) is True
    assert pool.should_failover(HttpStatusError(400, "u", "invalid temperature")) is False
    assert pool.should_failover(HttpStatusError(404, "u", "quota")) is False
    assert pool.should_failover(HttpStatusError(500, "u", "quota")) is False


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "wire-model",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_pooled_provider_sticky_failover_across_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del json, timeout
        authorization = headers.get("Authorization", "")
        calls.append((url, authorization))
        if authorization == "Bearer sk-primary":
            return httpx.Response(429, json={"error": {"message": "Quota exceeded"}})
        if url.startswith("https://second.example.com/"):
            return httpx.Response(403, json={"error": {"message": "access denied"}})
        return _completion_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    pool = CredentialPool(
        [
            EndpointTarget(api_key="sk-primary", base_url="https://first.example.com/v1"),
            EndpointTarget(api_key="sk-backup", base_url="https://second.example.com/v1"),
            EndpointTarget(api_key="sk-final", base_url="https://third.example.com/v1"),
        ]
    )
    provider = PooledProvider(model_name="wire-model", dialect=OpenAIChatDialect(), pool=pool)

    first = provider.complete(messages=[ChatMessage(role="user", content="hi")])
    assert first.content == "OK"
    assert first.usage.input_tokens == 10
    assert first.raw_model == "wire-model"
    assert pool.active_index == 2
    assert len(calls) == 3

    # Sticky: the next call starts at the working endpoint, not the exhausted ones.
    second = provider.complete(messages=[ChatMessage(role="user", content="again")])
    assert second.content == "OK"
    assert len(calls) == 4
    assert calls[-1] == ("https://third.example.com/v1/chat/completions", "Bearer sk-final")


@pytest.mark.parametrize("status", [400, 404])
def test_pooled_provider_does_not_failover_on_client_errors(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    attempts: list[str] = []

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del headers, json, timeout
        attempts.append(url)
        return httpx.Response(status, json={"error": {"message": "invalid parameter"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    pool = CredentialPool(
        [
            EndpointTarget(api_key="sk-primary", base_url="https://first.example.com/v1"),
            EndpointTarget(api_key="sk-backup", base_url="https://second.example.com/v1"),
        ]
    )
    provider = PooledProvider(model_name="wire-model", dialect=OpenAIChatDialect(), pool=pool)
    with pytest.raises(HttpStatusError) as excinfo:
        provider.complete(messages=[ChatMessage(role="user", content="hi")])
    assert excinfo.value.status_code == status
    assert len(attempts) == 1
    assert pool.active_index == 0


def test_openai_dialect_default_compat_omits_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del url, timeout
        captured["headers"] = headers
        captured["body"] = json
        return _completion_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    dialect = OpenAIChatDialect()
    dialect.send(
        EndpointTarget(api_key="sk-key", base_url="https://host.example.com/v1", headers={}),
        model="wire-model",
        messages=[
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call-1", name="read_file", arguments={"path": "a"}),),
            )
        ],
    )
    body = captured["body"]
    assert captured["headers"]["Authorization"] == "Bearer sk-key"
    assert body["messages"][0]["content"] is None
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


def test_openai_dialect_compat_flags_shape_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del url, headers, timeout
        captured["body"] = json
        return _completion_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    dialect = OpenAIChatDialect(
        compat=OpenAIChatCompat(
            max_tokens_field="max_completion_tokens",
            include_tool_result_name=True,
            non_null_assistant_content=True,
        ),
        max_tokens=256,
    )
    tools = (ToolSpec(name="read_file", description="Read", parameters={"type": "object"}),)
    dialect.send(
        EndpointTarget(api_key="sk-key", base_url="https://host.example.com/v1"),
        model="wire-model",
        messages=[
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call-1", name="read_file", arguments={"path": "a"}),),
            ),
            ChatMessage(role="tool", content="body", tool_call_id="call-1", name="read_file"),
        ],
        tools=tools,
    )
    body = captured["body"]
    assert body["max_completion_tokens"] == 256
    assert body["messages"][0]["content"] == ""
    assert body["messages"][1]["name"] == "read_file"
    assert body["tools"][0]["function"]["name"] == "read_file"


def test_openai_chat_compat_extra_body_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del url, headers, timeout
        captured["body"] = json
        return _completion_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    compat = OpenAIChatCompat.from_mapping(
        {
            "non_null_assistant_content": True,
            "extra_body": {"enable_thinking": False, "model": "should-not-override"},
        }
    )
    dialect = OpenAIChatDialect(compat=compat)
    dialect.send(
        EndpointTarget(api_key="sk-key", base_url="https://host.example.com/v1"),
        model="wire-model",
        messages=[ChatMessage(role="user", content="hello")],
    )
    body = captured["body"]
    assert body["enable_thinking"] is False
    assert body["model"] == "wire-model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_dialect_serializes_messages_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del timeout
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return httpx.Response(
            200,
            json={
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "read_file",
                        "input": {"path": "a"},
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    dialect = AnthropicMessagesDialect(max_tokens=512)
    endpoint = EndpointTarget(
        api_key="sk-anthropic",
        base_url="https://anthropic.example.com/",
        headers={"X-Trace": "trace-1"},
    )
    tools = (ToolSpec(name="read_file", description="Read", parameters={"type": "object"}),)
    response = dialect.send(
        endpoint,
        model="claude-test",
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="tool-1", name="read_file", arguments={"path": "a"}),),
            ),
            ChatMessage(role="tool", content="file body", tool_call_id="tool-1"),
        ],
        tools=tools,
        temperature=0.2,
    )
    assert captured["url"] == "https://anthropic.example.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-anthropic"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["headers"]["X-Trace"] == "trace-1"

    body = captured["body"]
    assert body["max_tokens"] == 512
    assert body["system"] == "be brief"
    assert body["messages"][0] == {"role": "user", "content": "hi"}
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": "file body",
    }
    assert body["tools"][0]["input_schema"] == {"type": "object"}

    assert response.content == "hello"
    assert response.tool_calls[0].arguments == {"path": "a"}
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 4
    assert response.raw_model == "claude-test"


def test_anthropic_dialect_raises_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del headers, json, timeout
        return httpx.Response(401, json={"error": {"message": "invalid credential"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    dialect = AnthropicMessagesDialect()
    with pytest.raises(HttpStatusError) as excinfo:
        dialect.send(
            EndpointTarget(api_key="sk-secret-value", base_url="https://anthropic.example.com"),
            model="claude-test",
            messages=[ChatMessage(role="user", content="hi")],
        )
    error = excinfo.value
    assert error.status_code == 401
    assert error.url == "https://anthropic.example.com/v1/messages"
    assert "invalid credential" in error.body_preview
    assert "sk-secret-value" not in str(error)


def test_providers_package_does_not_import_core_layers() -> None:
    providers_root = Path(__file__).resolve().parents[2] / "src" / "olympus" / "providers"
    modules = sorted(providers_root.rglob("*.py"))
    assert modules, "providers package must contain modules"

    offenders: list[str] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported: Sequence[str] = ()
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            for module in imported:
                if any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in _FORBIDDEN_PROVIDER_IMPORTS
                ):
                    offenders.append(f"{path.relative_to(providers_root)}: {module}")
    assert offenders == []


def test_no_vendor_specific_references_in_source_or_tests() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    scanned = sorted(
        path
        for base in (repo_root / "src", repo_root / "tests")
        for path in base.rglob("*.py")
    )
    assert scanned, "expected Python sources to scan"

    offenders: list[str] = []
    for path in scanned:
        text = path.read_text(encoding="utf-8").lower()
        for marker in _VENDOR_MARKERS:
            if marker in text:
                offenders.append(f"{path.relative_to(repo_root)}: {marker}")
    assert offenders == []


def test_convenience_providers_delegate_to_dialects(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_post(
        url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        del json, timeout
        calls.append((url, headers.get("Authorization"), headers.get("x-api-key")))
        if url.endswith("/v1/messages"):
            return httpx.Response(
                200, json={"content": [{"type": "text", "text": "anthropic-ok"}]}
            )
        return _completion_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    openai_provider = OpenAICompatibleProvider(
        model="wire-model",
        endpoints=[EndpointTarget(api_key="sk-openai", base_url="https://openai.example.com/v1")],
    )
    anthropic_provider = AnthropicProvider(
        model="claude-test",
        endpoints=[EndpointTarget(api_key="sk-anthropic", base_url="https://anthropic.example.com")],
    )

    assert (
        openai_provider.complete(messages=[ChatMessage(role="user", content="hi")]).content == "OK"
    )
    assert anthropic_provider.complete(
        messages=[ChatMessage(role="user", content="hi")]
    ).content == "anthropic-ok"
    assert calls == [
        ("https://openai.example.com/v1/chat/completions", "Bearer sk-openai", None),
        ("https://anthropic.example.com/v1/messages", None, "sk-anthropic"),
    ]
