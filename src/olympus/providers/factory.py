"""Resolve registry data plus environment/CLI overrides into a live provider."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from olympus.providers.auth import (
    CredentialPool,
    EndpointTarget,
    PooledProvider,
    apply_url_rewrites,
    resolve_env_mapping,
)
from olympus.providers.dialects import (
    AnthropicMessagesDialect,
    Dialect,
    OpenAIChatCompat,
    OpenAIChatDialect,
)
from olympus.providers.registry import EndpointSpec, ProviderRegistry, ProviderSpec
from olympus.providers.types import LLMProvider, ProviderError

_DEFAULT_USER_PROVIDERS_PATH = Path("~/.olympus/providers.json")


def build_provider(
    *,
    provider: str = "auto",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    env_file: Path | None = None,
    providers_file: Path | None = None,
    user_providers_path: Path | None = _DEFAULT_USER_PROVIDERS_PATH,
    project_providers_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = 180.0,
) -> LLMProvider:
    """Construct a provider from a registry spec, environment, or ad-hoc base URL."""
    proc_env = os.environ if environ is None else environ
    registry = ProviderRegistry.load(
        user_providers_path=user_providers_path,
        project_providers_path=project_providers_path,
        explicit_providers_path=providers_file,
        environ=proc_env,
    )

    explicit_base = base_url.strip().rstrip("/") if base_url and base_url.strip() else None
    requested = provider.strip().lower() or "auto"
    if requested == "auto":
        requested = "openai" if explicit_base is not None else registry.choose_auto(proc_env)

    spec = registry.get(requested)
    if spec is None:
        available = ", ".join(registry.list_ids())
        raise ProviderError(
            f"Unknown provider '{provider}'. Registered providers: {available}. "
            "Pass --base-url for an ad-hoc OpenAI-compatible endpoint."
        )

    provider_env_file = env_file
    if provider_env_file is None and spec.env_file:
        provider_env_file = Path(spec.env_file)
    env = resolve_env_mapping(provider_env_file=provider_env_file, environ=proc_env)

    chosen_model = model or env.get("OLYMPUS_MODEL") or spec.default_model

    seed_keys: list[str] = []
    if api_key and api_key.strip():
        seed_keys.append(api_key.strip())
    if spec.id == "openai":
        for name in ("OLYMPUS_API_KEY", "OLYMPUS_API_KEYS"):
            seed_keys.extend(_split_keys(env.get(name, "")))

    resolved_base = explicit_base
    if resolved_base is None and spec.id == "openai":
        resolved_base = _clean(env.get("OLYMPUS_BASE_URL"))

    endpoints = _resolve_endpoints(
        spec=spec,
        env=env,
        base_override=resolved_base,
        seed_keys=seed_keys,
        allow_keyless=resolved_base is not None,
    )

    return PooledProvider(
        model_name=chosen_model,
        dialect=_build_dialect(spec),
        pool=CredentialPool(endpoints),
        timeout=timeout,
    )


def _build_dialect(spec: ProviderSpec) -> Dialect:
    raw_max_tokens = spec.compat.get("max_tokens")
    max_tokens = raw_max_tokens if isinstance(raw_max_tokens, int) else None
    if spec.dialect == "anthropic-messages":
        return AnthropicMessagesDialect(max_tokens=max_tokens if max_tokens is not None else 4096)
    return OpenAIChatDialect(
        compat=OpenAIChatCompat.from_mapping(spec.compat),
        max_tokens=max_tokens,
    )


def _resolve_endpoints(
    *,
    spec: ProviderSpec,
    env: Mapping[str, str],
    base_override: str | None,
    seed_keys: Sequence[str],
    allow_keyless: bool,
) -> list[EndpointTarget]:
    targets: list[EndpointTarget] = []
    shorthand_base = base_override or _env_value(env, spec.base_url_env) or _clean(spec.base_url)

    if spec.endpoints:
        for endpoint in spec.endpoints:
            raw_base = (
                base_override
                or endpoint.base_url
                or _env_value(env, endpoint.base_url_env)
                or shorthand_base
            )
            if not raw_base:
                continue
            url = apply_url_rewrites(raw_base, endpoint.url_suffix_rewrite)
            own_keys = _endpoint_keys(endpoint, env)
            keys = list(seed_keys) if seed_keys else own_keys
            headers = {**spec.headers, **endpoint.headers}
            targets.extend(_targets_for(url, keys, headers, allow_keyless=allow_keyless))
    elif shorthand_base:
        own_keys = _shorthand_keys(spec, env)
        keys = list(seed_keys) if seed_keys else own_keys
        targets.extend(
            _targets_for(shorthand_base, keys, dict(spec.headers), allow_keyless=allow_keyless)
        )

    return _dedupe_targets(targets)


def _endpoint_keys(endpoint: EndpointSpec, env: Mapping[str, str]) -> list[str]:
    return _keys_from_env(env, endpoint.api_key_env, endpoint.api_keys_env, endpoint.api_key)


def _shorthand_keys(spec: ProviderSpec, env: Mapping[str, str]) -> list[str]:
    return _keys_from_env(env, spec.api_key_env, spec.api_keys_env, None)


def _keys_from_env(
    env: Mapping[str, str],
    api_key_env: str | None,
    api_keys_env: str | None,
    literal: str | None,
) -> list[str]:
    keys: list[str] = []
    if literal and literal.strip():
        keys.append(literal.strip())
    keys.extend(_split_keys(_env_value(env, api_key_env) or ""))
    keys.extend(_split_keys(_env_value(env, api_keys_env) or ""))
    return _merge_keys((), keys)


def _targets_for(
    base_url: str,
    keys: Sequence[str],
    headers: Mapping[str, str],
    *,
    allow_keyless: bool,
) -> list[EndpointTarget]:
    url = base_url.strip().rstrip("/")
    if not url:
        return []
    if keys:
        return [
            EndpointTarget(api_key=key, base_url=url, headers=dict(headers)) for key in keys
        ]
    if allow_keyless:
        return [EndpointTarget(api_key="", base_url=url, headers=dict(headers))]
    return []


def _dedupe_targets(targets: Sequence[EndpointTarget]) -> list[EndpointTarget]:
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    deduped: list[EndpointTarget] = []
    for target in targets:
        signature = (target.api_key, target.base_url, tuple(sorted(target.headers.items())))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(target)
    return deduped


def _merge_keys(seed: Sequence[str], own: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for key in (*seed, *own):
        cleaned = key.strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged


def _split_keys(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_value(env: Mapping[str, str], name: str | None) -> str | None:
    if not name:
        return None
    return _clean(env.get(name))


def _clean(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


class OpenAICompatibleProvider(PooledProvider):
    """Convenience constructor for direct Python use without the registry."""

    def __init__(
        self,
        *,
        model: str,
        endpoints: Sequence[EndpointTarget],
        compat: OpenAIChatCompat | None = None,
        max_tokens: int | None = None,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(
            model_name=model,
            dialect=OpenAIChatDialect(
                compat=compat if compat is not None else OpenAIChatCompat(),
                max_tokens=max_tokens,
            ),
            pool=CredentialPool(endpoints),
            timeout=timeout,
        )


class AnthropicProvider(PooledProvider):
    """Convenience constructor for direct Python use without the registry."""

    def __init__(
        self,
        *,
        model: str,
        endpoints: Sequence[EndpointTarget],
        max_tokens: int = 4096,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(
            model_name=model,
            dialect=AnthropicMessagesDialect(max_tokens=max_tokens),
            pool=CredentialPool(endpoints),
            timeout=timeout,
        )
