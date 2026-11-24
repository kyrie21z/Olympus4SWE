"""Shared HTTP transport helpers with TCP keep-alive and long read timeouts."""

from __future__ import annotations

import socket
import ssl
from typing import Any

import httpx

DEFAULT_READ_TIMEOUT_SECONDS = 300.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_TCP_KEEPALIVE_IDLE_SECONDS = 20
DEFAULT_TCP_KEEPALIVE_INTERVAL_SECONDS = 10
DEFAULT_TCP_KEEPALIVE_COUNT = 6
_DEFAULT_LIMITS = httpx.Limits()


def tcp_keepalive_socket_options(
    *,
    idle_seconds: int = DEFAULT_TCP_KEEPALIVE_IDLE_SECONDS,
    interval_seconds: int = DEFAULT_TCP_KEEPALIVE_INTERVAL_SECONDS,
    count: int = DEFAULT_TCP_KEEPALIVE_COUNT,
) -> list[tuple[int, int, int]]:
    """Build portable TCP keep-alive socket options for ``httpx.HTTPTransport``.

    Short-interval keep-alive probes (15-30s) keep intermediate NATs and cloud
    gateways from silently dropping long-lived reasoning/streaming connections
    during silent thinking windows.
    """
    options: list[tuple[int, int, int]] = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    tcp_idle = getattr(socket, "TCP_KEEPIDLE", None)
    if tcp_idle is None:
        # macOS exposes TCP_KEEPALIVE instead of Linux's TCP_KEEPIDLE.
        tcp_idle = getattr(socket, "TCP_KEEPALIVE", None)
    if tcp_idle is not None:
        options.append((socket.IPPROTO_TCP, int(tcp_idle), max(1, int(idle_seconds))))
    tcp_intvl = getattr(socket, "TCP_KEEPINTVL", None)
    if tcp_intvl is not None:
        options.append((socket.IPPROTO_TCP, int(tcp_intvl), max(1, int(interval_seconds))))
    tcp_cnt = getattr(socket, "TCP_KEEPCNT", None)
    if tcp_cnt is not None:
        options.append((socket.IPPROTO_TCP, int(tcp_cnt), max(1, int(count))))
    return options


class _KeepAliveHttpxClient(httpx.Client):
    """``httpx.Client`` that applies TCP keep-alive to direct and proxied transports."""

    def __init__(
        self,
        *,
        socket_options: list[tuple[int, int, int]],
        **kwargs: Any,
    ) -> None:
        self._keepalive_socket_options = list(socket_options)
        super().__init__(**kwargs)

    def _init_transport(
        self,
        verify: ssl.SSLContext | str | bool = True,
        cert: Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: httpx.Limits = _DEFAULT_LIMITS,
        transport: httpx.BaseTransport | None = None,
    ) -> httpx.BaseTransport:
        if transport is not None:
            return transport
        return httpx.HTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            socket_options=self._keepalive_socket_options,
        )

    def _init_proxy_transport(
        self,
        proxy: httpx.Proxy,
        verify: ssl.SSLContext | str | bool = True,
        cert: Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: httpx.Limits = _DEFAULT_LIMITS,
    ) -> httpx.BaseTransport:
        return httpx.HTTPTransport(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            proxy=proxy,
            socket_options=self._keepalive_socket_options,
        )


def build_keepalive_httpx_client(
    *,
    timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
    follow_redirects: bool = False,
) -> httpx.Client:
    """Create an ``httpx.Client`` with TCP keep-alive enabled and proxy env preserved."""
    read_timeout = max(1.0, float(timeout))
    connect_timeout = min(DEFAULT_CONNECT_TIMEOUT_SECONDS, read_timeout)
    return _KeepAliveHttpxClient(
        socket_options=tcp_keepalive_socket_options(),
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        follow_redirects=follow_redirects,
    )


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "DEFAULT_TCP_KEEPALIVE_COUNT",
    "DEFAULT_TCP_KEEPALIVE_IDLE_SECONDS",
    "DEFAULT_TCP_KEEPALIVE_INTERVAL_SECONDS",
    "build_keepalive_httpx_client",
    "tcp_keepalive_socket_options",
]
