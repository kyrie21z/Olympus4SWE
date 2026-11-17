"""Non-blocking observer fan-out for live turn telemetry."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

Observer = Callable[[str, dict[str, Any]], None]


def compose_observers(*observers: Observer | None) -> Observer:
    """Fan out ``(kind, payload)`` to every observer; UI failures never break the loop."""
    active = tuple(observer for observer in observers if observer is not None)

    def composed(kind: str, payload: dict[str, Any]) -> None:
        for observer in active:
            with suppress(Exception):
                observer(kind, payload)

    return composed
