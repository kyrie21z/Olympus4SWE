"""Context trimming that shrinks already-seen tool outputs without hiding fresh evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from olympus.providers.base import ChatMessage

LONG_CHARS = 4000
KEEP_CHARS = 2000
TRIM_NOTE = (
    "\n\n[Cut from {total:,} characters after you read it. "
    "Full output is preserved in this run's trace.]"
)


def shrink_seen(
    messages: list[ChatMessage],
    *,
    long_chars: int = LONG_CHARS,
    keep_chars: int = KEEP_CHARS,
) -> list[ChatMessage]:
    """Shrink oversized tool outputs that were superseded by a later assistant turn.

    The newest tool results (those after the last assistant message) stay byte-for-byte
    intact so the model always sees the evidence it is currently reasoning over.
    """
    last_assistant_idx = -1
    for idx, msg in enumerate(messages):
        if msg.role == "assistant":
            last_assistant_idx = idx
    if last_assistant_idx < 0:
        return list(messages)

    shrunk: list[ChatMessage] = []
    for idx, msg in enumerate(messages):
        if (
            idx < last_assistant_idx
            and msg.role == "tool"
            and len(msg.content) > long_chars
            and "[Cut from " not in msg.content
        ):
            note = TRIM_NOTE.format(total=len(msg.content))
            shrunk.append(replace(msg, content=msg.content[:keep_chars] + note))
        else:
            shrunk.append(msg)
    return shrunk


def chain(
    *steps: Callable[[list[ChatMessage]], list[ChatMessage]],
) -> Callable[[list[ChatMessage]], list[ChatMessage]]:
    """Compose message-list transforms into a single left-to-right pipeline."""

    def run(messages: list[ChatMessage]) -> list[ChatMessage]:
        current = messages
        for step in steps:
            current = step(current)
        return current

    return run
