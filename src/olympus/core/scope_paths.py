"""Repo-relative scope path normalization and conservative overlap detection."""

from __future__ import annotations

import fnmatch
from collections.abc import Collection
from pathlib import PurePosixPath

_GLOB_CHARS = ("*", "?", "[")


def normalize_scope_path(raw: str) -> str:
    """Normalize a repo-relative POSIX scope pattern and reject escapes."""
    cleaned = raw.strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Empty write_scope path is not allowed")
    if cleaned.startswith("/"):
        raise ValueError(f"write_scope must be repo-relative, got absolute path: {raw}")
    pure = PurePosixPath(cleaned.rstrip("/") or ".")
    if ".." in pure.parts:
        raise ValueError(f"write_scope cannot contain '..': {raw}")
    norm = pure.as_posix()
    return "." if norm == "." else f"{norm}/" if cleaned.endswith("/") else norm


def _has_glob(text: str) -> bool:
    return any(char in text for char in _GLOB_CHARS)


def _glob_literal_prefix(pattern: str) -> str:
    cut = min((pattern.find(char) for char in _GLOB_CHARS if char in pattern), default=len(pattern))
    return pattern[:cut]


def _glob_static_prefix(pattern: str) -> str:
    head = _glob_literal_prefix(pattern)
    return head[: head.rfind("/") + 1] if "/" in head else ""


def _glob_paths_overlap(a: str, b: str) -> bool:
    a_glob, b_glob = _has_glob(a), _has_glob(b)
    if a_glob and b_glob:
        pa, pb = _glob_static_prefix(a), _glob_static_prefix(b)
        return pa == pb or pa.startswith(pb) or pb.startswith(pa)
    pattern, concrete = (a, b) if a_glob else (b, a)
    concrete = concrete.rstrip("/")
    if fnmatch.fnmatchcase(concrete, pattern):
        return True
    prefix = _glob_static_prefix(pattern)
    if prefix:
        pdir = prefix.rstrip("/")
        return concrete == pdir or concrete.startswith(prefix) or prefix.startswith(f"{concrete}/")
    lit = _glob_literal_prefix(pattern)
    return not lit or concrete.startswith(lit) or lit.startswith(concrete)


def paths_overlap(a_raw: str, b_raw: str) -> bool:
    """Return True if two normalized repo-relative paths, prefixes, or globs can overlap."""
    a = normalize_scope_path(a_raw).rstrip("/")
    b = normalize_scope_path(b_raw).rstrip("/")
    if a == "." or b == "." or a == b:
        return True
    if _has_glob(a) or _has_glob(b):
        return _glob_paths_overlap(a, b)
    return a.startswith(f"{b}/") or b.startswith(f"{a}/")


def scopes_overlap(scope_a: list[str], scope_b: list[str]) -> list[tuple[str, str]]:
    """Return all conflicting pairs between two write_scope lists."""
    return [(pa, pb) for pa in scope_a for pb in scope_b if paths_overlap(pa, pb)]


def overlap_messages(label: str, left: Collection[str], right: Collection[str]) -> list[str]:
    """Format one message per conflicting scope pair."""
    return [f"{label}: {a}<->{b}" for a, b in scopes_overlap(list(left), list(right))]


def normalized_scopes(raw_paths: Collection[str], label: str, issues: list[str]) -> list[str]:
    """Normalize scope paths, recording each rejection instead of raising."""
    paths: list[str] = []
    for raw in raw_paths:
        try:
            paths.append(normalize_scope_path(raw))
        except ValueError as exc:
            issues.append(f"{label} invalid path ({raw}): {exc}")
    return paths
