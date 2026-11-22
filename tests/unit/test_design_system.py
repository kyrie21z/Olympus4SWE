"""Deterministic design-system and TypeScript-only gates for the Ariadne web SPA.

web/src/styles/tokens.css is the single file allowed to name a colour. Every other
stylesheet, component, and HTML shell under web/ must read tokens only, zero .js/.jsx
source files may live under web/src/, and legacy vanilla console files must stay absent.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
WEB_SRC = WEB_DIR / "src"
TOKENS_CSS = WEB_SRC / "styles" / "tokens.css"
STATIC_DIR = REPO_ROOT / "src" / "olympus" / "web" / "static"

COLOUR = re.compile(r"(?<![&\w])#[0-9a-fA-F]{3,8}\b|\b(?:rgb|rgba|hsl|hsla)\(")
BUTTONS = re.compile(r"(?:^|[\s,>+~])(?:button|\.btn)\b")
GROUNDS = {
    "transparent",
    "none",
    "var(--surface-paper)",
    "var(--surface-raised)",
    "var(--surface-sunk)",
    "var(--surface-bg)",
}
SIZES = {f"var(--text-{size})" for size in ("xs", "sm", "base", "lg", "xl", "2xl")}
WEIGHTS = {"400", "500", "normal", "var(--font-weight-normal)", "var(--font-weight-medium)"}
SHAPES = {
    "0",
    "var(--radius)",
    "var(--radius-lg)",
    "var(--shape-chip)",
    "var(--shape-circle)",
    "var(--shape-pill)",
    "var(--shape-bubble)",
}


def _css_sources() -> list[Path]:
    return sorted(path for path in WEB_SRC.rglob("*.css") if path != TOKENS_CSS)


def _code_sources() -> list[Path]:
    return sorted([*WEB_SRC.rglob("*.ts"), *WEB_SRC.rglob("*.tsx")])


def _blocks(css: str) -> list[tuple[str, str]]:
    """(selector, body) for every innermost rule, comments removed."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return [(m.group(1).strip(), m.group(2)) for m in re.finditer(r"([^{}]*)\{([^{}]*)\}", css)]


def _decls(body: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in body.split(";"):
        if ":" in part:
            prop, value = part.split(":", 1)
            out.append((prop.strip().lower(), value.strip()))
    return out


def _declarations() -> list[tuple[str, str, str, str]]:
    """(relpath, selector, property, value) across every non-token stylesheet."""
    rows: list[tuple[str, str, str, str]] = []
    for path in _css_sources():
        rel = path.relative_to(WEB_SRC).as_posix()
        for selector, body in _blocks(path.read_text(encoding="utf-8")):
            for prop, value in _decls(body):
                rows.append((rel, selector, prop, value))
    return rows


def _web_source_stamp() -> str:
    """Mirror web/vite.config.ts `sourceStamp()` offline without Node."""
    files = [
        WEB_DIR / name
        for name in (
            "package.json",
            "pnpm-lock.yaml",
            ".pnpmfile.cjs",
            "tsconfig.json",
            "vite.config.ts",
            "index.html",
            "../src/olympus/web/openapi.json",
        )
    ]
    files += [path for path in WEB_SRC.rglob("*") if path.is_file()]
    records = sorted(
        (path.resolve().relative_to(WEB_DIR.resolve(), walk_up=True).as_posix(), path)
        for path in files
    )
    digest = hashlib.sha256()
    for rel, path in records:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_no_untyped_js_sources_or_legacy_shell_assets() -> None:
    assert not (REPO_ROOT / "web-graph").exists(), "web-graph/ must be consolidated into web/"
    js_sources = sorted([*WEB_SRC.rglob("*.js"), *WEB_SRC.rglob("*.jsx")])
    assert not js_sources, f"untyped JS sources forbidden under web/src/: {js_sources}"
    for retired in ("app.js", "ui.js", "style.css", "tokens.css", "graph"):
        assert not (STATIC_DIR / retired).exists(), f"retired legacy asset still exists: {retired}"


def test_no_colour_literals() -> None:
    """Every colour outside tokens.css must reference a CSS custom property."""
    sources = [*_css_sources(), *_code_sources(), WEB_DIR / "index.html"]
    assert sources, "web/src sources missing"
    for path in sources:
        found = COLOUR.findall(path.read_text(encoding="utf-8"))
        assert not found, (
            f"{path.relative_to(WEB_DIR)} holds raw colours — use var(--token): {found[:8]}"
        )
    found_decls = [(f, s, p, v) for f, s, p, v in _declarations() if COLOUR.search(v)]
    assert not found_decls, f"raw colours in declarations: {found_decls[:8]}"


def test_no_shadows_except_floating() -> None:
    allowed = (".floating-menu", "dialog[open]")
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "box-shadow"
        and value != "none"
        and not any(marker in selector for marker in allowed)
    ]
    assert not found, f"only a floating menu or an open dialog takes --shadow-md: {found}"
    wrong = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "box-shadow" and value not in ("none", "var(--shadow-md)")
    ]
    assert not wrong, f"box-shadow is none or var(--shadow-md): {wrong}"


def test_buttons_never_fill() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if BUTTONS.search(selector)
        and prop in ("background", "background-color")
        and value not in GROUNDS
    ]
    assert not found, f"buttons never fill: {found}"


def test_font_sizes_use_tokens() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font-size" and value not in SIZES | {"inherit"}
    ]
    assert not found, f"font-size must be a --text-* token: {found[:10]}"
    shorthand = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font" and value != "inherit"
    ]
    assert not shorthand, f"use font-family/font-size, not the font shorthand: {shorthand}"


def test_two_weights() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font-weight" and value not in WEIGHTS
    ]
    assert not found, f"only 400 and 500: {found[:10]}"


def test_soft_rounded_tokens() -> None:
    tokens = TOKENS_CSS.read_text(encoding="utf-8")
    assert "--radius: 8px;" in tokens
    assert "--radius-lg: 12px;" in tokens
    assert "--shape-chip: 999px;" in tokens
    graph_css = (WEB_SRC / "styles" / "graph.css").read_text(encoding="utf-8")
    card_corners = [
        value
        for selector, body in _blocks(graph_css)
        if selector == ".ag-card"
        for prop, value in _decls(body)
        if prop == "border-radius"
    ]
    assert card_corners == ["var(--radius)"], "graph cards share the soft --radius corner"


def test_faces_and_corners_come_from_tokens() -> None:
    faces = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font-family" and value != "inherit" and not value.startswith("var(--face-")
    ]
    assert not faces, f"font-family must be a --face-* token: {faces}"
    corners = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "border-radius" and not set(value.split()) <= SHAPES
    ]
    assert not corners, f"border-radius must be --radius or a --shape-* token: {corners[:10]}"


def test_transitions_use_motion_token() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop in ("transition", "transition-duration")
        and re.search(r"(?<![\w-])\d*\.?\d+m?s\b", value)
    ]
    assert not found, f"transitions use var(--motion-fast): {found}"


def test_committed_spa_bundle_matches_web_sources() -> None:
    """Offline freshness gate: rebuild via `pnpm --dir web build` after any web/ or OpenAPI edit."""
    stamp = _web_source_stamp()
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert f'<meta name="ariadne-source-sha256" content="{stamp}">' in index_html

    banner = f"/*! ariadne | source-sha256:{stamp} */"
    assets = STATIC_DIR / "assets"
    for name in ("ariadne.js", "ariadne.css", "ariadne-graph.js", "ariadne-graph.css"):
        head = (assets / name).read_text(encoding="utf-8").splitlines()[:3]
        assert banner in head, f"{name} is stale; run `pnpm --dir web build`"
    assert not list(assets.glob("*.map")), "production static bundle must not commit sourcemaps"
