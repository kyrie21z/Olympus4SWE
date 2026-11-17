"""Deterministic design-system gate for the Ariadne console static assets.

style.css, index.html, ui.js and app.js may only read tokens. tokens.css is the
single file allowed to name a colour. These checks fail on any hand-written
colour, shadow, type size, weight, face, corner or transition duration.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src" / "olympus" / "web" / "static"
LINTED = ("style.css", "index.html", "ui.js", "app.js")


def _text(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


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
    """(file, selector, property, value) for style.css and every inline style attribute."""
    rows = [
        ("style.css", selector, prop, value)
        for selector, body in _blocks(_text("style.css"))
        for prop, value in _decls(body)
    ]
    for name in ("index.html", "ui.js", "app.js"):
        source = _text(name)
        for match in re.finditer(r'style="([^"]*)"', source):
            rows += [(name, "style=", prop, value) for prop, value in _decls(match.group(1))]
    return rows


def _is_svg_text(selector: str) -> bool:
    """SVG label sizes live in the viewBox's own units; the type scale is CSS-only."""
    return all(part.strip().startswith(".dag-svg") for part in selector.split(","))


def test_design_files_load_in_order() -> None:
    html = _text("index.html")
    tokens_at = html.find('href="/static/tokens.css"')
    style_at = html.find('href="/static/style.css"')
    assert tokens_at != -1, "index.html must load tokens.css"
    assert style_at != -1, "index.html must load style.css"
    assert tokens_at < style_at, "tokens.css must load before style.css"

    ui_at = html.find('src="/static/ui.js"')
    app_at = html.find('src="/static/app.js"')
    assert ui_at != -1, "index.html must load ui.js"
    assert app_at != -1, "index.html must load app.js"
    assert ui_at < app_at, "ui.js must load before app.js"


COLOUR = re.compile(r"(?<![&\w])#[0-9a-fA-F]{3,8}\b|\b(?:rgb|rgba|hsl|hsla)\(")


def test_no_colour_literals() -> None:
    """Every colour is a token, including SVG fills and inline style attributes."""
    for name in LINTED:
        source = _text(name)
        found = COLOUR.findall(source)
        assert not found, f"{name} holds raw colours — use var(--token): {found[:8]}"
    found_decls = [(f, s, p, v) for f, s, p, v in _declarations() if COLOUR.search(v)]
    assert not found_decls, f"raw colours in declarations: {found_decls[:8]}"


def test_no_shadows_except_floating() -> None:
    allowed = (".floating-menu", "dialog[open]")
    found = [
        (selector, value)
        for _f, selector, prop, value in _declarations()
        if prop == "box-shadow"
        and value != "none"
        and not any(marker in selector for marker in allowed)
    ]
    assert not found, f"only a floating menu or an open dialog takes --shadow-md: {found}"
    wrong = [
        (selector, value)
        for _f, selector, prop, value in _declarations()
        if prop == "box-shadow" and value not in ("none", "var(--shadow-md)")
    ]
    assert not wrong, f"box-shadow is none or var(--shadow-md): {wrong}"


BUTTONS = re.compile(r"(?:^|[\s,>+~])(?:button|\.btn)\b")
GROUNDS = {
    "transparent",
    "none",
    "var(--surface-paper)",
    "var(--surface-raised)",
    "var(--surface-sunk)",
    "var(--surface-bg)",
}


def test_buttons_never_fill() -> None:
    found = [
        (selector, value)
        for name, selector, prop, value in _declarations()
        if name == "style.css"
        and BUTTONS.search(selector)
        and prop in ("background", "background-color")
        and value not in GROUNDS
    ]
    assert not found, f"buttons never fill: {found}"


SIZES = {f"var(--text-{size})" for size in ("xs", "sm", "base", "lg", "xl", "2xl")}


def test_font_sizes_use_tokens() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font-size" and value not in SIZES | {"inherit"} and not _is_svg_text(selector)
    ]
    assert not found, f"font-size must be a --text-* token: {found[:10]}"
    shorthand = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font" and value != "inherit"
    ]
    assert not shorthand, f"use font-family/font-size, not the font shorthand: {shorthand}"


WEIGHTS = {"400", "500", "normal", "var(--font-weight-normal)", "var(--font-weight-medium)"}


def test_two_weights() -> None:
    found = [
        (name, selector, value)
        for name, selector, prop, value in _declarations()
        if prop == "font-weight" and value not in WEIGHTS
    ]
    assert not found, f"only 400 and 500: {found[:10]}"


SHAPES = {
    "0",
    "var(--radius)",
    "var(--shape-chip)",
    "var(--shape-circle)",
    "var(--shape-pill)",
    "var(--shape-bubble)",
}


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


UI_FUNCTIONS = (
    "esc",
    "uiCard",
    "uiBadge",
    "uiTable",
    "uiTabs",
    "uiNotice",
    "uiStatBand",
    "uiRow",
    "uiButton",
    "toolRow",
    "subPane",
    "receiptBlock",
    "stampSlots",
    "watchSlots",
)


def test_ui_primitives_and_watch_slots() -> None:
    ui = _text("ui.js")
    for name in UI_FUNCTIONS:
        assert re.search(rf"^function {name}\(", ui, re.MULTILINE), f"ui.js must define {name}"
    assert "MutationObserver" in ui, "watchSlots must observe the DOM"
    app = _text("app.js")
    assert re.search(r"watchSlots\(\);", app), "app.js must call watchSlots() at bootstrap"
