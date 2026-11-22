"""Deterministic OpenAPI snapshot export for the Ariadne API.

Run ``uv run python -m olympus.web.openapi`` to regenerate the checked-in
``src/olympus/web/openapi.json`` after changing any request/response contract.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from olympus.core.store import RunStore
from olympus.web.server import AriadneApp, create_fastapi_app

OPENAPI_SNAPSHOT_PATH = Path(__file__).resolve().parent / "openapi.json"


def build_openapi_schema() -> dict[str, Any]:
    """Generate the OpenAPI schema from a throwaway, fully-wired application."""
    with tempfile.TemporaryDirectory(prefix="olympus-openapi-") as tmp:
        workspace = Path(tmp)
        app = AriadneApp(store=RunStore(workspace / "openapi.db"), workspace_root=workspace)
        return create_fastapi_app(app).openapi()


def render_openapi_schema(schema: dict[str, Any]) -> str:
    """Canonical deterministic JSON text (sorted keys, UTF-8, trailing newline)."""
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def export_openapi_snapshot(path: Path = OPENAPI_SNAPSHOT_PATH) -> Path:
    """Write the live schema to ``path`` and return the written path."""
    path.write_text(render_openapi_schema(build_openapi_schema()), encoding="utf-8")
    return path


def main() -> None:
    written = export_openapi_snapshot()
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
