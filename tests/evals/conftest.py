"""Pytest wiring for the Stage 1 eval suite.

Live-provider tests are opt-in: they are skipped unless ``--run-live-api`` is
passed explicitly so the default offline CI run never touches the network.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run live LLM evaluation tests that call a real provider.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if bool(config.getoption("--run-live-api")):
        return
    skip_live = pytest.mark.skip(reason="need --run-live-api option to run")
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_live)
