"""Unit tests for the wave graph engine and YAML workflow loader."""

from __future__ import annotations

import threading

import pytest

from olympus.graph.engine import (
    END,
    START,
    Graph,
    GraphStateCollision,
    GraphVisitLimitError,
    Node,
    run_graph,
)
from olympus.graph.loader import (
    build_graph,
    describe_workflow,
    list_builtin_workflows,
    load_workflow,
)


def test_parallel_wave_runs_concurrently_and_merges_state() -> None:
    barrier = threading.Barrier(2, timeout=10)
    main_thread = threading.get_ident()
    solo_threads: list[int] = []
    wave_threads: list[int] = []

    def solo(state: dict[str, object]) -> dict[str, object]:
        solo_threads.append(threading.get_ident())
        return {"seed": 1}

    def left(state: dict[str, object]) -> dict[str, object]:
        barrier.wait()
        wave_threads.append(threading.get_ident())
        return {"left": state["seed"] + 1}

    def right(state: dict[str, object]) -> dict[str, object]:
        barrier.wait()
        wave_threads.append(threading.get_ident())
        return {"right": state["seed"] + 2}

    graph = Graph(name="fan-out")
    graph.add_node(Node("solo", solo))
    graph.add_node(Node("left", left))
    graph.add_node(Node("right", right))
    graph.add_edge(START, "solo")
    graph.add_edge("solo", "left")
    graph.add_edge("solo", "right")

    events: list[tuple[str, dict[str, object]]] = []
    state = run_graph(graph, {}, observer=lambda event, payload: events.append((event, payload)))

    assert state["left"] == 2
    assert state["right"] == 3
    assert solo_threads == [main_thread]
    assert len(wave_threads) == 2
    assert wave_threads[0] != wave_threads[1]
    kinds = [event for event, _ in events]
    assert kinds[0] == "graph_started"
    assert kinds[-1] == "graph_finished"
    assert kinds.count("node_started") == 3


def test_same_wave_state_collision_is_rejected() -> None:
    graph = Graph(name="collision")
    graph.add_node(Node("a", lambda state: {"shared": 1}))
    graph.add_node(Node("b", lambda state: {"shared": 2}))
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")

    with pytest.raises(GraphStateCollision) as excinfo:
        run_graph(graph, {})
    assert "shared" in str(excinfo.value)
    assert excinfo.value.collisions == {"shared": ["a", "b"]}


def test_internal_underscore_keys_may_collide() -> None:
    graph = Graph(name="internal")
    graph.add_node(Node("a", lambda state: {"_scratch": 1}))
    graph.add_node(Node("b", lambda state: {"_scratch": 2}))
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")

    state = run_graph(graph, {})
    assert state["_scratch"] in (1, 2)


def test_router_cycle_is_bounded_by_max_visits() -> None:
    def draft(state: dict[str, object]) -> dict[str, object]:
        attempts = int(state.get("attempts", 0))
        return {"attempts": attempts + 1}

    def review(state: dict[str, object]) -> dict[str, object]:
        return {"reviewed": True}

    graph = Graph(name="retry")
    graph.add_node(Node("draft", draft, max_visits=3))
    graph.add_node(Node("review", review, max_visits=3))
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_router("review", lambda state: "draft" if state["attempts"] < 99 else END)

    with pytest.raises(GraphVisitLimitError):
        run_graph(graph, {}, max_steps=20)


def test_max_steps_guard_raises() -> None:
    graph = Graph(name="endless")
    graph.add_node(Node("loop", lambda state: {"n": 1}, max_visits=100))
    graph.add_edge(START, "loop")
    graph.add_router("loop", lambda state: "loop")

    with pytest.raises(RuntimeError, match="max_steps"):
        run_graph(graph, {}, max_steps=3)


def test_graph_paused_then_resumed_from_next_node() -> None:
    calls: list[str] = []

    def gate(state: dict[str, object]) -> dict[str, object]:
        calls.append("gate")
        from olympus.graph.engine import GraphPaused

        raise GraphPaused("gate", "needs human approval", {"route": "awaiting"})

    def after(state: dict[str, object]) -> dict[str, object]:
        calls.append("after")
        return {"after_done": True}

    graph = Graph(name="pause")
    graph.add_node(Node("gate", gate))
    graph.add_node(Node("after", after))
    graph.add_edge(START, "gate")
    graph.add_edge("gate", "after")

    events: list[str] = []
    state = run_graph(
        graph, {}, observer=lambda event, payload: events.append(event), start_from="gate"
    )
    assert state["_paused_node"] == "gate"
    assert state["_pause_reason"] == "needs human approval"
    assert state["route"] == "awaiting"
    assert "graph_paused" in events
    assert calls == ["gate"]

    graph.nodes["gate"].fn = lambda state: {"route": "ok"}
    resumed = run_graph(graph, state, start_from="after")
    assert resumed["after_done"] is True
    assert calls == ["gate", "after"]


def test_node_error_routes_to_on_error_target() -> None:
    def boom(state: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("kaboom")

    graph = Graph(name="errors")
    graph.add_node(Node("risky", boom, on_error="recover"))
    graph.add_node(Node("recover", lambda state: {"recovered": True}))
    graph.add_edge(START, "risky")

    state = run_graph(graph, {})
    assert state["recovered"] is True
    assert "kaboom" in state["_last_error"]


def test_all_builtin_workflows_load_and_describe() -> None:
    names = list_builtin_workflows()
    assert set(names) == {"product_blueprint", "full_sdlc_tdd", "repo_fast_patch"}

    blueprint = load_workflow("product_blueprint")
    assert blueprint.stop_after == "plan"
    assert blueprint.topology == "modular_product"
    description = describe_workflow(blueprint)
    node_names = [node["name"] for node in description["graph"]["nodes"]]
    assert "intake" in node_names
    assert "design" in node_names
    assert "gate_plan" in node_names
    assert description["gates"]["approve_design"] is True

    full = load_workflow("full_sdlc_tdd")
    assert full.stop_after == "execute"
    assert full.topology == "tdd_first"
    assert all(full.gates.values())

    fast = load_workflow("repo_fast_patch")
    assert fast.stop_after == "execute"
    assert fast.topology == "minimal_patch"
    assert not any(fast.gates.values())
    fast_graph = build_graph(fast)
    assert fast_graph.describe()["routers"] == ["align", "intake"]


def test_workflow_path_loading_and_unknown_node_types(tmp_path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "\n".join(
            [
                "name: custom",
                "description: ad-hoc",
                "stop_after: goal",
                "nodes:",
                "  - name: intake",
                "    type: intake_triage",
                "edges:",
                "  - source: START",
                "    target: intake",
                "  - source: intake",
                "    target: END",
            ]
        ),
        encoding="utf-8",
    )
    definition = load_workflow(custom)
    assert definition.name == "custom"
    assert definition.stop_after == "goal"

    broken = tmp_path / "broken.yaml"
    broken.write_text(
        "\n".join(
            [
                "name: broken",
                "nodes:",
                "  - name: nope",
                "    type: does_not_exist",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown node types"):
        load_workflow(broken)

    with pytest.raises(FileNotFoundError):
        load_workflow("no_such_workflow")
