"""Deterministic wave-scheduled graph runtime (waku-inspired, dependency free).

The engine executes a named node graph in topological waves: a single ready node runs on
the calling thread, while two or more ready nodes fan out onto a thread pool. Pure-Python
routers decide the next wave from live state, and every node is visit-bounded so retry
cycles always converge.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

START = "START"
END = "END"

NodeFn = Callable[[dict[str, Any]], "dict[str, Any] | None"]
RouterFn = Callable[[Mapping[str, Any]], "str | Iterable[str] | None"]
Observer = Callable[[str, dict[str, Any]], None]


class GraphStateCollision(Exception):
    """Raised when two nodes in the same wave write the same state key."""

    def __init__(self, collisions: dict[str, list[str]]) -> None:
        self.collisions = collisions
        detail = "; ".join(f"{key} <- {', '.join(nodes)}" for key, nodes in collisions.items())
        super().__init__(f"Graph state collision detected: {detail}")


class GraphPaused(Exception):
    """Raised by a node that needs a non-deterministic human/upstream decision."""

    def __init__(
        self,
        node_name: str,
        reason: str,
        state_updates: Mapping[str, Any] | None = None,
    ) -> None:
        self.node_name = node_name
        self.reason = reason
        self.state_updates: dict[str, Any] = dict(state_updates or {})
        super().__init__(f"Graph paused at '{node_name}': {reason}")


class GraphVisitLimitError(RuntimeError):
    """Raised when a node is re-entered beyond its configured max_visits budget."""


@dataclass
class Node:
    name: str
    fn: NodeFn
    kind: str = "fn"
    max_visits: int = 1
    on_error: str | None = None


@dataclass
class Graph:
    """Named node set plus static edges and pure state routers."""

    name: str = "graph"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    routers: dict[str, RouterFn] = field(default_factory=dict)
    router_targets: dict[str, list[str]] = field(default_factory=dict)

    def add_node(
        self,
        node: Node | str,
        fn: NodeFn | None = None,
        *,
        kind: str = "fn",
        max_visits: int = 1,
        on_error: str | None = None,
    ) -> Node:
        resolved = node if isinstance(node, Node) else Node(
            name=node,
            fn=fn or _missing_node_fn(node),
            kind=kind,
            max_visits=max_visits,
            on_error=on_error,
        )
        self.nodes[resolved.name] = resolved
        self.edges.setdefault(resolved.name, [])
        return resolved

    def add_edge(self, source: str, target: str) -> None:
        self.edges.setdefault(source, [])
        if target not in self.edges[source]:
            self.edges[source].append(target)
        if target not in (START, END):
            self.edges.setdefault(target, [])

    def add_router(
        self,
        source: str,
        fn: RouterFn,
        *,
        targets: Sequence[str] | None = None,
    ) -> Graph:
        if source not in self.nodes:
            raise KeyError(f"Cannot attach router to unknown node: {source}")
        self.routers[source] = fn
        if targets is not None:
            self.router_targets[source] = [t for t in dict.fromkeys(targets) if t]
        return self

    def successors(self, node_name: str, state: Mapping[str, Any]) -> list[str]:
        """Resolve the next node names from a router first, then static edges."""
        router = self.routers.get(node_name)
        if router is None:
            return [n for n in self.edges.get(node_name, []) if n != START]
        routed = router(state)
        if routed is None:
            return [n for n in self.edges.get(node_name, []) if n != START]
        targets = [routed] if isinstance(routed, str) else list(routed)
        return [t for t in targets if t != START]

    def describe(self) -> dict[str, Any]:
        """Serializable topology summary used by the workflow API and projection."""
        edges_out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for source, targets in self.edges.items():
            if source == START:
                continue
            for target in targets:
                if (source, target) not in seen:
                    seen.add((source, target))
                    edges_out.append(
                        {
                            "source": source,
                            "target": target,
                            # Router fan-out is a runtime decision; the console draws it dashed.
                            "conditional": source in self.routers,
                        }
                    )
        for source, targets in self.router_targets.items():
            if source == START:
                continue
            for target in targets:
                if (source, target) not in seen:
                    seen.add((source, target))
                    edges_out.append(
                        {
                            "source": source,
                            "target": target,
                            "conditional": True,
                        }
                    )
        return {
            "name": self.name,
            "start": [n for n in self.edges.get(START, [])],
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind,
                    "max_visits": node.max_visits,
                    "on_error": node.on_error,
                    "routed": node.name in self.routers,
                }
                for node in self.nodes.values()
            ],
            "edges": edges_out,
            "routers": sorted(self.routers),
        }


def _missing_node_fn(name: str) -> NodeFn:
    def _raise(state: dict[str, Any]) -> dict[str, Any]:
        raise KeyError(f"Graph node '{name}' has no bound function")

    return _raise


def _notify(observer: Observer | None, event: str, payload: dict[str, Any]) -> None:
    if observer is not None:
        observer(event, payload)


def _normalize_targets(targets: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for target in targets:
        if target in (END, START):
            continue
        if target not in ordered:
            ordered.append(target)
    return ordered


def _run_wave(
    graph: Graph,
    wave: list[str],
    state: dict[str, Any],
    observer: Observer | None,
) -> tuple[dict[str, Any], GraphPaused | None, dict[str, BaseException]]:
    """Execute one wave; return merged updates, an optional pause, and node errors."""
    if len(wave) == 1:
        name = wave[0]
        node = graph.nodes[name]
        _notify(observer, "node_started", {"node": name, "kind": node.kind})
        try:
            update = node.fn(dict(state))
        except GraphPaused as paused:
            _notify(observer, "node_paused", {"node": name, "reason": paused.reason})
            return {}, paused, {}
        except BaseException as exc:  # noqa: BLE001 - routed or re-raised by the caller
            return {}, None, {name: exc}
        merged = dict(update or {})
        _notify(observer, "node_finished", {"node": name, "updated_keys": sorted(merged)})
        return merged, None, {}

    names = list(wave)
    for name in names:
        _notify(
            observer,
            "node_started",
            {"node": name, "kind": graph.nodes[name].kind, "wave_size": len(names)},
        )

    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {name: pool.submit(graph.nodes[name].fn, dict(state)) for name in names}
        for name, future in futures.items():
            try:
                results[name] = dict(future.result() or {})
            except BaseException as exc:  # noqa: BLE001 - routed or re-raised by the caller
                errors[name] = exc

    for name, failure in errors.items():
        if isinstance(failure, GraphPaused):
            _notify(observer, "node_paused", {"node": name, "reason": failure.reason})
            return {}, failure, {}

    written: dict[str, list[str]] = {}
    for name, update in results.items():
        for key in update:
            if str(key).startswith("_"):
                continue
            written.setdefault(str(key), []).append(name)
    collisions = {key: nodes for key, nodes in written.items() if len(nodes) > 1}
    if collisions:
        raise GraphStateCollision(collisions)

    wave_updates: dict[str, Any] = {}
    for name in names:
        if name in results:
            wave_updates.update(results[name])
            _notify(
                observer,
                "node_finished",
                {"node": name, "updated_keys": sorted(results[name])},
            )
    return wave_updates, None, errors


def run_graph(
    graph: Graph,
    state: Mapping[str, Any],
    *,
    observer: Observer | None = None,
    max_steps: int = 25,
    start_from: str | None = None,
) -> dict[str, Any]:
    """Execute ``graph`` until END, a pause, or a guard rail trips.

    A single ready node runs inline; two or more ready nodes fan out concurrently.
    Routers run on the live state after each merged wave.
    """
    working: dict[str, Any] = dict(state)
    visits: dict[str, int] = {}
    _notify(
        observer,
        "graph_started",
        {"graph": graph.name, "start_from": start_from},
    )

    if start_from is not None and start_from not in graph.nodes:
        raise KeyError(f"start_from node '{start_from}' is not in graph '{graph.name}'")
    frontier = (
        [start_from]
        if start_from is not None
        else _normalize_targets(graph.edges.get(START, []))
    )
    steps = 0

    while frontier:
        ready = _normalize_targets(frontier)
        if not ready:
            break
        if steps >= max_steps:
            raise RuntimeError(f"Graph '{graph.name}' exceeded max_steps={max_steps}")
        steps += 1

        for name in ready:
            node = graph.nodes.get(name)
            if node is None:
                raise KeyError(f"Graph '{graph.name}' has no node named '{name}'")
            if visits.get(name, 0) >= node.max_visits:
                raise GraphVisitLimitError(
                    f"Node '{name}' exceeded max_visits={node.max_visits} in graph "
                    f"'{graph.name}'"
                )
        for name in ready:
            visits[name] = visits.get(name, 0) + 1

        working["_wave"] = steps
        updates, paused, errors = _run_wave(graph, ready, working, observer)
        if paused is not None:
            working.update(updates)
            working.update(paused.state_updates)
            working["_paused_node"] = paused.node_name
            working["_pause_reason"] = paused.reason
            _notify(
                observer,
                "graph_paused",
                {"node": paused.node_name, "reason": paused.reason},
            )
            return working

        if errors:
            routed_errors: dict[str, str] = {}
            for name in ready:
                node = graph.nodes[name]
                failure = errors.get(name)
                if failure is None:
                    continue
                if node.on_error is None:
                    raise failure
                routed_errors[name] = str(failure)
                _notify(observer, "node_error", {"node": name, "error": str(failure)})
            working["_errors"] = {**working.get("_errors", {}), **routed_errors}
            working["_last_error"] = next(iter(routed_errors.values()), "")
        else:
            working.update(updates)
        working.pop("_paused_node", None)
        working.pop("_pause_reason", None)

        next_frontier: list[str] = []
        finished = False
        for name in ready:
            node = graph.nodes[name]
            if name in errors:
                targets = [node.on_error] if node.on_error is not None else []
            else:
                try:
                    targets = graph.successors(name, working)
                except Exception as route_error:  # noqa: BLE001 - surfaced as node error
                    if node.on_error is None:
                        raise
                    _notify(observer, "node_error", {"node": name, "error": str(route_error)})
                    targets = [node.on_error]
            for target in targets:
                if target == END:
                    finished = True
                    continue
                if target not in next_frontier:
                    next_frontier.append(target)

        _notify(
            observer,
            "routed",
            {"from": ready, "to": next_frontier, "wave": steps, "finished": finished},
        )
        if finished:
            _notify(observer, "graph_finished", {"graph": graph.name, "steps": steps})
            return working
        frontier = next_frontier

    working["_wave"] = steps
    _notify(observer, "graph_finished", {"graph": graph.name, "steps": steps})
    return working
