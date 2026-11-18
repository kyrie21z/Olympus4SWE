"""Unit tests for DAG validation, write_scope conflict detection, and dispatch logic."""

from __future__ import annotations

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    CheckKind,
    GoalSpec,
    RequirementItem,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import (
    PlanValidationError,
    assert_valid_plan,
    compute_dispatchable_tasks,
    normalize_scope_path,
    scopes_overlap,
    validate_plan,
)


def _check(cid: str = "C1", command: str = "pytest -q") -> AcceptanceCheck:
    return AcceptanceCheck(
        id=cid,
        kind=CheckKind.COMMAND,
        description="Run pytest",
        command=command,
    )


def _spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-1",
        goal="Build module",
        requirements=[
            RequirementItem(id="R1", description="Core logic", acceptance_criteria=["Tests pass"]),
            RequirementItem(id="R2", description="CLI", acceptance_criteria=["CLI works"]),
        ],
    )


def test_normalize_scope_rejects_escape_and_absolute_paths() -> None:
    assert normalize_scope_path("src/olympus/") == "src/olympus/"
    assert normalize_scope_path("./src/a.py") == "src/a.py"
    with pytest.raises(ValueError):
        normalize_scope_path("/etc/passwd")
    with pytest.raises(ValueError):
        normalize_scope_path("../outside")


def test_scopes_overlap_detects_nested_and_root_conflicts() -> None:
    assert scopes_overlap(["src/core"], ["src/core/store.py"])
    assert scopes_overlap(["."], ["tests/test_a.py"])
    assert not scopes_overlap(["src/core/"], ["src/cli/"])


def test_validate_plan_detects_cycles_and_missing_coverage() -> None:
    spec = _spec()
    cyclic = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                depends_on=["T2"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check()],
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R1"],
                depends_on=["T1"],
                write_scope=["src/b.py"],
                acceptance_checks=[_check()],
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        assert_valid_plan(cyclic, spec=spec)
    msg = str(exc.value)
    assert "cycle" in msg.lower()
    assert "R2" in msg


def test_validate_plan_forbids_overlapping_write_scope_for_concurrent_siblings() -> None:
    spec = _spec()
    conflicting = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                write_scope=["src/app/"],
                acceptance_checks=[_check()],
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R2"],
                write_scope=["src/app/cli.py"],
                acceptance_checks=[_check()],
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        assert_valid_plan(conflicting, spec=spec)
    assert "overlapping write_scope" in str(exc.value)

    # Making T2 depend on T1 serializes them, so shared write_scope becomes valid
    serialized = conflicting.model_copy(
        update={
            "tasks": [
                conflicting.tasks[0],
                conflicting.tasks[1].model_copy(update={"depends_on": ["T1"]}),
            ]
        }
    )
    assert_valid_plan(serialized, spec=spec)


def test_compute_dispatchable_tasks_respects_deps_slots_and_active_scopes() -> None:
    spec = _spec()
    plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.ACCEPTED,
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R2"],
                depends_on=["T1"],
                write_scope=["src/b.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.RUNNING,
            ),
            TaskNode(
                task_id="T3",
                title="C",
                objective="C",
                requirement_refs=["R2"],
                depends_on=["T1"],
                write_scope=["src/c.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.WAITING,
            ),
            TaskNode(
                task_id="T4",
                title="D",
                objective="D",
                requirement_refs=["R2"],
                depends_on=["T2"],
                write_scope=["src/d.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.WAITING,
            ),
        ],
    )
    # T1 accepted -> T3 ready; T4 blocked on T2 (RUNNING)
    ready = compute_dispatchable_tasks(plan, max_concurrent=2)
    assert [t.task_id for t in ready] == ["T3"]
    # When max_concurrent=1, T2 already consumes the only slot
    assert compute_dispatchable_tasks(plan, max_concurrent=1) == []


def test_validate_plan_rejects_hardcoded_host_workspace_path_in_command_check() -> None:
    spec = _spec()
    bad_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Bad",
                objective="Bad command path",
                requirement_refs=["R1", "R2"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check("C1", "cd /tmp/host_repo && pytest -q")],
            )
        ],
    )
    violations = validate_plan(bad_plan, spec=spec, workspace_root="/tmp/host_repo")
    assert any("hardcodes host workspace path" in v for v in violations)

    good_plan = bad_plan.model_copy(
        update={
            "tasks": [
                bad_plan.tasks[0].model_copy(
                    update={"acceptance_checks": [_check("C1", "pytest -q")]}
                )
            ]
        }
    )
    assert validate_plan(good_plan, spec=spec, workspace_root="/tmp/host_repo") == []


def test_validate_plan_rejects_broken_inline_python_syntax_and_accepts_multiline() -> None:
    spec = _spec()
    bad_try_cmd = (
        'python -c "from src.api.cfg import Config; c = Config(); '
        "try: c.limit = 99; assert False; except AttributeError: pass\""
    )
    bad_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Contract",
                objective="Verify config immutability",
                requirement_refs=["R1", "R2"],
                write_scope=["src/api/cfg.py"],
                acceptance_checks=[
                    _check("C-try", bad_try_cmd),
                    _check("C-quote", 'python -c "unclosed'),
                ],
            )
        ],
    )
    violations = validate_plan(bad_plan, spec=spec)
    assert any("C-try" in v and "invalid Python syntax" in v for v in violations)
    assert any("C-quote" in v and "invalid shell syntax" in v for v in violations)

    good_multiline_cmd = (
        'python -c "\nfrom src.api.cfg import Config\nc = Config()\n'
        'try:\n    c.limit = 99\n    assert False\nexcept AttributeError:\n    pass\n"'
    )
    good_plan = bad_plan.model_copy(
        update={
            "tasks": [
                bad_plan.tasks[0].model_copy(
                    update={
                        "acceptance_checks": [
                            _check("C-multi", good_multiline_cmd),
                            _check("C-pytest", "python -m pytest -c pyproject.toml -q"),
                        ]
                    }
                )
            ]
        }
    )
    assert validate_plan(good_plan, spec=spec) == []


def test_validate_plan_rejects_non_ancestor_sibling_import_and_accepts_dependency() -> None:
    spec = _spec()
    mw_cmd = (
        "python -c 'from src.api.limiter import FixedWindowLimiter; "
        "from src.api import server; assert FixedWindowLimiter'"
    )
    parallel_bad = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-limiter",
                title="Limiter",
                objective="Limiter core",
                requirement_refs=["R1"],
                write_scope=["src/api/limiter.py"],
                acceptance_checks=[_check("CL", "python -c 'import src.api.limiter'")],
            ),
            TaskNode(
                task_id="T1-middleware",
                title="Middleware",
                objective="Rate limit middleware",
                requirement_refs=["R2"],
                depends_on=[],
                write_scope=["src/api/middleware.py"],
                acceptance_checks=[_check("CM", mw_cmd)],
            ),
            TaskNode(
                task_id="T2-integrate",
                title="Export",
                objective="Export package symbol",
                requirement_refs=["R2"],
                depends_on=["T1-limiter", "T1-middleware"],
                write_scope=["src/api/__init__.py"],
                acceptance_checks=[_check("CI", "pytest -q")],
            ),
        ],
    )
    violations = validate_plan(parallel_bad, spec=spec)
    assert len(violations) == 1
    assert "T1-middleware" in violations[0]
    assert "T1-limiter" in violations[0]

    ordered_ok = parallel_bad.model_copy(
        update={
            "tasks": [
                parallel_bad.tasks[0],
                parallel_bad.tasks[1].model_copy(update={"depends_on": ["T1-limiter"]}),
                parallel_bad.tasks[2],
            ]
        }
    )
    assert validate_plan(ordered_ok, spec=spec) == []

    # Stdlib / third-party imports must never false-positive even when a sibling owns `src/`.
    stdlib_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-src",
                title="Core",
                objective="Core src",
                requirement_refs=["R1"],
                write_scope=["src/"],
                acceptance_checks=[_check("C1", "python -c 'pass'")],
            ),
            TaskNode(
                task_id="T2-docs",
                title="Docs",
                objective="Verify docs with pathlib and pytest",
                requirement_refs=["R2"],
                write_scope=["docs/"],
                acceptance_checks=[
                    _check("C2", "python -c 'import pathlib, pytest; assert pathlib.Path'")
                ],
            ),
        ],
    )
    assert validate_plan(stdlib_plan, spec=spec) == []
