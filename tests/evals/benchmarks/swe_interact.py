"""Docker-free adapter and Stage 1 -> Stage 2 E2E runner for scaleapi/SWE-Interact."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from olympus.core.contracts import AmbiguityItem, BudgetPolicy, GoalSpec, RunMode, StrictModel
from olympus.core.store import RunStore
from olympus.harness.worktree import build_sandbox_env
from olympus.providers.base import ChatMessage, LLMProvider
from olympus.providers.factory import build_provider
from olympus.stages.align import AlignStage
from olympus.stages.common import parse_model_json
from olympus.stages.orchestrate import OrchestrateStage
from tests.evals.oracle import (
    NEUTRAL_BOUNDARY_ANSWER,
    QuestionTracker,
    RuleBasedOracle,
    normalize_question,
)
from tests.evals.runner import CountingProvider
from tests.evals.schemas import HiddenSlot, LeakGuard, OracleReply

_GIT_CLONE_RE = re.compile(r"git\s+clone\s+(?P<url>https?://\S+|git@\S+|file://\S+|\S+)")
_GIT_CHECKOUT_RE = re.compile(r"git\s+checkout\s+(?P<sha>[0-9a-fA-F]{7,40})")
_PIP_INSTALL_RE = re.compile(r"\bpip3?\s+install\b(?P<args>[^;&|\n]+)")
_DIFF_TARGET_RE = re.compile(r'^diff --git "?a/.+?"? "?b/(?P<path>.+?)"?$')
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.-]{2,})`")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")
_POETRY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GENERIC_BACKTICK_TERMS = frozenset(
    {
        "comment",
        "data",
        "doc",
        "docs",
        "error",
        "errors",
        "false",
        "file",
        "function",
        "functions",
        "header",
        "item",
        "items",
        "key",
        "key_path",
        "keys",
        "list",
        "none",
        "path",
        "prefix",
        "result",
        "str",
        "table",
        "tables",
        "true",
        "type",
        "value",
        "values",
    }
)
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "all",
        "also",
        "and",
        "any",
        "are",
        "between",
        "both",
        "but",
        "can",
        "does",
        "each",
        "for",
        "from",
        "have",
        "into",
        "must",
        "not",
        "only",
        "should",
        "such",
        "that",
        "the",
        "their",
        "them",
        "then",
        "this",
        "when",
        "where",
        "which",
        "while",
        "with",
        "without",
    }
)

_PERSONA_ORACLE_SYSTEM_PROMPT = """Play the developer user described in the persona guide.
You know your private task intent (the Task Specification), revealing details only when asked.

Strict rules:
1. Answer THIS single question concisely and directly using the Task Specification as truth
   (do NOT reply "one thing at a time"; focus cleanly on the asked topic).
2. Do NOT proactively dump unrelated parts of the Task Specification that were not asked about.
3. If a detail about internal helpers, file layout, or local mechanics is unspecified,
   tell the assistant to inspect the repository and use its own judgment.
4. Never mention benchmarks, hidden tests, test patches, prompts, or grading.
5. Output ONLY JSON:
   {"answers": [{"question_index": 0, "answer": str, "matched_slot_ids": [str]}]}."""


class SWEInteractTask(StrictModel):
    """Parsed metadata, persona instructions, and hidden test patch for one task."""

    task_name: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    display_title: str = Field(min_length=1)
    display_description: str = Field(min_length=1)
    language: str = "python"
    category: str = ""
    repository_url: str = Field(min_length=1)
    base_commit_hash: str = Field(min_length=7)
    instruction_md: str = Field(min_length=1)
    persona_md: str = ""
    dockerfile: str = ""
    test_patch: str = Field(min_length=1)
    pip_packages: list[str] = Field(default_factory=list)
    uses_poetry_dev: bool = False
    hidden_slots: list[HiddenSlot] = Field(default_factory=list)


class SWEInteractOutcome(StrictModel):
    """End-to-end execution result for one SWE-Interact task."""

    task_name: str
    mode: Literal["interactive", "full"]
    run_id: str
    base_passed: bool
    new_passed: bool
    passed: bool
    base_exit_code: int
    new_exit_code: int
    test_patch_applied: bool
    model_patch: str = ""
    spec_id: str = ""
    spec_status: str = ""
    run_status: str = ""
    questions_asked: int = 0
    clarify_rounds: int = 0
    revealed_slots: list[str] = Field(default_factory=list)
    leakage_detected: bool = False
    task_statuses: dict[str, str] = Field(default_factory=dict)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    base_output: str = ""
    new_output: str = ""


class _PersonaAnswerItem(StrictModel):
    question_index: int = Field(ge=0)
    answer: str = Field(min_length=1)
    matched_slot_ids: list[str] = Field(default_factory=list)


class _PersonaOraclePayload(StrictModel):
    answers: list[_PersonaAnswerItem] = Field(default_factory=list)


def parse_task_toml(raw_toml: str) -> dict[str, Any]:
    """Parse SWE-Interact ``task.toml`` and extract normalized metadata fields."""
    data = tomllib.loads(raw_toml)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    task_sec = data.get("task", {})
    if not isinstance(task_sec, dict):
        task_sec = {}
    raw_name = str(task_sec.get("name", "")).strip()
    fallback_id = raw_name.split("/")[-1].strip() if raw_name else ""
    return {
        "task_id": str(metadata.get("task_id") or fallback_id).strip(),
        "display_title": str(metadata.get("display_title") or "").strip(),
        "display_description": str(metadata.get("display_description") or "").strip(),
        "language": str(metadata.get("language") or "python").strip().lower(),
        "category": str(metadata.get("category") or "").strip(),
        "repository_url": str(metadata.get("repository_url") or "").strip(),
        "base_commit_hash": str(metadata.get("base_commit_hash") or "").strip(),
    }


def _collapse_dockerfile_continuations(raw_dockerfile: str) -> str:
    return re.sub(r"\\\s*\n", " ", raw_dockerfile)


def _extract_pip_packages(args_text: str) -> list[str]:
    try:
        tokens = shlex.split(args_text)
    except ValueError:
        tokens = args_text.split()
    packages: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-r", "--requirement", "-c", "--constraint", "-i", "--index-url"}:
            skip_next = True
            continue
        if token.startswith("-") or token in {".", "./"}:
            continue
        packages.append(token)
    return packages


def parse_dockerfile(raw_dockerfile: str) -> tuple[str | None, str | None, list[str], bool]:
    """Extract repo URL, base commit, pip packages, and Poetry dev flag from Dockerfile."""
    collapsed = _collapse_dockerfile_continuations(raw_dockerfile)
    clone_match = _GIT_CLONE_RE.search(collapsed)
    checkout_match = _GIT_CHECKOUT_RE.search(collapsed)
    repo_url = clone_match.group("url") if clone_match else None
    base_sha = checkout_match.group("sha") if checkout_match else None

    pip_packages: list[str] = []
    uses_poetry_dev = False
    for line in collapsed.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "poetry install" in stripped:
            uses_poetry_dev = True
        for match in _PIP_INSTALL_RE.finditer(stripped):
            for pkg in _extract_pip_packages(match.group("args")):
                if pkg not in pip_packages:
                    pip_packages.append(pkg)
    return repo_url, base_sha, pip_packages, uses_poetry_dev


def extract_persona_behavior(persona_md: str, instruction_md: str) -> str:
    """Return the persona disclosure rules without duplicating the trailing task block."""
    if not persona_md.strip():
        return ""
    stripped_instr = instruction_md.strip()
    if stripped_instr and stripped_instr in persona_md:
        idx = persona_md.find(stripped_instr)
        return persona_md[:idx].strip()
    marker = "# Task block"
    if marker in persona_md:
        return persona_md.split(marker, 1)[0].strip()
    return persona_md.strip()


def _extract_keywords(text: str, *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).strip("`.,:;()[]")
        lowered = token.casefold()
        if len(lowered) < 3 or lowered in _STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords or [text[:40]]


def extract_hidden_slots(instruction_md: str, display_description: str) -> list[HiddenSlot]:
    """Extract deterministic HiddenSlot entries from ``instruction.md`` clauses."""
    desc_norm = " ".join(display_description.split()).casefold()
    bullets: list[str] = []
    for raw_line in instruction_md.splitlines():
        line = raw_line.strip()
        if line.startswith(("- ", "* ")):
            candidate = line[2:].strip()
        else:
            continue
        norm = " ".join(candidate.split()).casefold()
        if len(candidate) >= 16 and norm not in desc_norm:
            bullets.append(candidate)

    if not bullets:
        for para in instruction_md.split("\n\n"):
            cleaned = " ".join(para.split())
            if len(cleaned) >= 24 and cleaned.casefold() not in desc_norm:
                bullets.append(cleaned)

    slots: list[HiddenSlot] = []
    for idx, clause in enumerate(bullets[:8], start=1):
        keywords = _extract_keywords(clause)
        symbols = [
            sym
            for sym in dict.fromkeys(_BACKTICK_SYMBOL_RE.findall(clause))
            if sym.casefold() not in desc_norm
            and sym.casefold() not in _GENERIC_BACKTICK_TERMS
            and ("_" in sym or "." in sym or any(ch.isupper() for ch in sym))
        ]
        topic = " ".join(keywords[:3]) if keywords else f"requirement {idx}"
        slots.append(
            HiddenSlot(
                slot_id=f"S{idx}",
                topic=topic,
                keywords=keywords,
                aliases=symbols[:4],
                answer=clause,
                forbidden_leak_terms=symbols[:6],
            )
        )
    return slots


def load_swe_interact_task_dir(task_dir: Path, *, task_name: str | None = None) -> SWEInteractTask:
    """Load one SWE-Interact task from a local directory."""
    resolved = task_dir.resolve()
    toml_path = resolved / "task.toml"
    instr_path = resolved / "instruction.md"
    patch_path = resolved / "tests" / "test.patch"
    dockerfile_path = resolved / "environment" / "Dockerfile"
    persona_path = resolved / "environment" / "user-server" / "persona.md"

    if not toml_path.is_file():
        raise FileNotFoundError(f"Missing task.toml under {resolved}")
    if not instr_path.is_file():
        raise FileNotFoundError(f"Missing instruction.md under {resolved}")
    if not patch_path.is_file():
        raise FileNotFoundError(f"Missing tests/test.patch under {resolved}")

    meta = parse_task_toml(toml_path.read_text(encoding="utf-8"))
    dockerfile = dockerfile_path.read_text(encoding="utf-8") if dockerfile_path.is_file() else ""
    df_repo, df_sha, pip_packages, uses_poetry_dev = parse_dockerfile(dockerfile)

    repo_url = meta["repository_url"] or df_repo or ""
    base_sha = meta["base_commit_hash"] or df_sha or ""
    if not repo_url or not base_sha:
        raise ValueError(f"Task {resolved} missing repository_url or base_commit_hash")

    instruction_md = instr_path.read_text(encoding="utf-8")
    persona_md = persona_path.read_text(encoding="utf-8") if persona_path.is_file() else ""
    display_title = meta["display_title"] or meta["task_id"] or resolved.name
    display_desc = meta["display_description"] or display_title
    resolved_name = task_name or resolved.name

    return SWEInteractTask(
        task_name=resolved_name,
        task_id=meta["task_id"] or resolved_name,
        display_title=display_title,
        display_description=display_desc,
        language=meta["language"] or "python",
        category=meta["category"],
        repository_url=repo_url,
        base_commit_hash=base_sha,
        instruction_md=instruction_md,
        persona_md=persona_md,
        dockerfile=dockerfile,
        test_patch=patch_path.read_text(encoding="utf-8"),
        pip_packages=pip_packages,
        uses_poetry_dev=uses_poetry_dev,
        hidden_slots=extract_hidden_slots(instruction_md, display_desc),
    )


def _fetch_gh_file(repo: str, rel_path: str, *, required: bool) -> str | None:
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/{rel_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if required:
            raise RuntimeError(f"Failed to fetch {repo}/{rel_path}: {proc.stderr.strip()}")
        return None
    payload = json.loads(proc.stdout)
    encoded = payload.get("content")
    if isinstance(encoded, str) and encoded.strip():
        return base64.b64decode(encoded).decode("utf-8")
    download_url = payload.get("download_url")
    if isinstance(download_url, str) and download_url:
        with urllib.request.urlopen(download_url, timeout=30) as resp:  # noqa: S310
            raw_bytes: bytes = resp.read()
            return raw_bytes.decode("utf-8")
    if required:
        raise RuntimeError(f"Empty content and download_url for {repo}/{rel_path}")
    return None


def fetch_swe_interact_task(
    task_name: str,
    *,
    split: str = "multiturn",
    cache_dir: Path | None = None,
    repo: str = "scaleapi/SWE-Interact",
) -> SWEInteractTask:
    """Fetch a SWE-Interact task from GitHub (or local cache) without touching solution/."""
    target_dir = (cache_dir / split / task_name) if cache_dir is not None else None
    required_files = ("task.toml", "instruction.md", "tests/test.patch")
    optional_files = ("environment/Dockerfile", "environment/user-server/persona.md")

    if target_dir is not None and all((target_dir / rel).is_file() for rel in required_files):
        return load_swe_interact_task_dir(target_dir, task_name=task_name)

    with tempfile.TemporaryDirectory(prefix="swe-interact-fetch-") as tmp:
        stage_dir = target_dir if target_dir is not None else Path(tmp) / task_name
        prefix = f"data/{split}/{task_name}"
        for rel in required_files:
            content = _fetch_gh_file(repo, f"{prefix}/{rel}", required=True)
            assert content is not None
            dest = stage_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        for rel in optional_files:
            content = _fetch_gh_file(repo, f"{prefix}/{rel}", required=False)
            if content is not None:
                dest = stage_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
        return load_swe_interact_task_dir(stage_dir, task_name=task_name)


class PersonaUserOracle:
    """User simulator backed by SWE-Interact ``persona.md`` + ``instruction.md``."""

    def __init__(
        self,
        task: SWEInteractTask,
        *,
        provider: LLMProvider | None = None,
        max_questions: int = 8,
    ) -> None:
        self._task = task
        self._provider = provider
        self._max_questions = max(0, max_questions)
        self._fallback_oracle = RuleBasedOracle(task.hidden_slots, max_questions=max_questions)
        self._revealed_slot_ids: set[str] = set()
        self._leakage_detected = False
        self._leak_terms: list[str] = []
        self.last_error: str | None = None

    @property
    def revealed_slot_ids(self) -> AbstractSet[str]:
        if self._provider is None:
            return self._fallback_oracle.revealed_slot_ids
        return self._revealed_slot_ids

    @property
    def leakage_detected(self) -> bool:
        if self._provider is None:
            return self._fallback_oracle.leakage_detected
        return self._leakage_detected

    @property
    def leak_terms(self) -> Sequence[str]:
        if self._provider is None:
            return self._fallback_oracle.leak_terms
        return tuple(self._leak_terms)

    def reply(self, questions: Sequence[AmbiguityItem]) -> list[OracleReply]:
        if self._provider is None:
            return self._fallback_oracle.reply(questions)

        snapshot = list(questions)
        if not snapshot:
            return []

        behavior = extract_persona_behavior(self._task.persona_md, self._task.instruction_md)
        system_prompt = _PERSONA_ORACLE_SYSTEM_PROMPT
        if behavior:
            system_prompt = f"{system_prompt}\n\nPersona guide:\n{behavior}"

        known_slot_ids = {s.slot_id for s in self._task.hidden_slots}
        replies: list[OracleReply] = []
        for question in snapshot:
            replies.append(self._reply_one(question, system_prompt, known_slot_ids))
        return replies

    def _reply_one(
        self,
        question: AmbiguityItem,
        system_prompt: str,
        known_slot_ids: set[str],
    ) -> OracleReply:
        assert self._provider is not None
        slot_index = [
            {
                "slot_id": s.slot_id,
                "topic": s.topic,
                "already_revealed": s.slot_id in self._revealed_slot_ids,
            }
            for s in self._task.hidden_slots
        ]
        question_payload = [
            {"index": 0, "question": question.question, "options": question.options}
        ]
        user_prompt = (
            f"Initial request title: {self._task.display_title}\n"
            f"Initial request description: {self._task.display_description}\n\n"
            f"Task Specification (source of truth; answer ONLY what is asked):\n"
            f"{self._task.instruction_md}\n\n"
            f"Tracked topic IDs (tag any directly answered topics in matched_slot_ids):\n"
            f"{json.dumps(slot_index, ensure_ascii=False)}\n\n"
            f"Assistant clarification question (JSON):\n"
            f"{json.dumps(question_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Return JSON with question_index=0 now."
        )
        try:
            response = self._provider.complete(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ],
                temperature=0.0,
            )
            payload = parse_model_json(response.content, _PersonaOraclePayload)
        except Exception as exc:
            self.last_error = f"persona oracle fallback: {exc}"
            return self._fallback_oracle.reply([question])[0]

        item = payload.answers[0] if payload.answers else None
        if item is None:
            return OracleReply(
                question=question.question,
                matched_slot_id=None,
                answer=NEUTRAL_BOUNDARY_ANSWER,
                revealed=False,
            )

        matched = [
            sid
            for sid in item.matched_slot_ids
            if sid in known_slot_ids and len(self._revealed_slot_ids) < self._max_questions
        ]
        for sid in matched:
            if len(self._revealed_slot_ids) < self._max_questions:
                self._revealed_slot_ids.add(sid)

        revealed_slots = [
            s for s in self._task.hidden_slots if s.slot_id in self._revealed_slot_ids
        ]
        allowed_terms = {
            term.casefold()
            for s in revealed_slots
            for term in (*s.forbidden_leak_terms, *s.aliases)
        }
        remaining = [
            s.model_copy(
                update={
                    "forbidden_leak_terms": [
                        t for t in s.forbidden_leak_terms if t.casefold() not in allowed_terms
                    ],
                }
            )
            for s in self._task.hidden_slots
            if s.slot_id not in self._revealed_slot_ids
            and any(t.casefold() not in allowed_terms for t in s.forbidden_leak_terms)
        ]
        answer = item.answer.strip() or NEUTRAL_BOUNDARY_ANSWER
        leaks = LeakGuard.check_text_leakage(answer, remaining)
        if leaks:
            self._leakage_detected = True
            for term in leaks:
                if term not in self._leak_terms:
                    self._leak_terms.append(term)
            answer = LeakGuard.redact_terms(answer, leaks)

        return OracleReply(
            question=question.question,
            matched_slot_id=matched[0] if matched else None,
            answer=answer,
            revealed=bool(matched),
            leakage_detected=bool(leaks),
            leak_terms=leaks,
        )


def _poetry_dev_packages(repo_dir: Path) -> list[str]:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.is_file():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return []
    tool = data.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    groups = poetry.get("group", {}) if isinstance(poetry, dict) else {}
    dev_group = groups.get("dev", {}) if isinstance(groups, dict) else {}
    deps = dev_group.get("dependencies", {}) if isinstance(dev_group, dict) else {}
    if not isinstance(deps, dict):
        return []
    out: list[str] = []
    for name in deps:
        if name.lower() == "python":
            continue
        if _POETRY_NAME_RE.fullmatch(name):
            out.append(name)
    return out


def _append_git_exclude(repo_dir: Path, patterns: Sequence[str]) -> None:
    exclude_path = repo_dir / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.is_file() else ""
    lines = set(existing.splitlines())
    missing = [pat for pat in patterns if pat not in lines]
    if missing:
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        exclude_path.write_text(existing + prefix + "\n".join(missing) + "\n", encoding="utf-8")


def prepare_swe_interact_workspace(
    task: SWEInteractTask,
    repo_dir: Path,
    *,
    install_deps: bool = True,
    python_executable: str | None = None,
) -> Path:
    """Clone ``task.repository_url``, checkout ``base_commit_hash``, and provision ``.venv``."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    subprocess.run(
        ["git", "clone", task.repository_url, str(repo_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", task.base_commit_hash],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    if (repo_dir / ".gitmodules").is_file():
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=repo_dir,
            check=False,
            capture_output=True,
            text=True,
        )
    subprocess.run(["git", "config", "user.email", "olympus@eval.local"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Olympus Eval"], cwd=repo_dir, check=True)
    _append_git_exclude(repo_dir, [".venv/", ".olympus/"])

    if install_deps:
        venv_dir = repo_dir / ".venv"
        venv_cmd = ["uv", "venv", str(venv_dir)]
        if python_executable:
            venv_cmd.extend(["--python", python_executable])
        subprocess.run(venv_cmd, cwd=repo_dir, check=True, capture_output=True, text=True)

        venv_python = venv_dir / "bin" / "python"
        install_targets: list[str] = []
        if (
            (repo_dir / "pyproject.toml").is_file()
            or (repo_dir / "setup.py").is_file()
            or (repo_dir / "setup.cfg").is_file()
        ):
            install_targets.extend(["-e", "."])
        install_targets.extend(task.pip_packages)
        if task.uses_poetry_dev:
            install_targets.extend(_poetry_dev_packages(repo_dir))
        if "pytest" not in install_targets:
            install_targets.append("pytest")

        deduped_targets: list[str] = []
        for idx, item in enumerate(install_targets):
            if item == "-e":
                deduped_targets.append(item)
                continue
            if idx > 0 and install_targets[idx - 1] == "-e":
                deduped_targets.append(item)
                continue
            if item not in deduped_targets:
                deduped_targets.append(item)

        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv_python), *deduped_targets],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    return repo_dir


def capture_model_patch(repo_dir: Path, base_commit: str) -> str:
    """Capture workspace changes against ``base_commit`` without keeping changes staged."""
    subprocess.run(
        ["git", "reset", "--soft", base_commit],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "add", "-A", "--", "."],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    diff_proc = subprocess.run(
        ["git", "diff", "--cached", "--binary"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "reset", "-q"], cwd=repo_dir, check=True, capture_output=True)
    return diff_proc.stdout


def extract_patch_target_paths(patch_text: str) -> list[str]:
    """Return sorted relative file paths touched by a unified git patch."""
    paths: set[str] = set()
    for line in patch_text.splitlines():
        match = _DIFF_TARGET_RE.match(line)
        if match:
            paths.add(match.group("path"))
    return sorted(paths)


def apply_hidden_test_patch(repo_dir: Path, test_patch: str) -> Path:
    """Reset any files touched by ``test_patch``, apply it cleanly, and return ``test.sh``."""
    for rel in extract_patch_target_paths(test_patch):
        co = subprocess.run(
            ["git", "checkout", "HEAD", "--", rel],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if co.returncode != 0:
            subprocess.run(
                ["git", "rm", "-r", "--cached", "--ignore-unmatch", "--", rel],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            target = repo_dir / rel
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as tf:
        tf.write(test_patch)
        patch_file = Path(tf.name)
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_file)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git apply test.patch failed: {proc.stderr.strip()}")
    finally:
        patch_file.unlink(missing_ok=True)

    test_sh = repo_dir / "test.sh"
    if not test_sh.is_file():
        raise FileNotFoundError(f"Missing test.sh in {repo_dir} after applying test.patch")
    test_sh.chmod(test_sh.stat().st_mode | 0o111)
    return test_sh


def run_test_sh(
    repo_dir: Path,
    mode: Literal["base", "new"],
    *,
    timeout_sec: int = 600,
) -> tuple[int, str]:
    """Execute ``bash test.sh <mode>`` inside ``repo_dir`` with isolated ``.venv`` env."""
    env = build_sandbox_env(repo_dir)
    if not (repo_dir / ".venv").is_dir():
        # Keep the virtualenv bin dir without dereferencing its python symlink.
        py_bin = str(Path(sys.executable).parent)
        env["PATH"] = f"{py_bin}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.run(
        ["bash", "test.sh", mode],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, combined[-8000:]


def _run_stage1_for_task(
    *,
    align: AlignStage,
    run_id: str,
    task: SWEInteractTask,
    mode: Literal["interactive", "full"],
    oracle: PersonaUserOracle,
    align_max_steps: int,
    max_clarify_rounds: int,
) -> tuple[GoalSpec, int, int]:
    raw_goal = (
        task.instruction_md
        if mode == "full"
        else f"{task.display_title}\n\n{task.display_description}"
    )
    spec = align.propose_spec(
        run_id=run_id,
        raw_goal=raw_goal,
        enable_critique=False,
        max_steps=align_max_steps,
    )
    questions_asked = 0
    clarify_rounds = 0
    tracker = QuestionTracker()

    if mode == "interactive":
        seen_keys: set[str] = set()
        for _round in range(max_clarify_rounds):
            unanswered = [amb for amb in spec.open_ambiguities if amb.answer is None]
            fresh = [
                amb for amb in unanswered if normalize_question(amb.question) not in seen_keys
            ]
            if not fresh:
                break
            clarify_rounds += 1
            tracker.record(amb.question for amb in fresh)
            for amb in fresh:
                seen_keys.add(normalize_question(amb.question))
            questions_asked += len(fresh)
            replies = oracle.reply(fresh)
            answers = {
                amb.id: rep.answer
                for amb, rep in zip(fresh, replies, strict=True)
            }
            spec = align.apply_clarifications(run_id=run_id, answers=answers)
            if not spec.unresolved_blockers():
                break

    if spec.unresolved_blockers():
        reason = (
            "full specification mode"
            if mode == "full"
            else "interactive clarification rounds exhausted"
        )
        spec = align.confirm_spec(
            run_id=run_id,
            approver="swe-interact-harness",
            accept_risk=True,
            reason=reason,
        )
    else:
        spec = align.confirm_spec(
            run_id=run_id,
            approver="swe-interact-harness",
        )
    return spec, questions_asked, clarify_rounds


def run_swe_interact_e2e(
    task: SWEInteractTask,
    *,
    agent_provider: LLMProvider,
    oracle_provider: LLMProvider | None = None,
    mode: Literal["interactive", "full"] = "interactive",
    work_root: Path | None = None,
    install_deps: bool = True,
    python_executable: str | None = None,
    align_max_steps: int = 12,
    max_clarify_rounds: int = 2,
    worker_max_steps: int = 24,
    cleanup: bool = True,
) -> SWEInteractOutcome:
    """Run Stage 1 -> Stage 2 -> hidden ``test.patch`` verification on one SWE-Interact task."""
    start = time.monotonic()
    counting_agent = CountingProvider(agent_provider)
    oracle_source = CountingProvider(oracle_provider) if oracle_provider is not None else None
    oracle = PersonaUserOracle(task, provider=oracle_source)

    owns_tmp = work_root is None
    if work_root is None:
        base_dir = Path(tempfile.mkdtemp(prefix="swe-interact-run-"))
    else:
        base_dir = Path(work_root).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = base_dir / "repo"
    state_dir = base_dir / "olympus_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    run_id = ""
    spec_id = ""
    spec_status = ""
    run_status = ""
    questions_asked = 0
    clarify_rounds = 0
    model_patch = ""
    task_statuses: dict[str, str] = {}
    error_msg: str | None = None
    test_patch_applied = False
    base_exit_code = -1
    new_exit_code = -1
    base_output = ""
    new_output = ""

    try:
        prepare_swe_interact_workspace(
            task,
            repo_dir,
            install_deps=install_deps,
            python_executable=python_executable,
        )
        store = RunStore(state_dir / "olympus.db")
        run_id = store.create_run(
            title=f"SWE-Interact: {task.task_id}",
            mode=RunMode.EXECUTION,
            budget_policy=BudgetPolicy(
                max_llm_calls=150,
                max_total_tokens=4_000_000,
                max_concurrent_tasks=3,
            ),
        )
        align = AlignStage(
            store=store,
            provider=counting_agent,
            workspace_root=repo_dir,
            trace_dir=state_dir / "traces",
        )
        orchestrate = OrchestrateStage(
            store=store,
            provider=counting_agent,
            workspace_root=repo_dir,
            artifacts_root=state_dir / "artifacts",
        )

        spec, questions_asked, clarify_rounds = _run_stage1_for_task(
            align=align,
            run_id=run_id,
            task=task,
            mode=mode,
            oracle=oracle,
            align_max_steps=align_max_steps,
            max_clarify_rounds=max_clarify_rounds,
        )
        spec_id = spec.spec_id
        spec_status = spec.status.value

        orchestrate.propose_plan(run_id=run_id, auto_approve=True)
        final_snap = orchestrate.run_execution(
            run_id=run_id,
            apply_to_workspace=True,
            worker_max_steps=worker_max_steps,
        )
        final_plan = final_snap.get("plan")
        if final_plan is not None:
            task_statuses = {t.task_id: t.status.value for t in final_plan.tasks}
        run_status = str(final_snap.get("status", ""))
    except Exception as exc:
        error_msg = str(exc)

    if repo_dir.exists() and (repo_dir / ".git").exists():
        try:
            model_patch = capture_model_patch(repo_dir, task.base_commit_hash)
            apply_hidden_test_patch(repo_dir, task.test_patch)
            test_patch_applied = True
            base_exit_code, base_output = run_test_sh(repo_dir, "base")
            new_exit_code, new_output = run_test_sh(repo_dir, "new")
        except Exception as verify_exc:
            error_msg = (
                f"{error_msg}; verify failed: {verify_exc}"
                if error_msg
                else f"verify failed: {verify_exc}"
            )

    if owns_tmp and cleanup:
        shutil.rmtree(base_dir, ignore_errors=True)

    total_calls = counting_agent.llm_calls + (oracle_source.llm_calls if oracle_source else 0)
    total_in = counting_agent.input_tokens + (oracle_source.input_tokens if oracle_source else 0)
    total_out = counting_agent.output_tokens + (oracle_source.output_tokens if oracle_source else 0)
    base_passed = test_patch_applied and base_exit_code == 0
    new_passed = test_patch_applied and new_exit_code == 0

    return SWEInteractOutcome(
        task_name=task.task_name,
        mode=mode,
        run_id=run_id,
        base_passed=base_passed,
        new_passed=new_passed,
        passed=base_passed and new_passed,
        base_exit_code=base_exit_code,
        new_exit_code=new_exit_code,
        test_patch_applied=test_patch_applied,
        model_patch=model_patch,
        spec_id=spec_id,
        spec_status=spec_status,
        run_status=run_status,
        questions_asked=questions_asked,
        clarify_rounds=clarify_rounds,
        revealed_slots=sorted(oracle.revealed_slot_ids),
        leakage_detected=oracle.leakage_detected,
        task_statuses=task_statuses,
        llm_calls=total_calls,
        input_tokens=total_in,
        output_tokens=total_out,
        duration_seconds=round(time.monotonic() - start, 2),
        error=error_msg,
        base_output=base_output,
        new_output=new_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Olympus4SWE E2E against a SWE-Interact task.")
    parser.add_argument("--task", help="Task directory name under scaleapi/SWE-Interact")
    parser.add_argument("--task-dir", type=Path, help="Local directory containing one task")
    parser.add_argument("--split", default="multiturn", choices=("multiturn", "singleturn"))
    parser.add_argument("--mode", default="interactive", choices=("interactive", "full"))
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--oracle-provider", default=None)
    parser.add_argument("--oracle-model", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--worker-max-steps", type=int, default=24)
    args = parser.parse_args(argv)

    if args.task_dir is not None:
        task = load_swe_interact_task_dir(args.task_dir)
    elif args.task:
        task = fetch_swe_interact_task(args.task, split=args.split, cache_dir=args.cache_dir)
    else:
        parser.error("Provide --task or --task-dir")

    env_file = args.env_file if args.env_file and args.env_file.is_file() else None
    agent_provider = build_provider(
        provider=args.provider,
        model=args.model,
        env_file=env_file,
        environ=os.environ,
        timeout=args.timeout,
    )
    oracle_provider = (
        build_provider(
            provider=args.oracle_provider or args.provider,
            model=args.oracle_model or args.model,
            env_file=env_file,
            environ=os.environ,
            timeout=args.timeout,
        )
        if args.mode == "interactive"
        else None
    )

    outcome = run_swe_interact_e2e(
        task,
        agent_provider=agent_provider,
        oracle_provider=oracle_provider,
        mode=args.mode,
        work_root=args.work_root,
        worker_max_steps=args.worker_max_steps,
        cleanup=not args.keep_workspace,
    )
    summary = outcome.model_dump(exclude={"model_patch", "base_output", "new_output"})
    summary["model_patch_bytes"] = len(outcome.model_patch.encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
