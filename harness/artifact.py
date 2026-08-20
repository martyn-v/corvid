"""Build the per-run JSONL artifact so runs are comparable across memory designs."""

import hashlib
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from corvid.contracts import Provenance, QuoteRequest
from harness.models import Case, ConfigFile
from harness.score import Outcome


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_sha256(directory: Path, pattern: str) -> str:
    """Digest over matching files, order-independent — pins a run's inputs."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob(pattern)):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    """Returns (HEAD commit, worktree dirty) — the code under test."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return commit, bool(status.strip())


def run_header(
    *,
    started: datetime,
    world: ConfigFile,
    case_count: int,
    world_path: Path,
    cases_path: Path,
    emails_dir: Path,
    group: str,
    commit: str,
    dirty: bool,
    models: dict[str, str],
) -> dict:
    """The run record: when, what code, what inputs, what models.

    The seed records intent only — emails are LLM-rendered and cached, so
    the content hashes are what actually pin this run's inputs.
    """
    return {
        "type": "run",
        "started": started.isoformat(),
        "cases": case_count,
        "world": str(world_path),
        "group": group,
        "commit": commit,
        "dirty": dirty,
        "seed": world.generation.seed,
        "renderer": {
            "model": world.generation.renderer.model,
            "temperature": world.generation.renderer.temperature,
        },
        "cases_sha256": file_sha256(cases_path),
        "emails_sha256": dir_sha256(emails_dir, "*.eml"),
        "models": models,
    }


def question_series(counts: Iterable[tuple[str, int]]) -> dict[str, list[int]]:
    """Groups (persona, questions asked) pairs into per-persona series.

    Input order is send order, so each series reads as questions over the
    persona's email sequence — the headline claim of the eval.
    """
    series: dict[str, list[int]] = {}
    for persona, count in counts:
        series.setdefault(persona, []).append(count)
    return series


def case_record(
    case: Case,
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    scores: dict[str, Outcome],
    asked: list[str],
    seconds: float,
) -> dict:
    """One JSON-safe artifact line: what was filled, from where, and how it scored."""
    values = {
        path: value
        for path in QuoteRequest.dot_fields()
        if (value := _leaf(request, path)) is not None
    }
    return {
        "type": "case",
        "key": case.key,
        "persona": case.persona,
        "index": case.index,
        "date": case.date.isoformat(),
        "values": values,
        "scores": scores,
        "provenance": {
            path: {
                "source": prov.source,
                "fact_uuid": prov.fact_uuid,
                "valid_at": prov.valid_at.isoformat() if prov.valid_at else None,
            }
            for path, prov in provenance.items()
        },
        "asked": asked,
        "seconds": seconds,
    }


def _leaf(request: QuoteRequest, path: str) -> object:
    part: object = request
    for attr in path.split("."):
        part = getattr(part, attr)
    return part
