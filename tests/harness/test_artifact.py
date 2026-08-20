import datetime
import re
from pathlib import Path

from corvid.contracts import Provenance, QuoteRequest
from harness.artifact import (
    case_record,
    dir_sha256,
    file_sha256,
    git_state,
    question_series,
    run_header,
)
from harness.models import (
    Case,
    ConfigFile,
    Contact,
    EmailStyle,
    GenerationSettings,
    GenerationVariables,
    Location,
    Persona,
    Range,
    RendererSettings,
    Timeline,
)


def _case() -> Case:
    return Case(
        index=2,
        persona="acme-alimentos",
        date=datetime.date(2023, 1, 16),
        origin=Location(locode="COBOG", name="Bogotá, Colombia"),
        destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
        mode="ocean_fcl",
        commodity="packaged non-perishable foods",
        pieces=11,
        weight_kg=7669,
        origin_omitted=True,
        n=5,
    )


def test_question_series_groups_per_persona_in_send_order():
    counts = [("acme", 3), ("nord", 2), ("nord", 0), ("acme", 1), ("nord", 1)]
    assert question_series(counts) == {"acme": [3, 1], "nord": [2, 0, 1]}


def test_case_record_is_json_safe_and_carries_provenance():
    request = QuoteRequest.model_validate(
        {
            "requester": {"name": "Ana Ruiz"},
            "origin": {"name": "Bogotá"},
            "destination": {"name": "Rotterdam"},
        }
    )
    valid_at = datetime.datetime(2023, 1, 6, 12, 0, tzinfo=datetime.timezone.utc)
    provenance = {
        "origin.name": Provenance(
            source="learned", fact_uuid="abc-123", valid_at=valid_at
        ),
        "destination.name": Provenance(source="email"),
    }
    scores = {"origin.name": "correct", "destination.name": "correct"}

    record = case_record(
        _case(), request, provenance, scores, asked=["origin.name"], seconds=4.2
    )

    assert record == {
        "type": "case",
        "key": "005-acme-alimentos-02",
        "persona": "acme-alimentos",
        "index": 2,
        "date": "2023-01-16",
        "values": {
            "requester.name": "Ana Ruiz",
            "origin.name": "Bogotá",
            "destination.name": "Rotterdam",
        },
        "scores": scores,
        "provenance": {
            "origin.name": {
                "source": "learned",
                "fact_uuid": "abc-123",
                "valid_at": "2023-01-06T12:00:00+00:00",
            },
            "destination.name": {
                "source": "email",
                "fact_uuid": None,
                "valid_at": None,
            },
        },
        "asked": ["origin.name"],
        "seconds": 4.2,
    }


def _world() -> ConfigFile:
    return ConfigFile(
        personas=[
            Persona(
                id="acme-alimentos",
                company="Acme Alimentos",
                contact=Contact(name="Ana Ruiz", email="ana@acme.example"),
                origin=Location(locode="COBOG", name="Bogotá, Colombia"),
                destination=Location(locode="NLRTM", name="Rotterdam, Netherlands"),
                mode="ocean_fcl",
                commodity="packaged non-perishable foods",
                omit_origin=True,
                style=EmailStyle(language="es", tone="formal"),
            )
        ],
        generation=GenerationSettings(
            seed=1234,
            emails_per_persona=10,
            omission_rate=0.5,
            timeline=Timeline(
                start_date=datetime.date(2023, 1, 6),
                start_offset_days=Range(min=1, max=3),
                gap_days=Range(min=2, max=5),
            ),
            renderer=RendererSettings(model="ollama/gemma4:31b", temperature=0.7),
            variables=GenerationVariables(
                pieces=Range(min=1, max=30), weight_kg=Range(min=1000, max=12000)
            ),
        ),
    )


def test_file_sha256_is_stable_and_content_sensitive(tmp_path: Path):
    f = tmp_path / "a.jsonl"
    f.write_text("one")
    first = file_sha256(f)
    assert first == file_sha256(f)
    f.write_text("two")
    assert file_sha256(f) != first


def test_dir_sha256_covers_every_matching_file(tmp_path: Path):
    (tmp_path / "a.eml").write_text("a")
    (tmp_path / "b.eml").write_text("b")
    (tmp_path / "ignored.txt").write_text("x")
    first = dir_sha256(tmp_path, "*.eml")
    (tmp_path / "ignored.txt").write_text("y")
    assert dir_sha256(tmp_path, "*.eml") == first
    (tmp_path / "b.eml").write_text("changed")
    assert dir_sha256(tmp_path, "*.eml") != first


def test_git_state_reports_commit_and_dirty_flag():
    commit, dirty = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert isinstance(dirty, bool)


def test_run_header_pins_code_inputs_and_models(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}")
    emails = tmp_path / "emails"
    emails.mkdir()
    (emails / "001.eml").write_text("hi")
    started = datetime.datetime(2026, 8, 20, 19, 0, tzinfo=datetime.timezone.utc)

    header = run_header(
        started=started,
        world=_world(),
        case_count=30,
        world_path=Path("harness/world.yaml"),
        cases_path=cases,
        emails_dir=emails,
        group="eval",
        commit="deadbeef" * 5,
        dirty=True,
        models={"agent": "ollama/gemma4:31b"},
    )

    assert header == {
        "type": "run",
        "started": "2026-08-20T19:00:00+00:00",
        "cases": 30,
        "world": "harness/world.yaml",
        "group": "eval",
        "commit": "deadbeef" * 5,
        "dirty": True,
        "seed": 1234,
        "renderer": {"model": "ollama/gemma4:31b", "temperature": 0.7},
        "cases_sha256": file_sha256(cases),
        "emails_sha256": dir_sha256(emails, "*.eml"),
        "models": {"agent": "ollama/gemma4:31b"},
    }
