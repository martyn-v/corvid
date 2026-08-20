import harness.generate as generate
from test_validate_email import PERSONA, StubModel, make_case, make_raw

HEADERS = (
    "From: Marta Restrepo <marta@acme.example.com>\n"
    "To: quotes@forwarder.example\n"
    "Date: Fri, 06 Jan 2023 09:00:00 +0000\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=utf-8\n"
)


def patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(generate, "CASES_PATH", tmp_path / "cases.jsonl")
    monkeypatch.setattr(generate, "EMAILS_DIR", tmp_path / "emails")


def write_cached(tmp_path, case, raw):
    emails = tmp_path / "emails"
    emails.mkdir(parents=True, exist_ok=True)
    (emails / f"{case.key}.eml").write_text(HEADERS + raw)


def test_valid_cached_email_is_left_alone(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    cached = make_raw()
    write_cached(tmp_path, case, cached)
    model = StubModel(outputs=[make_raw(quantities="20 pieces weighing 5000 kg")])

    generate.render(model, [case], [PERSONA])

    assert model.calls == 0
    assert (tmp_path / "emails" / f"{case.key}.eml").read_text() == HEADERS + cached


def test_invalid_cached_email_is_rerendered(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(tmp_path, case, make_raw(quantities="14 pieces"))  # weight missing
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA])

    assert model.calls == 1
    assert "9385 kg" in (tmp_path / "emails" / f"{case.key}.eml").read_text()


def test_cached_headers_do_not_satisfy_validation(monkeypatch, tmp_path):
    """The From header names the sender; only the rendered prose counts."""
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(tmp_path, case, make_raw(signature="Marta Restrepo\nACME Alimentos SAS"))
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA])

    assert model.calls == 1
