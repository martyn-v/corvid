import harness.generate as generate
from harness.models import RendererSettings
from harness.renderer import cache_key, stored_key
from test_validate_email import PERSONA, StubModel, make_case, make_raw

SETTINGS = RendererSettings(model="stub", temperature=0.0)

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


def write_cached(tmp_path, case, raw, key=None):
    headers = HEADERS
    if key is not None:
        headers += f"X-Corvid-Render-Key: {key}\n"
    emails = tmp_path / "emails"
    emails.mkdir(parents=True, exist_ok=True)
    (emails / f"{case.key}.eml").write_text(headers + raw)
    return headers + raw


def test_valid_cached_email_with_current_key_is_left_alone(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    cached = write_cached(tmp_path, case, make_raw(), key=cache_key(case, PERSONA, SETTINGS))
    model = StubModel(outputs=[make_raw(quantities="20 pieces weighing 5000 kg")])

    generate.render(model, [case], [PERSONA], SETTINGS)

    assert model.calls == 0
    assert (tmp_path / "emails" / f"{case.key}.eml").read_text() == cached


def test_cached_email_with_stale_key_is_rerendered(monkeypatch, tmp_path):
    """A prompt or model change moves the key; a valid email still re-renders."""
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(tmp_path, case, make_raw(), key="0" * 16)
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA], SETTINGS)

    assert model.calls == 1
    eml = (tmp_path / "emails" / f"{case.key}.eml").read_text()
    assert stored_key(eml) == cache_key(case, PERSONA, SETTINGS)


def test_cached_email_without_key_is_rerendered(monkeypatch, tmp_path):
    """Pre-cache-key emails carry no key; we can't know what rendered them."""
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(tmp_path, case, make_raw())
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA], SETTINGS)

    assert model.calls == 1


def test_invalid_cached_email_is_rerendered(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(  # weight missing
        tmp_path, case, make_raw(quantities="14 pieces"), key=cache_key(case, PERSONA, SETTINGS)
    )
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA], SETTINGS)

    assert model.calls == 1
    assert "9385 kg" in (tmp_path / "emails" / f"{case.key}.eml").read_text()


def test_cached_headers_do_not_satisfy_validation(monkeypatch, tmp_path):
    """The From header names the sender; only the rendered prose counts."""
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    write_cached(
        tmp_path,
        case,
        make_raw(signature="Marta Restrepo\nACME Alimentos SAS"),
        key=cache_key(case, PERSONA, SETTINGS),
    )
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA], SETTINGS)

    assert model.calls == 1


def test_rendered_email_carries_the_current_key(monkeypatch, tmp_path):
    patch_paths(monkeypatch, tmp_path)
    case = make_case()
    model = StubModel(outputs=[make_raw()])

    generate.render(model, [case], [PERSONA], SETTINGS)

    eml = (tmp_path / "emails" / f"{case.key}.eml").read_text()
    assert stored_key(eml) == cache_key(case, PERSONA, SETTINGS)
