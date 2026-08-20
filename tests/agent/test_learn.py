from datetime import datetime

from corvid.agent.ask import Exchange
from corvid.agent.learn import answers_episode, email_episode
from corvid.agent.parse_email import EmailAddress, ParsedEmail

EMAIL = ParsedEmail(
    raw=b"raw bytes",
    subject="Quote request",
    sender=EmailAddress(
        display_name="Marta Restrepo", address="marta@acme.example.com"
    ),
    date=datetime(2023, 3, 6, 9, 0),
    body_text="Need a quote to Rotterdam.",
    warnings=[],
)


def test_email_episode_is_named_for_the_file_and_renders_headers():
    name, body = email_episode("harness/emails/018-acme-alimentos-09.eml", EMAIL)
    assert name == "018-acme-alimentos-09"
    assert body == (
        "From: Marta Restrepo <marta@acme.example.com>\n"
        "Subject: Quote request\n"
        "\n"
        "Need a quote to Rotterdam."
    )


def test_answers_episode_renders_only_answered_questions():
    asked = [
        Exchange(path="origin.name", question="Where from?", answer="Bogotá"),
        Exchange(path="requester.company", question="Which company?", answer=None),
    ]
    episode = answers_episode("harness/emails/018-acme-alimentos-09.eml", EMAIL, asked)
    assert episode is not None
    name, body = episode
    assert name == "018-acme-alimentos-09-answers"
    assert "Marta Restrepo <marta@acme.example.com>" in body
    assert "Q: Where from?\nA: Bogotá" in body
    assert "company" not in body


def test_answers_episode_is_none_when_nothing_was_answered():
    asked = [Exchange(path="origin.name", question="Where from?", answer=None)]
    assert answers_episode("x/018.eml", EMAIL, asked) is None
    assert answers_episode("x/018.eml", EMAIL, []) is None


def test_answers_episode_names_a_sender_without_display_name():
    email = EMAIL.model_copy(
        update={"sender": EmailAddress(display_name=None, address="m@acme.example.com")}
    )
    asked = [Exchange(path="origin.name", question="Where from?", answer="Bogotá")]
    _, body = answers_episode("x/018.eml", email, asked)
    assert body.startswith("m@acme.example.com ")
