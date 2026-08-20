import datetime
import re
from email.utils import format_datetime

from corvid.logging import make_logger
from harness import text
from harness.models import Case, Persona
from num2words import num2words
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


TO_ADDRESS = "quotes@forwarder.example"

logger = make_logger("renderer")

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    template="""\
You write short business emails for a freight customer requesting a quote.
Output exactly one Subject header line, then one blank line, then the body:

Subject: <a short quote request subject>

<body>

Do not output any other header lines.
Rules:
- Include every fact from the list. Do not skip any.
- Do not add facts that are not in the list: no reference numbers, no
  container types, no prices, no postal addresses.
- The shipment travels from the Origin to the Destination. The
  Destination is where the goods arrive. Never present the Destination
  as the place the shipment departs from — "from <destination>" or
  "from your location in <destination>" would state the route backwards.
- If the list has no origin, do not mention, hint at, or invent a
  departure location. Phrase the request as shipping "to <destination>"
  and leave where it ships from unsaid.
{intro_rule}- Body: 3 to 6 sentences. Use real line breaks: greeting on its own
  line, blank lines between paragraphs, and the sign-off on its own
  lines. Wrap body lines at roughly 72 characters.
- After the sign-off, end with a realistic signature block on its own
  lines: the sender's full name, company, and email address, exactly as
  given in the facts. Do not invent phone numbers, job titles, websites,
  or postal addresses.
- Language: {language}. Tone: {tone}.
""",
    input_variables=["language", "tone", "intro_rule"],
)

FIRST_EMAIL_INTRO_RULE = """\
- This is the sender's first email to this freight forwarder: after the
  greeting, open with a brief introduction of the sender and their
  company (one or two sentences) before making the request. Describe the
  company's business only in terms the facts support (its name, its
  commodity, where it ships from and to).
"""

USER_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    Facts:
    {facts}
    """,
    input_variables=["facts"],
)


def destination_leaked_as_origin(body: str, destination: str) -> bool:
    """True when the destination city is presented as the departure point.

    Catches "from <destination-city>" (en) and "desde <city>" (es), with a
    few words allowed in between ("from your location in Rotterdam") — but
    not across a direction flip ("from Bogotá to Rotterdam" is a correct
    lane, not a leak).
    """
    city = destination.split(",")[0].strip()
    pattern = (
        r"\b(?:from|desde)\b"
        r"(?:(?!\b(?:to|a|hasta|hacia|destined|arriving|para)\b|[.\n]).){0,40}?"
        rf"\b{re.escape(city)}\b"
    )
    return re.search(pattern, body, re.IGNORECASE | re.DOTALL) is not None


def _word_key(s: str) -> str:
    """Normalized for spelled-out number comparison: folded, connectors dropped."""
    s = re.sub(r"[-,]", " ", text.fold(s))
    s = re.sub(r"\b(and|y)\b", " ", s)
    return " ".join(s.split())


def _number_stated(body: str, value: int, language: str) -> bool:
    """The renderer writes '9,385 kg' or 'twenty-one pieces'; both state the fact."""
    collapsed = re.sub(r"(?<=\d)[,.\s](?=\d)", "", body)
    if re.search(rf"\b{value}\b", collapsed):
        return True
    return _word_key(num2words(value, lang=language)) in _word_key(body)


def validate_email(raw: str, case: Case, persona: Persona) -> list[str]:
    """Check a rendered email against its case; returns problems, empty if valid.

    The cheap contract between the answer key and the prose: every fact the
    case states must appear, and the origin must not appear when the coin
    omitted it.
    """
    problems = []
    folded = text.fold(raw)

    origin_city = text.city(case.origin.name)
    if case.origin_omitted:
        if origin_city in folded:
            problems.append(f"origin omitted but '{case.origin.name}' appears")
    elif origin_city not in folded:
        problems.append(f"origin '{case.origin.name}' not stated")

    if text.city(case.destination.name) not in folded:
        problems.append(f"destination '{case.destination.name}' not stated")
    if destination_leaked_as_origin(raw, case.destination.name):
        problems.append(
            f"destination '{case.destination.name}' presented as the departure point"
        )

    language = persona.style.language
    if not _number_stated(raw, case.pieces, language):
        problems.append(f"pieces '{case.pieces}' not stated")
    if not _number_stated(raw, case.weight_kg, language):
        problems.append(f"weight '{case.weight_kg}' not stated")

    for label, value in (
        ("name", persona.contact.name),
        ("company", persona.company),
        ("email", persona.contact.email),
    ):
        if value not in raw:
            problems.append(f"sender {label} '{value}' missing")
    return problems


def summarize_facts(case: Case, persona: Persona) -> str:
    facts = []
    facts.append(f"- Sender Company: {persona.company}")
    facts.append(f"- Sender: {persona.contact.name}")
    facts.append(f"- Sender email: {persona.contact.email}")
    if not case.origin_omitted:
        facts.append(f"- Origin: {case.origin}")
    facts.append(f"- Destination: {case.destination}")
    facts.append(f"- Mode: {case.mode}")
    facts.append(f"- Commodity: {case.commodity}")
    facts.append(f"- Pieces: {case.pieces}")
    facts.append(f"- Weight (kg): {case.weight_kg}")
    if case.change_reason is not None:
        facts.append(f"- Mention: {case.change_reason}")
    return "\n".join(facts)


def email_raw(eml: str) -> str:
    """The rendered part of an .eml this renderer wrote: Subject onward.

    The transport headers above it are constructed, not rendered — they
    state the sender's name and address for free, so they don't count.
    """
    return eml[eml.index("Subject:") :]


def render_email(model: BaseChatModel, case: Case, persona: Persona) -> str:
    logger.debug(
        "rendering email",
        key=case.key,
        persona=persona.id,
        first_email=case.index == 1,
        origin_omitted=case.origin_omitted,
        change_email=case.change_reason is not None,
    )
    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(
                language=persona.style.language,
                tone=persona.style.tone,
                intro_rule=FIRST_EMAIL_INTRO_RULE if case.index == 1 else "",
            )
        ),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(facts=summarize_facts(case, persona))
        ),
    ]

    attempts = 3
    for attempt in range(1, attempts + 1):
        response = model.invoke(messages)
        raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        problems = validate_email(raw, case, persona)
        if not problems:
            break
        logger.warning(
            "rendered email fails validation, retrying",
            key=case.key,
            attempt=attempt,
            problems=problems,
        )
    else:
        raise ValueError(
            f"{case.key}: invalid after {attempts} attempts ({'; '.join(problems)}):\n{raw}"
        )

    sent_at = datetime.datetime.combine(
        case.date, datetime.time(9, 0), tzinfo=datetime.timezone.utc
    )
    headers = "\n".join(
        [
            f"From: {persona.contact.name} <{persona.contact.email}>",
            f"To: {TO_ADDRESS}",
            f"Date: {format_datetime(sent_at)}",
            "MIME-Version: 1.0",
            "Content-Type: text/plain; charset=utf-8",
        ]
    )

    return f"{headers}\n{raw.strip()}"
