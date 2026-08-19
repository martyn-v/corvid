import datetime
from email.utils import format_datetime

from corvid.logging import make_logger
from harness.models import Case, Persona
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
- If the list has no origin, do not mention or hint at an origin.
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

    response = model.invoke(messages)

    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
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
