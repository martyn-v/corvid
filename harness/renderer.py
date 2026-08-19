import datetime
from email.utils import format_datetime

from harness.models import Fact, Persona
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


TO_ADDRESS = "quotes@forwarder.example"

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
- Body: 3 to 6 sentences, signed with the contact name. Use real line
  breaks: greeting on its own line, blank lines between paragraphs, and
  the sign-off on its own lines. Wrap body lines at roughly 72 characters.
- Language: {language}. Tone: {tone}.
""",
    input_variables=["language", "tone"],
)

USER_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    Facts:
    {facts}
    """,
    input_variables=["facts"],
)


def summarize_facts(fact: Fact, persona: Persona) -> str:
    facts = []
    facts.append(f"- Company: {persona.company}")
    facts.append(f"- Sender: {persona.contact.name}")
    if not fact.origin_omitted:
        facts.append(f"- Origin: {fact.origin}")
    facts.append(f"- Destination: {fact.destination}")
    facts.append(f"- Mode: {fact.mode}")
    facts.append(f"- Commodity: {fact.commodity}")
    facts.append(f"- Pieces: {fact.pieces}")
    facts.append(f"- Weight (kg): {fact.weight_kg}")
    if fact.change_reason is not None:
        facts.append(f"- Mention: {fact.change_reason}")
    return "\n".join(facts)


def render_email(model: BaseChatModel, fact: Fact, persona: Persona) -> str:
    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(
                language=persona.style.language,
                tone=persona.style.tone,
            )
        ),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(facts=summarize_facts(fact, persona))
        ),
    ]

    response = model.invoke(messages)

    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    sent_at = datetime.datetime.combine(
        fact.date, datetime.time(9, 0), tzinfo=datetime.timezone.utc
    )
    headers = "\n".join(
        [
            f"From: {persona.contact.name} <{persona.contact.email}>",
            f"To: {TO_ADDRESS}",
            f"Date: {format_datetime(sent_at)}",
        ]
    )

    return f"{headers}\n{raw.strip()}"
