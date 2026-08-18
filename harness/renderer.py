from harness.models import Fact, Persona
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    template="""\
You write short business emails for a freight customer requesting a quote.
Output the email as a raw .eml (RFC 5322) message: header lines first
(From, To, Subject, Date), then one blank line, then the body.
Rules:
- From: the contact name with a plausible address at the company's domain.
  To: quotes@forwarder.example. Subject: a short quote request subject.
  Date: the date from the facts, in RFC 5322 format.
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
    facts.append(f"- Date: {fact.date}")
    facts.append(f"- Company: {persona.company}")
    facts.append(f"- Sender: {persona.contact}")
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

    return raw.strip()
