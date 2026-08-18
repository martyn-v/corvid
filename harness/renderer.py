from harness.models import Fact, Persona
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# The Fact and Persona are sent as the human message (their JSON dumps);
# this system prompt carries every rule. The emails are the answer sheet
# for the memory milestones, so the hard constraints below are what keep
# facts.jsonl and the rendered text in agreement.
SYSTEM_PROMPT = """\
You write one email from a shipper to their freight forwarder, asking for
a quote on a single upcoming shipment.

You are given two JSON objects: the sender's persona (company, contact,
language, tone) and the facts of this one shipment. The email is part of
an ongoing correspondence with the same forwarder. "index" is the email's
position in that correspondence: at index 1 the sender may briefly
introduce their company; at any later index write a routine message with
no introduction.

Voice:
- Write as the named contact at the company, in the persona's language,
  matching the persona's tone.
- Sign off with the contact's first name; use the company name in the
  signature or introduction, not both.

Hard content rules:
- The email must state: the destination, the transport mode in a
  shipper's own words (ocean_reefer -> a refrigerated/reefer container;
  ocean_fcl -> a full container / FCL), the commodity, the piece count,
  and the weight in kg. Work them into natural sentences; never output a
  form, a table, or a labeled field list.
- If origin_omitted is true: do not mention, hint at, or imply the
  origin city, port, or country anywhere in the email.
- If origin_omitted is false: state the origin.
- If change_reason is not null, this email announces a relocation: state
  the new origin and work the given reason into the message, paraphrased
  naturally in the persona's language.
- Reproduce pieces and weight_kg exactly as given. Thousands separators
  are fine; rounding, converting to tonnes, or approximating is not.
- Invent no logistics facts beyond the JSON: no prices, no deadlines or
  dates other than the email's date, no container or booking numbers,
  no street addresses, no ports or routings, no incoterms, no extra
  cargo details. Pleasantries and filler are welcome where the tone
  calls for them; new facts are not.

Output format:
- Plain text only, no markdown.
- Line 1: "Date: " followed by the shipment JSON's date.
- Line 2: "Subject: " followed by a short subject in the persona's
  language.
- Then a blank line and the body. A few sentences; the length of a real
  operational email.

Return only the email text, nothing else.
"""

USER_PROMPT_TEMPLATE = PromptTemplate(
    template="""
    Sender persona: 
    <persona>{persona}</persona>
    \n
    <facts>{facts}</facts>
    """,
    input_variables=["persona", "facts"],
)


def render_email(model: BaseChatModel, fact: Fact, persona: Persona) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(
                persona=persona.model_dump_json(
                    include={"id", "company", "contact", "style"}
                ),
                facts=fact.model_dump_json(),
            )
        ),
    ]

    response = model.invoke(messages)

    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    return raw.strip()
