import json
from pydantic import BaseModel, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from corvid.contracts import QuoteRequest
from corvid.agent.parse_email import ParsedEmail
from langchain_core.language_models import BaseChatModel

SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    template="""You are an inbox assistant that extracts structured information from emails. Extract the relevant information and return it as a single JSON object matching this JSON Schema exactly. No prose, no markdown fences.

{schema}

Extract values exactly as stated in the email. Do not convert units or reformat identifiers; report what the email says.
Output compact single-line JSON with no whitespace. Omit optional fields that are not present in the email rather than emitting null.
""",
    input_variables=["schema"],
)


class QuoteRequestExtractionError(Exception):
    """Raised when there is an error extracting the quote request."""


class ExtractionResult(BaseModel):
    request: QuoteRequest
    raw: dict


def extract_quote_request(
    email: ParsedEmail, model: BaseChatModel
) -> ExtractionResult:
    """Extracts the quote request from the parsed email."""

    schema = json.dumps(QuoteRequest.model_json_schema())

    messages = [
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(schema=schema)),
        HumanMessage(content=email.body_text),
    ]

    response = model.invoke(messages)

    raw = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise QuoteRequestExtractionError(
            f"Model did not return valid JSON: {e}"
        ) from e

    try:
        return ExtractionResult(
            request=QuoteRequest.model_validate(data),
            raw=data,
        )
    except ValidationError as e:
        raise QuoteRequestExtractionError(
            f"Extraction failed schema validation: {e}",
        ) from e
