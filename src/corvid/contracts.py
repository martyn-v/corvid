from datetime import datetime
from typing import Literal, Union, get_args, get_origin
from pydantic import BaseModel, Field


def _leaf_value(model: BaseModel, path: str) -> object:
    """Returns the value at a dot path, e.g. "origin.name"."""
    value: object = model
    for part in path.split("."):
        value = getattr(value, part)
    return value


def _nested_model(annotation: object) -> type[BaseModel] | None:
    """Returns the BaseModel type in an annotation (unwrapping Optional), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) is Union:
        for arg in get_args(annotation):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


class Requester(BaseModel):
    name: str | None = Field(
        default=None,
        description="The requester's name as written in the email, e.g. John Doe",
        json_schema_extra={"required_for_quote": True},
    )
    email: str | None = Field(
        default=None,
        description="The requester's email address as written in the email, e.g. john.doe@example.org",
        json_schema_extra={"required_for_quote": True},
    )
    company: str | None = Field(
        default=None,
        description="The requester's company as written in the email, e.g. Acme Inc.",
        json_schema_extra={"required_for_quote": True},
    )


class Location(BaseModel):
    name: str | None = Field(
        default=None,
        description="The location name as written in the email, e.g. Cartagena without country or code",
        json_schema_extra={"required_for_quote": True},
    )
    country: str | None = Field(
        default=None,
        description="The location country as written in the email, e.g. Colombia without code",
    )
    code: str | None = Field(
        default=None,
        description="The location code as written in the email, e.g. CTG (IATA) or COCTG (UN/LOCODE) without country prefix",
    )


class QuoteRequest(BaseModel):
    requester: Requester
    origin: Location
    destination: Location

    @staticmethod
    def _walk_fields(
        model: type[BaseModel], prefix: str = ""
    ) -> list[tuple[str, object]]:
        """Returns (dot path, FieldInfo) for every leaf field, in declaration order."""
        fields = []
        for name, info in model.model_fields.items():
            path = f"{prefix}{name}"
            nested = _nested_model(info.annotation)
            if nested is not None:
                fields.extend(QuoteRequest._walk_fields(nested, f"{path}."))
            else:
                fields.append((path, info))
        return fields

    @staticmethod
    def dot_fields() -> list[str]:
        """Returns a list of all fields in the quote request in dot notation."""
        return [path for path, _ in QuoteRequest._walk_fields(QuoteRequest)]

    @staticmethod
    def required_dot_fields() -> list[str]:
        """Returns the dot paths of fields marked required_for_quote in the ontology."""
        return [
            path
            for path, info in QuoteRequest._walk_fields(QuoteRequest)
            if (getattr(info, "json_schema_extra", None) or {}).get(
                "required_for_quote"
            )
        ]

    def missing(self) -> list[str]:
        """Returns a list of missing required fields in the quote request."""
        return [
            path
            for path in QuoteRequest.required_dot_fields()
            if not _leaf_value(self, path)
        ]


class Provenance(BaseModel):
    source: Literal["email", "learned", "answered_question"]
    fact_uuid: str | None = None  # graphiti edge, when source == "learned"
    valid_at: datetime | None = None  # what memory believed — feeds the stale check


def present_fields(request: QuoteRequest) -> set[str]:
    """Returns the dot paths of all fields with a value, required or not."""
    return {
        path
        for path in QuoteRequest.dot_fields()
        if _leaf_value(request, path) is not None
    }


def apply_fill(
    request: QuoteRequest,
    provenance: dict[str, Provenance],
    path: str,
    value: str,
    prov: Provenance,
) -> None:
    """Sets the field at the dot path and records its provenance, as one move.

    FILL and ASK must write fields through this so no field changes without
    a provenance record, and nothing already accounted for is silently
    overwritten.
    """
    if path not in QuoteRequest.dot_fields():
        raise KeyError(f"Unknown quote request field: {path}")
    if path in provenance:
        raise ValueError(
            f"{path} already has provenance ({provenance[path].source}); refusing to overwrite"
        )

    *parents, leaf = path.split(".")
    target = request
    for parent in parents:
        target = getattr(target, parent)
    setattr(target, leaf, value)
    provenance[path] = prov
