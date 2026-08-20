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
    )
    email: str | None = Field(
        default=None,
        description="The requester's email address as written in the email, e.g. john.doe@example.org",
    )
    company: str | None = Field(
        default=None,
        description="The requester's company as written in the email, e.g. Acme Inc.",
    )


class Location(BaseModel):
    name: str | None = Field(
        default=None,
        description="The location name as written in the email, e.g. Cartagena, ",
    )


class QuoteRequest(BaseModel):
    requester: Requester
    origin: Location
    destination: Location

    @staticmethod
    def dot_fields() -> list[str]:
        """Returns a list of all fields in the quote request in dot notation."""

        def walk(model: type[BaseModel], prefix: str) -> list[str]:
            fields = []
            for name, info in model.model_fields.items():
                path = f"{prefix}{name}"
                nested = _nested_model(info.annotation)
                if nested is not None:
                    fields.extend(walk(nested, f"{path}."))
                else:
                    fields.append(path)
            return fields

        return walk(QuoteRequest, "")

    def missing(self) -> list[str]:
        """Returns a list of missing required fields in the quote request."""
        missing_fields = []
        if not self.requester.name:
            missing_fields.append("requester.name")
        if not self.requester.email:
            missing_fields.append("requester.email")
        if not self.origin.name:
            missing_fields.append("origin.name")
        if not self.destination.name:
            missing_fields.append("destination.name")
        return missing_fields


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
