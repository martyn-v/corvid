from pydantic import BaseModel, Field

# --- entity types: class name = type name, docstring guides extraction ---


class Customer(BaseModel):
    """The company that sent this email and requests the shipping quote.
    Example: 'Nordfrost Seafood AB'. NOT the cargo, NOT a product,
    NOT a location."""


class Contact(BaseModel):
    """A person's name: the human who signed or wrote the email.
    Example: 'Erik Lindqvist'. NOT a code, NOT a company, NOT a place."""


class Location(BaseModel):
    """A city or port. The name is the city name only, e.g. 'Gothenburg' —
    never with country ('Gothenburg, Sweden' is wrong). A UN/LOCODE like
    SEGOT is an attribute, not a name."""

    locode: str | None = Field(None, description="UN/LOCODE, e.g. SEGOT")


entity_types = {
    "Customer": Customer,
    "Contact": Contact,
    "Location": Location,
}

# --- edge types ---


class ShipsFrom(BaseModel):
    """Customer's origin location for shipments. State the fact exactly as:
    '<customer name> ships from <location name>'. Use the city name only, no country. No extra details."""


class ShipsTo(BaseModel):
    """Customer's destination location for shipments. State the fact exactly as:
    '<customer name> ships to <location name>'. Use the city name only, no country. No extra details."""


class WorksFor(BaseModel):
    """Contact belongs to a customer. State the fact exactly as:
    '<person name> works for <customer name>'."""


edge_types = {
    "SHIPS_FROM": ShipsFrom,
    "SHIPS_TO": ShipsTo,
    "WORKS_FOR": WorksFor,
}

# which edges are allowed between which entity pairs
edge_type_map = {
    ("Customer", "Location"): ["SHIPS_FROM", "SHIPS_TO"],
    ("Contact", "Customer"): ["WORKS_FOR"],
}
