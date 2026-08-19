from harness.models import Fact


def build_fact_key(fact: Fact) -> str:
    """Build a unique key for a fact based on its attributes."""
    return f"{fact.n:03d}-{fact.persona}-{fact.index:02d}"
