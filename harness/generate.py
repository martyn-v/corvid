import datetime
from pathlib import Path
from random import Random
from corvid.llm import create_model
from harness.models import Fact, Persona, GenerationVariables, ConfigFile
import yaml
from langchain_core.language_models import BaseChatModel

from harness.renderer import render_email
from harness.utils import build_fact_key

START_DATE = datetime.date(2023, 1, 1)
EMAILS_DIR = Path("harness/emails")


def _make_row(
    p: Persona,
    i: int,
    date: datetime.date,
    prng: Random,
    variables: GenerationVariables,
) -> Fact:
    origin = p.origin
    is_change = False
    change_reason: str | None = None
    if p.change is not None:
        is_change = i == p.change.at_email
        if i >= p.change.at_email:
            origin = p.change.origin
        if is_change:
            change_reason = p.change.reason_in_email

    coin = prng.random()  # always draw: keeps later draws stable across config changes
    origin_omitted = p.omit_origin and coin < 0.7
    if is_change or i == 1:
        origin_omitted = False  # first email and change email must state the origin

    return Fact(
        n=None,
        index=i,
        persona=p.id,
        date=date,
        origin=origin,
        destination=p.destination,
        mode=p.mode,
        commodity=p.commodity,
        pieces=prng.randint(variables.pieces.min, variables.pieces.max),
        weight_kg=prng.randint(variables.weight_kg.min, variables.weight_kg.max),
        origin_omitted=origin_omitted,
        change_reason=change_reason,
    )


def build_facts(
    personas: list[Persona],
    seed: int,
    emails_per_persona: int,
    vars: GenerationVariables,
) -> list[Fact]:
    rows: list[Fact] = []
    for p in personas:
        prng = Random(f"{seed}-{p.id}")
        date = START_DATE + datetime.timedelta(days=prng.randint(0, 5))
        for i in range(1, emails_per_persona + 1):
            rows.append(_make_row(p, i, date, prng, vars))
            date += datetime.timedelta(days=prng.randint(3, 10))

    rows.sort(key=lambda r: (r.date, r.persona))
    for n, row in enumerate(rows, 1):
        row.n = n

    return rows


def render(model: BaseChatModel, facts: list[Fact], personas: list[Persona]) -> None:
    persona_by_id = {p.id: p for p in personas}
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    with open("facts.jsonl", "w") as f:
        for fact in facts:
            f.write(fact.model_dump_json() + "\n")
            path = EMAILS_DIR / f"{build_fact_key(fact)}.eml"
            if path.exists():
                continue
            email = render_email(model, fact, persona_by_id[fact.persona])
            with open(path, "w") as f_email:
                f_email.write(email)


def main():
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
    config = ConfigFile.model_validate(data)

    facts = build_facts(
        config.personas,
        config.generation.seed,
        config.generation.emails_per_persona,
        config.generation.variables,
    )

    model = create_model(model="ollama/qwen3.5:9b", temperature=0.7)

    render(model, facts, config.personas)


if __name__ == "__main__":
    main()
