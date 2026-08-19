import datetime
from random import Random
from corvid.llm import create_model
from harness.models import Fact, FactDraft, Persona, GenerationSettings, ConfigFile
import yaml
from langchain_core.language_models import BaseChatModel

from harness.paths import EMAILS_DIR, FACTS_PATH, WORLD_PATH
from harness.renderer import render_email


def _make_row(
    p: Persona,
    i: int,
    date: datetime.date,
    prng: Random,
    settings: GenerationSettings,
) -> FactDraft:
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
    origin_omitted = p.omit_origin and coin < settings.omission_rate
    if is_change or i == 1:
        origin_omitted = False  # first email and change email must state the origin

    return FactDraft(
        index=i,
        persona=p.id,
        date=date,
        origin=origin,
        destination=p.destination,
        mode=p.mode,
        commodity=p.commodity,
        pieces=prng.randint(settings.variables.pieces.min, settings.variables.pieces.max),
        weight_kg=prng.randint(settings.variables.weight_kg.min, settings.variables.weight_kg.max),
        origin_omitted=origin_omitted,
        change_reason=change_reason,
    )


def build_facts(personas: list[Persona], settings: GenerationSettings) -> list[Fact]:
    timeline = settings.timeline
    drafts: list[FactDraft] = []
    for p in personas:
        prng = Random(f"{settings.seed}-{p.id}")
        date = timeline.start_date + datetime.timedelta(
            days=prng.randint(timeline.start_offset_days.min, timeline.start_offset_days.max)
        )
        for i in range(1, settings.emails_per_persona + 1):
            drafts.append(_make_row(p, i, date, prng, settings))
            date += datetime.timedelta(
                days=prng.randint(timeline.gap_days.min, timeline.gap_days.max)
            )

    drafts.sort(key=lambda d: (d.date, d.persona))
    return [Fact(n=n, **dict(draft)) for n, draft in enumerate(drafts, 1)]


def render(model: BaseChatModel, facts: list[Fact], personas: list[Persona]) -> None:
    persona_by_id = {p.id: p for p in personas}
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FACTS_PATH, "w") as f:
        for fact in facts:
            f.write(fact.model_dump_json() + "\n")
            path = EMAILS_DIR / f"{fact.key}.eml"
            if path.exists():
                continue
            email = render_email(model, fact, persona_by_id[fact.persona])
            with open(path, "w") as f_email:
                f_email.write(email)


def main():
    with open(WORLD_PATH, "r") as f:
        data = yaml.safe_load(f)
    config = ConfigFile.model_validate(data)

    facts = build_facts(config.personas, config.generation)

    renderer = config.generation.renderer
    model = create_model(model=renderer.model, temperature=renderer.temperature)

    render(model, facts, config.personas)


if __name__ == "__main__":
    main()
