import datetime
import time
from random import Random
from corvid.llm import create_model
from corvid.logging import make_logger
from harness.models import Fact, FactDraft, Persona, GenerationSettings, ConfigFile
import yaml
from langchain_core.language_models import BaseChatModel

from harness.paths import EMAILS_DIR, FACTS_PATH, WORLD_PATH
from harness.renderer import render_email

logger = make_logger("generate")


def _make_row(
    p: Persona,
    i: int,
    date: datetime.date,
    prng: Random,
    settings: GenerationSettings,
) -> FactDraft:
    """Draw one email's ground-truth facts for persona `p`.

    Applies the scripted origin change once `i` reaches it, and the seeded
    omission coin flip — except on the first and change emails, which must
    state the origin.
    """
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
    """Build the full answer sheet: every persona's facts, interleaved by date.

    One PRNG per persona (seeded from seed + persona id) keeps each
    persona's draws stable when others change. Facts are numbered (`n`)
    only after the global date sort — the number is the send order the
    agent will see.
    """
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
    facts = [Fact(n=n, **dict(draft)) for n, draft in enumerate(drafts, 1)]
    logger.info(
        "facts built",
        facts=len(facts),
        personas=len(personas),
        seed=settings.seed,
        first_date=facts[0].date.isoformat(),
        last_date=facts[-1].date.isoformat(),
    )
    return facts


def render(model: BaseChatModel, facts: list[Fact], personas: list[Persona]) -> None:
    """Write the answer sheet (facts.jsonl) and render each fact to an email.

    The cache rule is "skip if the .eml exists": delete a file to re-render
    it. facts.jsonl is always rewritten in full.
    """
    persona_by_id = {p.id: p for p in personas}
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    rendered = cached = 0
    with open(FACTS_PATH, "w") as f:
        for fact in facts:
            f.write(fact.model_dump_json() + "\n")
            path = EMAILS_DIR / f"{fact.key}.eml"
            if path.exists():
                logger.debug("email cached", key=fact.key)
                cached += 1
                continue
            start = time.perf_counter()
            email = render_email(model, fact, persona_by_id[fact.persona])
            with open(path, "w") as f_email:
                f_email.write(email)
            rendered += 1
            logger.info(
                "email rendered",
                key=fact.key,
                seconds=round(time.perf_counter() - start, 1),
            )
    logger.info(
        "render complete",
        rendered=rendered,
        cached=cached,
        facts_path=str(FACTS_PATH),
        emails_dir=str(EMAILS_DIR),
    )


def main():
    """Load the world, build the facts, render the emails."""
    with open(WORLD_PATH, "r") as f:
        data = yaml.safe_load(f)
    config = ConfigFile.model_validate(data)
    logger.info(
        "world loaded",
        world=str(WORLD_PATH),
        personas=[p.id for p in config.personas],
        renderer_model=config.generation.renderer.model,
    )

    facts = build_facts(config.personas, config.generation)

    renderer = config.generation.renderer
    model = create_model(model=renderer.model, temperature=renderer.temperature)

    render(model, facts, config.personas)


if __name__ == "__main__":
    main()
