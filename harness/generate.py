import datetime
import time
from random import Random
from corvid.llm import create_model
from corvid.logging import make_logger
from harness.models import (
    Case,
    CaseDraft,
    Persona,
    GenerationSettings,
    ConfigFile,
    RendererSettings,
)
import yaml
from langchain_core.language_models import BaseChatModel

from harness.paths import EMAILS_DIR, CASES_PATH, WORLD_PATH
from harness.renderer import cache_key, email_raw, render_email, stored_key, validate_email

logger = make_logger("generate")


def _make_case(
    p: Persona,
    i: int,
    date: datetime.date,
    prng: Random,
    settings: GenerationSettings,
) -> CaseDraft:
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

    return CaseDraft(
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


def build_cases(personas: list[Persona], settings: GenerationSettings) -> list[Case]:
    """Build the full answer sheet: every persona's cases, interleaved by date.

    One PRNG per persona (seeded from seed + persona id) keeps each
    persona's draws stable when others change. Cases are numbered (`n`)
    only after the global date sort — the number is the send order the
    agent will see.
    """
    timeline = settings.timeline
    drafts: list[CaseDraft] = []
    for p in personas:
        prng = Random(f"{settings.seed}-{p.id}")
        date = timeline.start_date + datetime.timedelta(
            days=prng.randint(timeline.start_offset_days.min, timeline.start_offset_days.max)
        )
        for i in range(1, settings.emails_per_persona + 1):
            drafts.append(_make_case(p, i, date, prng, settings))
            date += datetime.timedelta(
                days=prng.randint(timeline.gap_days.min, timeline.gap_days.max)
            )

    drafts.sort(key=lambda d: (d.date, d.persona))
    cases = [Case(n=n, **dict(draft)) for n, draft in enumerate(drafts, 1)]
    logger.info(
        "cases built",
        cases=len(cases),
        personas=len(personas),
        seed=settings.seed,
        first_date=cases[0].date.isoformat(),
        last_date=cases[-1].date.isoformat(),
    )
    return cases


def render(
    model: BaseChatModel,
    cases: list[Case],
    personas: list[Persona],
    renderer: RendererSettings,
) -> None:
    """Write the answer sheet (cases.jsonl) and render each case to an email.

    The cache is content-keyed: a cached .eml is kept only when the render
    key it carries matches the current facts + prompt + model and it still
    validates against its case. A prompt or model change re-renders
    everything; deleting a file still re-renders that one. cases.jsonl is
    always rewritten in full.
    """
    persona_by_id = {p.id: p for p in personas}
    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    rendered = cached = 0
    with open(CASES_PATH, "w") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")
            path = EMAILS_DIR / f"{case.key}.eml"
            render_key = cache_key(case, persona_by_id[case.persona], renderer)
            if path.exists():
                eml = path.read_text()
                if stored_key(eml) != render_key:
                    logger.info(
                        "cached email has a stale render key, re-rendering",
                        key=case.key,
                    )
                else:
                    problems = validate_email(
                        email_raw(eml), case, persona_by_id[case.persona]
                    )
                    if not problems:
                        logger.debug("email cached", key=case.key)
                        cached += 1
                        continue
                    logger.warning(
                        "cached email fails validation, re-rendering",
                        key=case.key,
                        problems=problems,
                    )
            start = time.perf_counter()
            email = render_email(
                model, case, persona_by_id[case.persona], render_key=render_key
            )
            with open(path, "w") as f_email:
                f_email.write(email)
            rendered += 1
            logger.info(
                "email rendered",
                key=case.key,
                seconds=round(time.perf_counter() - start, 1),
            )
    logger.info(
        "render complete",
        rendered=rendered,
        cached=cached,
        cases_path=str(CASES_PATH),
        emails_dir=str(EMAILS_DIR),
    )


def main():
    """Load the world, build the cases, render the emails."""
    with open(WORLD_PATH, "r") as f:
        data = yaml.safe_load(f)
    config = ConfigFile.model_validate(data)
    logger.info(
        "world loaded",
        world=str(WORLD_PATH),
        personas=[p.id for p in config.personas],
        renderer_model=config.generation.renderer.model,
    )

    cases = build_cases(config.personas, config.generation)

    renderer = config.generation.renderer
    model = create_model(model=renderer.model, temperature=renderer.temperature)

    render(model, cases, config.personas, renderer)


if __name__ == "__main__":
    main()
