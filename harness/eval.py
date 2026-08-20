"""Run the agent graph over every generated email in the harness.

Usage:
    uv run -m harness.eval [--cleanup]

Eval data lives in the "eval" group, wiped at the start of every run so
each run is cold. Pass --cleanup to also wipe it when the run completes;
by default it is kept for browsing in Neo4j until the next run.
"""

import asyncio
import json
import sys
import time
from collections import Counter

import yaml
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from corvid.agent.graph import build_graph
from corvid.logging import make_logger
from corvid.memory.graphiti import make_graphiti, GraphitiMemory, wipe_group
from corvid.config import graphiti_config
from corvid.llm import create_model
from harness.customer import answer_question
from harness.models import Case, ConfigFile
from harness.paths import CASES_PATH, EMAILS_DIR, WORLD_PATH
from harness.report import render_case
from harness.score import score_case


GROUP_ID = "eval"

logger = make_logger("eval")


async def main(cleanup: bool = False):
    model = create_model(format="json", temperature=0)
    graphiti = make_graphiti(graphiti_config)
    memory = GraphitiMemory(graphiti, group_id=GROUP_ID)
    graph = build_graph(model, memory, checkpointer=InMemorySaver())

    wiped = await wipe_group(graphiti.driver, GROUP_ID)
    if wiped:
        logger.info("group wiped for cold start", group=GROUP_ID, nodes=wiped)

    with open(CASES_PATH) as f:
        cases = [Case.model_validate(json.loads(line)) for line in f]
    with open(WORLD_PATH) as f:
        world = ConfigFile.model_validate(yaml.safe_load(f))
    personas = {p.id: p for p in world.personas}
    logger.info(
        "eval starting",
        cases=len(cases),
        personas=list(personas),
        cases_path=str(CASES_PATH),
        world=str(WORLD_PATH),
        group=GROUP_ID,
    )

    run_start = time.perf_counter()
    totals: Counter[str] = Counter()
    questions_total = 0
    for case in cases:
        logger.info(
            "case starting",
            key=case.key,
            n=case.n,
            of=len(cases),
            persona=case.persona,
            date=case.date.isoformat(),
        )
        case_start = time.perf_counter()
        config: RunnableConfig = {"configurable": {"thread_id": case.key}}
        result = await graph.ainvoke(
            {"file_path": str(EMAILS_DIR / f"{case.key}.eml")}, config
        )
        if "__interrupt__" in result:
            questions = result["__interrupt__"][0].value
            # One entry per asked path, None when unanswered — an empty dict
            # would not resolve the interrupt (see ask_node).
            answers = {
                path: answer_question(case, personas[case.persona], path)
                for path in questions
            }
            logger.info(
                "agent asked, customer answered",
                key=case.key,
                paths=list(questions),
                answered=[p for p, a in answers.items() if a is not None],
            )
            result = await graph.ainvoke(Command(resume=answers), config)

        qr = result["quote_request"]
        scores = score_case(case, qr, result["provenance"])
        totals.update(scores.values())
        asked = result.get("asked", [])
        questions_total += len(asked)
        logger.info(
            "case scored",
            key=case.key,
            scores=dict(Counter(scores.values())),
            questions=len(asked),
            seconds=round(time.perf_counter() - case_start, 1),
        )
        print(render_case(case, qr, result["provenance"], scores))
        if asked:
            print(f"  ? asked {len(asked)}: {', '.join(e.path for e in asked)}")
        print()

    logger.info(
        "eval complete",
        totals=dict(totals),
        questions=questions_total,
        cases=len(cases),
        seconds=round(time.perf_counter() - run_start, 1),
    )
    print(f"\nScore totals: {dict(totals)}")
    print(f"Questions asked: {questions_total} over {len(cases)} cases")

    if cleanup:
        wiped = await wipe_group(graphiti.driver, GROUP_ID)
        logger.info("group cleaned up", group=GROUP_ID, nodes=wiped)


if __name__ == "__main__":
    asyncio.run(main(cleanup="--cleanup" in sys.argv[1:]))
