"""Run the agent graph over every generated email in the harness."""

import asyncio
import json
from collections import Counter

import yaml
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from corvid.agent.graph import build_graph
from corvid.memory.graphiti import make_graphiti, GraphitiMemory
from corvid.config import graphiti_config
from corvid.llm import create_model
from harness.customer import answer_question
from harness.models import Case, ConfigFile
from harness.paths import CASES_PATH, EMAILS_DIR, WORLD_PATH
from harness.report import render_case
from harness.score import score_case


async def main():
    model = create_model(format="json", temperature=0)
    memory = GraphitiMemory(make_graphiti(graphiti_config))
    graph = build_graph(model, memory, checkpointer=InMemorySaver())

    with open(CASES_PATH) as f:
        cases = [Case.model_validate(json.loads(line)) for line in f]
    with open(WORLD_PATH) as f:
        world = ConfigFile.model_validate(yaml.safe_load(f))
    personas = {p.id: p for p in world.personas}

    totals: Counter[str] = Counter()
    questions_total = 0
    for case in cases:
        config: RunnableConfig = {"configurable": {"thread_id": case.key}}
        result = await graph.ainvoke(
            {"file_path": str(EMAILS_DIR / f"{case.key}.eml")}, config
        )
        if "__interrupt__" in result:
            questions = result["__interrupt__"][0].value
            answers = {
                path: answer
                for path in questions
                if (answer := answer_question(case, personas[case.persona], path))
                is not None
            }
            result = await graph.ainvoke(Command(resume=answers), config)

        qr = result["quote_request"]
        scores = score_case(case, qr, result["provenance"])
        totals.update(scores.values())
        print(render_case(case, qr, result["provenance"], scores))
        asked = result.get("asked", [])
        questions_total += len(asked)
        if asked:
            print(f"  ? asked {len(asked)}: {', '.join(e.path for e in asked)}")
        print()

    print(f"\nScore totals: {dict(totals)}")
    print(f"Questions asked: {questions_total} over {len(cases)} cases")


if __name__ == "__main__":
    asyncio.run(main())
