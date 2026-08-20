"""Run the agent graph over every generated email in the harness."""

import asyncio
import json
from collections import Counter

from corvid.agent.graph import build_graph
from corvid.memory.graphiti import make_graphiti, GraphitiMemory
from corvid.config import graphiti_config
from corvid.llm import create_model
from harness.models import Case
from harness.paths import CASES_PATH, EMAILS_DIR
from harness.report import render_case
from harness.score import score_case


async def main():
    model = create_model(format="json", temperature=0)
    memory = GraphitiMemory(make_graphiti(graphiti_config))
    graph = build_graph(model, memory)

    with open(CASES_PATH) as f:
        cases = [Case.model_validate(json.loads(line)) for line in f]

    totals: Counter[str] = Counter()
    for case in cases:
        result = await graph.ainvoke({"file_path": str(EMAILS_DIR / f"{case.key}.eml")})
        qr = result["quote_request"]
        scores = score_case(case, qr, result["provenance"])
        totals.update(scores.values())
        print(render_case(case, qr, result["provenance"], scores))
        print()

    print(f"\nScore totals: {dict(totals)}")


if __name__ == "__main__":
    asyncio.run(main())
