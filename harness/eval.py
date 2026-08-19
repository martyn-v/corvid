"""Run the agent graph over every generated email in the harness."""

import asyncio
import json

from corvid.agent.graph import build_graph
from corvid.llm import create_model
from harness.models import Case
from harness.paths import CASES_PATH, EMAILS_DIR


async def main():
    model = create_model(format="json", temperature=0)
    graph = build_graph(model)

    with open(CASES_PATH) as f:
        cases = [Case.model_validate(json.loads(line)) for line in f]

    for case in cases:
        result = await graph.ainvoke({"file_path": str(EMAILS_DIR / f"{case.key}.eml")})
        qr = result["quote_request"]
        print(f"{case.key}: {qr.origin} -> {qr.destination}")


if __name__ == "__main__":
    asyncio.run(main())
