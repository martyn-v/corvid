"""Ingest every generated email into the graph, one episode per email.

Milestone 2 scaffolding: the agent loop (milestone 3) supersedes this by
learning one email at a time inside the loop.

Usage:
    uv run -m scripts.ingest_all
"""

import asyncio
import json

from corvid.agent.extract import render_email_prompt
from corvid.agent.parse_email import parse_eml
from corvid.config import graphiti_config
from corvid.logging import make_logger
from corvid.memory.graphiti import GraphitiMemory, make_graphiti
from harness.models import Case
from harness.paths import EMAILS_DIR, CASES_PATH

logger = make_logger("ingest")

GROUP_ID = "ingest"  # keeps the browse graph separate from eval's


async def main():
    logger.info("starting ingestion", cases_path=str(CASES_PATH))
    graphiti = make_graphiti(graphiti_config)
    memory = GraphitiMemory(graphiti, group_id=GROUP_ID)
    logger.debug("building indices and constraints")
    await graphiti.build_indices_and_constraints()
    count = 0
    with open(CASES_PATH, "r") as f:
        for line in f:
            case = Case.model_validate(json.loads(line))

            parsed = parse_eml((EMAILS_DIR / f"{case.key}.eml").read_bytes())
            if parsed.date is None:
                raise ValueError(f"{case.key}: email has no Date header")
            logger.debug(
                "adding episode",
                case_key=case.key,
                reference_time=parsed.date.isoformat(),
            )
            await memory.learn(case.key, render_email_prompt(parsed), parsed.date)
            count += 1
    logger.info("ingestion complete", cases_ingested=count)


if __name__ == "__main__":
    asyncio.run(main())
