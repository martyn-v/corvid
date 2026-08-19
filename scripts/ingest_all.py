"""Ingest every generated email into the graph, one episode per email.

Milestone 2 scaffolding: the agent loop (milestone 3) supersedes this by
learning one email at a time inside the loop.

Usage:
    uv run -m scripts.ingest_all
"""

import asyncio
import json

from corvid.config import graphiti_config
from corvid.logging import make_logger
from corvid.memory.graphiti import make_graphiti
from corvid.memory.learn import learn, parse_email
from harness.models import Case
from harness.paths import EMAILS_DIR, CASES_PATH

logger = make_logger("ingest")


async def main():
    logger.info("starting ingestion", cases_path=str(CASES_PATH))
    graphiti = make_graphiti(graphiti_config)
    logger.debug("building indices and constraints")
    await graphiti.build_indices_and_constraints()
    count = 0
    with open(CASES_PATH, "r") as f:
        for line in f:
            case = Case.model_validate(json.loads(line))

            body, date = parse_email((EMAILS_DIR / f"{case.key}.eml").read_text())
            logger.debug(
                "adding episode", case_key=case.key, reference_time=date.isoformat()
            )
            await learn(graphiti, case.key, body, date)
            count += 1
    logger.info("ingestion complete", cases_ingested=count)


if __name__ == "__main__":
    asyncio.run(main())
