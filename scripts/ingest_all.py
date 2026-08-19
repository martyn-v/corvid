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
from harness.models import Fact
from harness.paths import EMAILS_DIR, FACTS_PATH

logger = make_logger("ingest")


async def main():
    logger.info("starting ingestion", facts_path=str(FACTS_PATH))
    graphiti = make_graphiti(graphiti_config)
    logger.debug("building indices and constraints")
    await graphiti.build_indices_and_constraints()
    count = 0
    with open(FACTS_PATH, "r") as f:
        for line in f:
            fact = Fact.model_validate(json.loads(line))

            body, date = parse_email((EMAILS_DIR / f"{fact.key}.eml").read_text())
            logger.debug(
                "adding episode", fact_key=fact.key, reference_time=date.isoformat()
            )
            await learn(graphiti, fact.key, body, date)
            count += 1
    logger.info("ingestion complete", facts_ingested=count)


if __name__ == "__main__":
    asyncio.run(main())
