"""Manual smoke check of RECALL against an ingested graph.

Usage:
    uv run -m scripts.check_recall
"""

import asyncio

from corvid.config import graphiti_config
from corvid.memory.graphiti import make_graphiti
from corvid.memory.recall import recall

CHECKS = [
    ("ACME Alimentos SAS", "Where does ACME Alimentos ship from?"),
    ("Nordfrost Seafood AB", "Where does Nordfrost ship to?"),
]


async def check():
    graphiti = make_graphiti(graphiti_config)
    for customer, question in CHECKS:
        results = await recall(graphiti, customer, question)
        print(f"--- {customer}: {question}")
        for r in results:
            print(f"{r.fact}\n  valid: {r.valid_at}  invalid: {r.invalid_at}\n")


if __name__ == "__main__":
    asyncio.run(check())
