"""Wipe one group's nodes from the graph.

Usage:
    uv run -m scripts.wipe_group           # wipes the "eval" group
    uv run -m scripts.wipe_group ingest    # wipes a named group
"""

import asyncio
import sys

from corvid.config import graphiti_config
from corvid.memory.graphiti import make_graphiti, wipe_group


async def main(group_id: str) -> None:
    graphiti = make_graphiti(graphiti_config)
    try:
        count = await wipe_group(graphiti.driver, group_id)
        print(f"Wiped {count} nodes from group '{group_id}'")
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval"))
