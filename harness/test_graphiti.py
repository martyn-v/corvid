import asyncio
from harness.ingest import graphiti  # reuse your configured instance


async def check():
    results = await graphiti.search("Where does ACME Alimentos ship from?")
    for r in results:
        print(f"{r.fact}\n  valid: {r.valid_at}  invalid: {r.invalid_at}\n")


asyncio.run(check())
