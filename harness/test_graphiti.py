import asyncio

from harness.ingest import graphiti  # reuse the configured instance


async def find_customer_uuid(name: str) -> str:
    """Resolve a customer entity node to its UUID by exact name.

    Uses a direct Cypher query. This is deterministic and free: no LLM
    call, no embedding. Fine for the harness; the agent loop later
    resolves the sender the same way (the From header gives the exact
    company name).
    """
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (n:Entity:Customer {name: $name}) RETURN n.uuid AS uuid",
        name=name,
    )
    if not records:
        raise LookupError(f"No Customer node named {name!r}")
    return records[0]["uuid"]


async def recall(customer: str, question: str):
    """RECALL: search scoped to one customer via center node reranking."""
    uuid = await find_customer_uuid(customer)
    results = await graphiti.search(question, center_node_uuid=uuid, num_results=3)
    print(f"--- {customer}: {question}")
    for r in results:
        print(f"{r.fact}\n  valid: {r.valid_at}  invalid: {r.invalid_at}\n")


async def check():
    await recall("ACME Alimentos SAS", "Where does ACME Alimentos ship from?")
    await recall("Nordfrost Seafood AB", "Where does Nordfrost ship to?")


asyncio.run(check())
