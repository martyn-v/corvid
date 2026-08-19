"""RECALL: query the graph for what memory knows about one customer."""

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge


async def find_customer_uuid(graphiti: Graphiti, name: str) -> str:
    """Resolve a customer entity node to its UUID by exact name.

    Uses a direct Cypher query. This is deterministic and free: no LLM
    call, no embedding. The From header gives the exact company name.
    """
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (n:Entity:Customer {name: $name}) RETURN n.uuid AS uuid",
        name=name,
    )
    if not records:
        raise LookupError(f"No Customer node named {name!r}")
    return records[0]["uuid"]


async def recall(
    graphiti: Graphiti, customer: str, question: str, num_results: int = 3
) -> list[EntityEdge]:
    """Search scoped to one customer via center node reranking."""
    uuid = await find_customer_uuid(graphiti, customer)
    return await graphiti.search(question, center_node_uuid=uuid, num_results=num_results)
