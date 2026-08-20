"""RECALL: query the graph for what memory knows about one customer."""

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge


async def find_contact_uuid(graphiti: Graphiti, email: str) -> str:
    """Resolve the sender's Contact node to its UUID by exact email.

    Uses a direct Cypher query. This is deterministic and free: no LLM
    call, no embedding. The From line gives the exact address, and it is
    the only identifier guaranteed present in every email.
    """
    records, _, _ = await graphiti.driver.execute_query(
        "MATCH (n:Entity:Contact {email: $email}) RETURN n.uuid AS uuid",
        email=email,
    )
    if not records:
        raise LookupError(f"No Contact node with email {email!r}")
    return records[0]["uuid"]


async def recall(
    graphiti: Graphiti, email: str, question: str, num_results: int = 3
) -> list[EntityEdge]:
    """Search scoped to one sender via center node reranking."""
    uuid = await find_contact_uuid(graphiti, email)
    return await graphiti.search(question, center_node_uuid=uuid, num_results=num_results)
