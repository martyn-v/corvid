from dataclasses import dataclass
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from openai import AsyncOpenAI
from datetime import datetime
from graphiti_core.nodes import EpisodeType
from corvid.memory.ontology import edge_type_map, edge_types, entity_types
from corvid.memory.port import Memory, RecalledFact


@dataclass
class GraphitiConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_api_key: str
    llm_model: str
    llm_small_model: str
    llm_temperature: float
    llm_base_url: str
    embedding_model: str
    embedding_dim: int


def _make_nothink_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Create an LLM client using OpenAI but set reasoning_effort to none so we dont use thinking for Ollama models"""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    orig_create = client.chat.completions.create

    async def create(*args, **kwargs):
        kwargs.setdefault("extra_body", {})["reasoning_effort"] = "none"
        return await orig_create(*args, **kwargs)

    client.chat.completions.create = create  # type: ignore # instance-level wrap
    return client


def make_graphiti(config: GraphitiConfig) -> Graphiti:
    # Configure Ollama LLM client
    llm_config = LLMConfig(
        api_key=config.llm_api_key,  # Ollama doesn't require a real API key, but some placeholder is needed
        model=config.llm_model,
        small_model=config.llm_small_model,
        temperature=config.llm_temperature,
        base_url=config.llm_base_url,  # Ollama's OpenAI-compatible endpoint
    )

    llm_client = OpenAIGenericClient(
        config=llm_config,
        client=_make_nothink_client(config.llm_api_key, config.llm_base_url),
    )

    # Initialize Graphiti with Ollama clients
    graphiti = Graphiti(
        config.neo4j_uri,
        config.neo4j_user,
        config.neo4j_password,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=config.llm_api_key,  # Placeholder API key
                embedding_model=config.embedding_model,
                embedding_dim=config.embedding_dim,
                base_url=config.llm_base_url,
            )
        ),
        cross_encoder=OpenAIRerankerClient(config=llm_config),
    )
    return graphiti


class GraphitiMemory:
    """A memory implementation that uses Graphiti for recalling and learning facts."""

    def __init__(self, graphiti: Graphiti):
        self.graphiti = graphiti

    async def recall(self, customer: str, question: str) -> list[RecalledFact]:
        records, _, _ = await self.graphiti.driver.execute_query(
            "MATCH (n:Entity:Customer {name: $name}) RETURN n.uuid AS uuid",
            name=customer,
        )

        if not records:
            return []  # cold start: memory knows nothing yet
        edges = await self.graphiti.search(
            question, center_node_uuid=records[0]["uuid"], num_results=3
        )
        return [
            RecalledFact(fact=e.fact, uuid=e.uuid, valid_at=e.valid_at) for e in edges
        ]

    async def learn(self, name: str, body: str, date: datetime) -> None:
        await self.graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description="customer email",
            reference_time=date,
            source=EpisodeType.text,  # default is message; text fits emails
            entity_types=entity_types,
            edge_types=edge_types,
            edge_type_map=edge_type_map,
            excluded_entity_types=["Entity"],
        )
