from email import message_from_string, policy
from email.utils import parsedate_to_datetime
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
import json
from pathlib import Path
from harness.models import Fact
from graphiti_core.nodes import EpisodeType
from harness.utils import build_fact_key
from corvid.memory.ontology import entity_types, edge_types, edge_type_map

EMAILS_DIR = Path("harness/emails")

# Configure Ollama LLM client
llm_config = LLMConfig(
    api_key="ollama",  # Ollama doesn't require a real API key, but some placeholder is needed
    model="qwen3:30b-a3b-instruct-2507-q8_0",
    small_model="qwen3:30b-a3b-instruct-2507-q8_0",
    temperature=0,
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
)


def make_nothink_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Create an LLM client using OpenAI but set reasoning_effort to none so we dont use thinking for Ollama models"""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    orig_create = client.chat.completions.create

    async def create(*args, **kwargs):
        kwargs.setdefault("extra_body", {})["reasoning_effort"] = "none"
        return await orig_create(*args, **kwargs)

    client.chat.completions.create = create  # instance-level wrap
    return client


llm_client = OpenAIGenericClient(
    config=llm_config, client=make_nothink_client("ollama", "http://localhost:11434/v1")
)

# Initialize Graphiti with Ollama clients
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "corvidpass",
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",  # Placeholder API key
            embedding_model="mxbai-embed-large ",
            embedding_dim=1024,
            base_url="http://localhost:11434/v1",
        )
    ),
    cross_encoder=OpenAIRerankerClient(config=llm_config),
)


async def _process_fact(fact: Fact):
    fact_key = build_fact_key(fact)
    path = EMAILS_DIR / f"{fact_key}.eml"
    with open(path, "r") as f:
        email_content = f.read()

    message = message_from_string(email_content, policy=policy.default)
    sender = message["From"]
    date = parsedate_to_datetime(message["Date"])
    content = f"From: {sender}\nSubject: {message['Subject']}\n\n{message.get_content().strip()}"
    await graphiti.add_episode(
        name=fact_key,
        episode_body=content,
        source_description="customer email",
        reference_time=date,
        source=EpisodeType.text,  # default is message; text fits emails
        entity_types=entity_types,
        edge_types=edge_types,
        edge_type_map=edge_type_map,
        excluded_entity_types=["Entity"],
    )
    print("Episode added for fact:", fact_key)


async def main():
    await graphiti.build_indices_and_constraints()
    with open("facts.jsonl", "r") as f:
        for line in f:
            fact_data = json.loads(line)
            fact = Fact.model_validate(fact_data)
            await _process_fact(fact)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
