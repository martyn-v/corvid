from email import message_from_string, policy
from email.utils import parsedate_to_datetime

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

EMAILS_DIR = Path("harness/emails")

# Configure Ollama LLM client
llm_config = LLMConfig(
    api_key="ollama",  # Ollama doesn't require a real API key, but some placeholder is needed
    model="qwen2.5:14b-instruct",
    small_model="qwen2.5:14b-instruct",
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
)

llm_client = OpenAIGenericClient(config=llm_config)

# Initialize Graphiti with Ollama clients
graphiti = Graphiti(
    "bolt://localhost:7687",
    "neo4j",
    "corvidpass",
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",  # Placeholder API key
            embedding_model="nomic-embed-text-v2-moe:latest",
            embedding_dim=768,
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
