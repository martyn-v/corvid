"""Benchmark Ollama models running the real Graphiti ingest pipeline.

For each model this runs graphiti.add_episode() on a real email from
harness/emails — once cold (all Ollama models freshly unloaded, so latency
includes model load) and once hot (model resident). Benchmark data is written
under group_id "benchmark" and wiped between calls, so both runs do identical
work against an empty group and your real graph is untouched.

Usage:
    uv run -m harness.benchmark                  # benchmark DEFAULT_MODELS
    uv run -m harness.benchmark gemma4:31b ...   # benchmark specific models
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from email import message_from_string, policy
from email.utils import parsedate_to_datetime
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EpisodeType
from neo4j import GraphDatabase

from harness.ingest import make_nothink_client

OLLAMA = "http://localhost:11434"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "corvidpass")
EMAILS_DIR = Path("harness/emails")
GROUP_ID = "benchmark"

DEFAULT_MODELS = [
    "qwen3.5:27b",
    "qwen3.5:9b",
    "gemma4:31b",
    "gemma4:12b",
    "qwen3:30b-a3b-instruct-2507-q8_0",
]


def make_graphiti(model: str) -> Graphiti:
    llm_config = LLMConfig(
        api_key="ollama",
        model=model,
        small_model=model,
        temperature=0,
        base_url=f"{OLLAMA}/v1",
    )
    return Graphiti(
        NEO4J_URI,
        *NEO4J_AUTH,
        llm_client=OpenAIGenericClient(
            config=llm_config,
            client=make_nothink_client("ollama", f"{OLLAMA}/v1"),
        ),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",
                embedding_model="nomic-embed-text-v2-moe:latest",
                embedding_dim=768,
                base_url=f"{OLLAMA}/v1",
            )
        ),
        cross_encoder=OpenAIRerankerClient(config=llm_config),
    )


def unload_all_models() -> None:
    with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=30) as resp:
        loaded = [m["name"] for m in json.load(resp)["models"]]
    for model in loaded:
        subprocess.run(["ollama", "stop", model], check=True, capture_output=True)


def wipe_benchmark_group() -> int:
    """Delete all benchmark nodes; returns how many nodes existed."""
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=NEO4J_AUTH, warn_notification_severity=None
    )
    with driver.session() as session:
        count = session.run(
            "MATCH (n {group_id: $gid}) DETACH DELETE n RETURN count(n)", gid=GROUP_ID
        ).single()[0]
    driver.close()
    return count


def load_email() -> tuple[str, datetime]:
    paths = sorted(EMAILS_DIR.glob("*.eml"))
    if not paths:
        sys.exit(f"no .eml files found in {EMAILS_DIR}")
    message = message_from_string(paths[0].read_text(), policy=policy.default)
    content = (
        f"From: {message['From']}\nSubject: {message['Subject']}\n\n"
        f"{message.get_content().strip()}"
    )
    return content, parsedate_to_datetime(message["Date"])


async def timed_add_episode(graphiti: Graphiti, content: str, date) -> dict:
    start = time.perf_counter()
    try:
        await graphiti.add_episode(
            name="benchmark-episode",
            episode_body=content,
            source_description="customer email",
            reference_time=date,
            source=EpisodeType.text,
            group_id=GROUP_ID,
        )
        error = None
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - start
    nodes = wipe_benchmark_group()
    return {"latency": latency, "error": error, "nodes": nodes}


async def bench_model(model: str, content: str, date) -> tuple[dict, dict]:
    graphiti = make_graphiti(model)
    try:
        unload_all_models()
        cold = await timed_add_episode(graphiti, content, date)
        hot = await timed_add_episode(graphiti, content, date)
    finally:
        await graphiti.close()
    return cold, hot


async def main(models: list[str]) -> None:
    content, date = load_email()

    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=30) as resp:
        installed = {m["name"] for m in json.load(resp)["models"]}

    setup = make_graphiti(models[0])
    await setup.build_indices_and_constraints()
    await setup.close()
    wipe_benchmark_group()

    results = []
    for model in models:
        if model not in installed:
            print(f"skipping {model}: not installed", file=sys.stderr)
            continue
        print(f"benchmarking {model} ...", file=sys.stderr)
        cold, hot = await bench_model(model, content, date)
        results.append((model, cold, hot))
        for label, r in (("cold", cold), ("hot", hot)):
            status = r["error"] or f"ok, {r['nodes']} nodes"
            print(f"  {label}: {r['latency']:.1f}s ({status})", file=sys.stderr)

    header = f"{'model':<22} {'cold (s)':>9} {'hot (s)':>9} {'nodes':>6}  status"
    print(f"\n{header}")
    print("-" * (len(header) + 20))
    for model, cold, hot in results:
        status = cold["error"] or hot["error"] or "ok"
        print(
            f"{model:<22} {cold['latency']:>9.1f} {hot['latency']:>9.1f}"
            f" {hot['nodes']:>6}  {status}"
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or DEFAULT_MODELS))
