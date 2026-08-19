"""Benchmark Ollama models running the real Graphiti ingest pipeline.

For each model this runs graphiti.add_episode() on a real email from
harness/emails — once cold (all Ollama models freshly unloaded, so latency
includes model load) and once hot (model resident). Benchmark data is written
under group_id "benchmark" and wiped between calls, so both runs do identical
work against an empty group and your real graph is untouched.

Usage:
    uv run -m scripts.benchmark_models                  # benchmark DEFAULT_MODELS
    uv run -m scripts.benchmark_models gemma4:31b ...   # benchmark specific models
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from neo4j import GraphDatabase

from corvid.config import graphiti_config
from corvid.memory.graphiti import make_graphiti
from corvid.memory.learn import parse_email

OLLAMA = graphiti_config.llm_base_url.removesuffix("/v1")
NEO4J_AUTH = (graphiti_config.neo4j_user, graphiti_config.neo4j_password)
EMAILS_DIR = Path("harness/emails")
GROUP_ID = "benchmark"

DEFAULT_MODELS = [
    "qwen3.5:27b",
    "qwen3.5:9b",
    "gemma4:31b",
    "gemma4:12b",
    "qwen3:30b-a3b-instruct-2507-q8_0",
]


def make_benchmark_graphiti(model: str) -> Graphiti:
    """Central config, with the LLM swapped for the model under test."""
    return make_graphiti(replace(graphiti_config, llm_model=model, llm_small_model=model))


def unload_all_models() -> None:
    with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=30) as resp:
        loaded = [m["name"] for m in json.load(resp)["models"]]
    for model in loaded:
        subprocess.run(["ollama", "stop", model], check=True, capture_output=True)


def wipe_benchmark_group() -> int:
    """Delete all benchmark nodes; returns how many nodes existed."""
    driver = GraphDatabase.driver(
        graphiti_config.neo4j_uri, auth=NEO4J_AUTH, warn_notification_severity=None
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
    return parse_email(paths[0].read_text())


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
    graphiti = make_benchmark_graphiti(model)
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

    setup = make_benchmark_graphiti(models[0])
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
