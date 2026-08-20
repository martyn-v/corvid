from pathlib import Path

# Ground truth the agent never sees.
WORLD_PATH = Path("harness/world.yaml")
CASES_PATH = Path("harness/cases.jsonl")

# The world/agent contract: the generator writes here, the agent reads here.
EMAILS_DIR = Path("harness/emails")

# One JSONL artifact per eval run, for comparing runs across memory designs.
RUNS_DIR = Path("harness/runs")
