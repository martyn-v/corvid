from pathlib import Path

# Ground truth the agent never sees.
WORLD_PATH = Path("harness/world.yaml")
FACTS_PATH = Path("harness/facts.jsonl")

# The world/agent contract: the generator writes here, the agent reads here.
EMAILS_DIR = Path("harness/emails")
