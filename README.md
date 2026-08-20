# Corvid

**Does a freight agent learn its customers the way a human operator
does?** Corvid simulates customer relationships over time. Synthetic
customers send freight quote requests in a sequence. An agent collects
knowledge about each customer in a temporal knowledge graph. The
knowledge makes the agent better at the next request.

The name comes from the crows. Corvids remember individual humans for
years.

## Status

**Working end-to-end, small world.** The full loop runs: a seeded
generator produces ground-truth emails, the agent handles them one at a
time against a [Graphiti](https://github.com/getzep/graphiti) memory on
Neo4j, and the harness scores every field against the answer key.
Everything runs locally on Ollama. The world is deliberately tiny — two
personas, thirty emails — enough to exercise omission, recall, and one
scripted change. [DESIGN.md](DESIGN.md) holds the full design and the
larger ambitions.

## How it works

**Ground truth first.** [harness/world.yaml](harness/world.yaml)
defines two personas: `nordfrost` states everything (the control);
`acme-alimentos` omits its origin on a seeded coin flip and moves city
at email 10 (the supersession test). A seeded generator draws each
persona's 15 cases — dates, weights, omissions, the change — into a
JSONL answer sheet. No person labels anything, ever.

**An LLM renders, never authors.** Each case's facts are rendered into
a plausible `.eml` email by a local model. The answer key exists before
the prose does.

**The agent is a LangGraph state graph**
([src/corvid/agent/graph.py](src/corvid/agent/graph.py)):

- **parse** the raw email, **extract** a structured `QuoteRequest`
  (requester, origin, destination), every present field tagged with
  provenance `source: email`
- **recall** missing fields from the knowledge graph and **fill** them
  with provenance `source: memory`
- **ask** the customer about whatever is still missing — a LangGraph
  interrupt; the harness answers from ground truth, so the agent never
  knows who's on the other end
- **learn**: the raw email becomes a Graphiti episode, and answered
  questions become a second episode. The extracted request itself never
  feeds memory.

**The memory is Graphiti**, a temporal knowledge graph over Neo4j, with
a custom ontology ([src/corvid/memory/ontology.py](src/corvid/memory/ontology.py)):
`Customer`, `Contact`, and `Location` entities; `SHIPS_FROM`,
`SHIPS_TO`, and `WORKS_FOR` edges with validity intervals, so a
customer's move supersedes the old fact instead of contradicting it.

**Grading is free.** [harness/eval.py](harness/eval.py) replays all
thirty emails against a cold graph and diffs the result against the
answer sheet, one line per field: `correct`, `wrong`, `missing`, or
`hallucinated` — a fill marked `source: email` for a fact the email
omitted counts as hallucinated even when the value happens to be right.
It also counts the headline number: **questions asked per episode,
falling as memory works**.

## Running it

Requires [mise](https://mise.jdx.dev) (Python + uv), Docker, and a
local [Ollama](https://ollama.com) with the models named in
[harness/world.yaml](harness/world.yaml) and
[src/corvid/config.py](src/corvid/config.py).

```sh
docker compose up -d        # Neo4j (browser at localhost:7474)
uv run -m harness.generate  # build the answer sheet, render the emails
uv run -m harness.eval      # run the agent over every email, score it
uv run pytest               # tests
```

Rendered emails are cached on disk; delete one to re-render it. Eval
runs in the `eval` graph group, wiped at the start of each run so every
run is cold; pass `--cleanup` to also wipe it afterwards. Utility
scripts live in [scripts/](scripts/): recall smoke checks, group wipes,
and an Ollama model benchmark against the real Graphiti ingest.

## Roadmap

- **Anchor identity on the sender's email, not the company name.**
  The company is a _derived_ fact: emails without a signature have no
  extractable company, and even when extraction infers one (e.g.
  `acme-alimentos` from the domain), recall does an exact string match
  against the Customer node name (`ACME Alimentos SAS`) and silently
  misses. The sender address is the only identifier guaranteed present
  in every email. Plan:
  - Give the `Contact` entity an `email` attribute in the ontology
    (same pattern as `Location.locode`); the address sits verbatim in
    the From line of every episode.
  - Anchor recall with `MATCH (n:Entity:Contact {email: $email})` and
    use that node as `center_node_uuid`; key `recall_missing_fields` on
    `requester.email` instead of `requester.company`.
  - Keep the edge ontology unchanged — `SHIPS_FROM`/`SHIPS_TO` stay on
    Customer, reachable via `WORKS_FOR` (two hops from the centroid),
    so colleagues at the same company still share knowledge.
  - Verify graphiti reliably populates `Contact.email`; fallback is
    matching the Contact by From display-name.
- **Per-episode question series.** Eval prints one aggregate count; the
  headline claim is questions *falling over the sequence*. Emit
  questions-per-episode, per persona, in send order.
- **A `stale` score category.** A fill using the pre-change origin
  after the change email currently scores plain `wrong`. Check the fill
  against what was true at that time and name the failure.
- **Persisted run artifact.** Eval only prints. Write scores,
  provenance, and questions to a JSONL per run so runs are comparable
  across memory designs.
- **Renderer validation pass.** Nothing checks a rendered email states
  its facts — or omits the origin when the coin said so. One cheap
  check per email against its case, so the answer key can't silently
  disagree with the prose.
- **Content-keyed render cache.** The cache is "skip if the file
  exists", so prompt or model changes require deleting emails by hand.
  Key it on facts + prompt + model instead.
- **Score more than two fields.** Only `origin.name` and
  `destination.name` are graded; requester name/email/company are
  `required_for_quote` and drive the ask loop but are never scored.
- **Graph diff against ground truth.** The other half of free grading:
  facts learned, missed, invented — straight from Neo4j vs world.yaml.

## Lineage

- [Squawkbox](https://github.com/martyn-v/squawkbox): one state, one
  event, one decision, one diff. Static.
- [Freightcase](https://github.com/martyn-v/freightcase): one email, one
  case, one human gate. Reactive.
- [Aviary](https://github.com/martyn-v/aviary): many shipments, a
  running clock, no agent. Alive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
