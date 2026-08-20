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
the prose does — and a cheap validation pass keeps them agreeing: every
rendered email is checked against its case (each fact stated, the
origin absent when the coin omitted it), and the render is retried
until it passes.

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
Recall anchors on the sender's address (`Contact.email`) — the only
identifier guaranteed present in every email; the company is a derived
fact, reachable from the contact via `WORKS_FOR`.

**Grading is free.** [harness/eval.py](harness/eval.py) replays all
thirty emails against a cold graph and diffs the result against the
answer sheet, one line per field: `correct`, `wrong`, `missing`,
`hallucinated`, or `stale` — a fill marked `source: email` for a fact
the email omitted counts as hallucinated even when the value happens to
be right, and a fill matching the pre-change origin after the change
email is stale, not plain wrong. It also emits the headline number:
**questions asked per episode, per persona, in send order — falling as
memory works**. Every run writes a JSONL artifact to `harness/runs/`
(scores, values, provenance, questions per case) so runs are comparable
across memory designs.

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

Rendered emails are cached on disk; delete one to re-render it. A
cached email that no longer validates against its case is re-rendered
automatically. Eval
runs in the `eval` graph group, wiped at the start of each run so every
run is cold; pass `--cleanup` to also wipe it afterwards. Utility
scripts live in [scripts/](scripts/): recall smoke checks, group wipes,
and an Ollama model benchmark against the real Graphiti ingest.

## Roadmap

- **Give the world more to omit.** Today origin is the only omittable
  fact, and the generator forces email 1 to state it — so memory always
  knows the answer before the first omission and a legitimate question
  is impossible by construction. Mode and commodity are stable
  per-customer facts sitting in the world but absent from
  `QuoteRequest`. Plan: add them to the contract as
  `required_for_quote`, give personas per-fact omission (the grown
  shape in [docs/persona-example.yaml](docs/persona-example.yaml)), and
  grow the ontology to match — one edge type per new fact; the recall
  question follows the ontology. First emails state everything, later
  ones get terse: the relationship arc that makes the questions curve
  fall.
- **Content-keyed render cache.** The cache is "skip if the file
  exists", so prompt or model changes require deleting emails by hand.
  Key it on facts + prompt + model instead.
- **Score more than two fields.** Only `origin.name` and
  `destination.name` are graded; requester name/email/company are
  `required_for_quote` and drive the ask loop but are never scored.
- **Graph diff against ground truth.** The other half of free grading:
  facts learned, missed, invented — straight from Neo4j vs world.yaml.

### Probabilistic fill

Real customers are higher-dimensional: one ships from A to B for
commodity a, but A to C for commodity b. The fill should stop being
newest-wins and become a probabilistic answer conditioned on everything
the current email states — "given this origin and this commodity, 7 of
9 past shipments went to Rotterdam." No agent-memory library does this
natively (verified against Graphiti's source and the 2026 crop:
Mem0, Zep, Letta, MemoryOS, A-MEM, MAGMA); it is a small mechanism
Corvid builds on top, and the probabilities are counts over still-valid
edges, never LLM guesses — so supersession retracts evidence for free.
In dependency order:

1. **Weighted lanes in the world.** Personas get
   `lanes: [{origin, commodity, destination, weight}]`; each email
   draws one; the answer sheet records the draw. Without conditional
   structure in the world, nothing downstream is testable.
2. **Ontology captures the conditioning fields.** The vote can only
   condition on what the graph stores: add a `Commodity` entity and
   edge, maybe mode. Per-episode noise (quantity, weight) stays out of
   memory on purpose.
3. **Multi-valued `SHIPS_TO`.** Docstring tweak so a new destination
   does not supersede the old ones, plus the mirror of the current
   supersession test: old destination edges stay open. Likeliest pain
   point.
4. **Episode-level recall.** New memory-port method: this customer's
   episodes plus their still-valid edges (the `entity_edges` hop). One
   Cypher query.
5. **The vote.** Replace newest-wins in `choose_fact`: score each past
   episode by how many of the current email's known field-values it
   shares, vote on the missing field weighted by score, fill past a
   threshold. Provenance carries the tally (7/9).
6. **Grade it.** Fill accuracy against the drawn lane; later,
   calibration of the learned probabilities against the lane weights.

## Lineage

- [Squawkbox](https://github.com/martyn-v/squawkbox): one state, one
  event, one decision, one diff. Static.
- [Freightcase](https://github.com/martyn-v/freightcase): one email, one
  case, one human gate. Reactive.
- [Aviary](https://github.com/martyn-v/aviary): many shipments, a
  running clock, no agent. Alive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
