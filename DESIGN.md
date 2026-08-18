# Corvid

**Does a freight agent learn its customers the way a human operator
does?** Corvid simulates customer relationships over time. Synthetic
customers send freight requests in a sequence. An agent collects
knowledge about each customer in a temporal knowledge graph. The
knowledge makes the agent better at the next request.

The name comes from the crows. Corvids remember individual humans for
years.

Status: **design draft**. The design is under iteration. No code exists.

## The mechanics in one paragraph

The stream contains many customers, interleaved. Learning means the
agent writes and revises facts in a graph. The agent does not train a
model. The agent must keep customers apart, recall the correct facts at
the correct time, and detect when a fact stops being true.

## Why this exists

Long-term memory is the one agentic capability that the sibling projects
(Squawkbox, Freightcase, the Aviary design) do not touch. The interesting
questions are not about storage. They are about trust: when to recall a
fact, when to fill a gap from memory, and when to detect that a
remembered fact is stale. A knowledge graph that grows on screen is also
demonstrable. Batch memory systems are not.

## Design rule zero: no hand-labeling

The generator creates the ground truth first, as structured data, from
seeded templates. The renderer creates prose from the ground truth
afterward. The answer key exists before any email exists. No person
labels anything.

The direction is one-way: structure in, prose out. An LLM renders facts
into emails. The LLM never authors facts. The reverse direction (prose to
labels) is hand-verification with extra steps. Corvid never uses it to
make ground truth.

## The system as a whole

The system has two halves. The **harness** generates and grades, offline.
The **agent** is the system under study.

### The harness

```
personas.yaml ──> world generator ──> ground-truth world
 (templates)        (seeded)          (facts + timed changes)
                                            │
                                            v
                                     episode renderer (LLM)
                                            │
                                            v
                                  episodes.jsonl (emails, in order)
```

- **Personas** (about 15). Each persona is a customer with facts and
  habits: a default origin, a weight-unit habit ("tons" means metric), a
  contact person, a language, a verbosity level, a typo rate.
- **Timed changes** are part of the persona: "port changes to COCTG at
  episode 30", "contact leaves at episode 45". These changes are the
  memory-hygiene tests. Each change costs one line of YAML.
- **The renderer** turns the facts of one episode into one plausible
  email. Style parameters differ per persona. Rendering runs offline, is
  seeded where possible, and is cached. The cost is small: 15 personas x
  50 episodes is about 750 short generations. A local Ollama instance
  completes this overnight.
- **No human is in the loop.** The ground truth is the oracle. When the
  agent asks a question, the harness answers from the world. A full run
  executes unattended. Runs are replayable and comparable across memory
  designs.

### The agent: the loop for one episode

```
INPUT:  episode #23: email from ACME

  1. RECALL   query the graph: what do we know about ACME?
  2. EXTRACT  parse the email into the task schema
              (Freightcase-style quote request fields)
  3. FILL     fill gaps in the extraction from memory;
              mark the provenance "learned", never silently
  4. ACT      emit the structured quote request, plus
              questions for each field that is not stated
              and not known
  5. LEARN    write new, confirmed, or superseding facts to
              the graph with timestamps; a contradiction
              supersedes the old edge and flags the change;
              answered questions are ingested too, tagged
              by source
```

- **Input**: an ordered stream of rendered emails, one at a time. The
  agent cannot look ahead.
- **Actions per episode**: recall, extract, fill, act, learn.
- **Outputs**, three kinds:
  1. Per episode: the filled request and the questions asked.
  2. Cumulative: the knowledge graph (nodes, edges, timestamps,
     provenance).
  3. Meta: metrics over the run.

## The memory

The memory is a temporal knowledge graph:

- **Nodes**: customers, contacts, locations, commodities, preferences.
- **Edges**: typed relations (SHIPS_FROM, PREFERS_UNIT, CONTACT_IS).
  Each edge carries a validity interval and a provenance (the episode
  that taught it).
- **Supersession, not deletion.** When a new fact contradicts an old
  fact, the old edge closes its validity interval and the new edge opens
  one. History stays queryable: "what did we believe in episode 25?"
- A fill from memory is a proposal with provenance. This is the
  Freightcase rule: what the email stated and what memory supplied are
  never blended silently.

**Decision: Graphiti.** The goal of this project is to learn a memory
tool, not to build one. Graphiti is the open-source temporal graph
memory engine (from Zep). It supplies the machinery the design needs:
entity extraction, typed edges, validity intervals, and supersession on
contradiction. The harness grades any memory implementation, so a
hand-rolled comparison stays possible later. It is not a goal.

**Decision: answered questions become memory, tagged by source.** When
the agent asks a question and the harness answers it (as the customer,
from the ground truth), the answer is ingested as an episode too. The
agent learns from the conversation, not only from the inbound email. A
direct answer to a direct question is the strongest fact: it deserves
more confidence than a passively stated one, not less.

Every edge carries a source tag: `email` or `answered_question`. The tag
keeps the grading clean: the questions curve falls partly because asking
itself teaches, and the graph diff can report the two sources
separately. The tag also exposes the interesting trade-off: a question
has a cost (it bothers the customer) and a payoff (a permanent
high-confidence fact).

**Decision: raw episodes in, constrained by a declared ontology.** The
LEARN step gives Graphiti the raw email as an episode. Corvid declares
its ontology as custom entity and edge types (Pydantic models: Customer,
Contact, Location; SHIPS_FROM, PREFERS_UNIT, CONTACT_IS) and passes it
at ingestion, so extraction is steered toward the schema instead of
free-form. A pre-pass that feeds Graphiti pre-structured facts would
reduce the tool to a database and defeat the goal of learning it.

Two jobs stay separate: task extraction (quote fields, per episode,
disposable) never feeds memory. Graphiti reads the source email itself.

One fallback lever, kept cheap: Graphiti also accepts structured JSON
episodes. If raw-email extraction is too noisy on local models, a run
parameter (`--episode-body raw|extracted`) feeds the agent's validated
task extraction as the episode instead. The two modes are then a
measurable comparison on the same case stream, not a redesign.

**Architecture: graphiti-core in-process, Neo4j in docker compose.**
Graphiti is a Python library, not a required service. Corvid imports
graphiti-core directly and connects to a Neo4j container over Bolt. The
compose file holds one service: neo4j with the Bolt port (7687), the
browser port (7474), and a data volume. A Graphiti server container
exists but adds a middleware layer this project does not need; current
practice is to use the library directly.

Graphiti needs an LLM and an embedder for its extraction. The default is
OpenAI. Ollama works through the OpenAI-compatible endpoint. Extraction
quality with small local models is a known weak spot. Stage 3 measures
this instead of assuming it; the model behind Graphiti is a run
parameter, like the agent model in Squawkbox.

## Grading

Grading is free. The harness supplies it:

- **Graph diff**: compare the final graph with the ground-truth world.
  Count facts learned, missed, and invented.
- **Temporal correctness**: compare the fills of each episode with the
  facts that were true at that time. A fill with the old port after the
  scripted change is a stale-memory failure. The harness detects it
  automatically.
- **The headline chart**: questions asked per episode. The curve goes
  down when memory works. Memory that works makes the agent stop asking.

## What is demonstrable

- A 3D force-directed graph that grows on screen during a replay. Nodes
  and edges appear when the agent learns. An edge flips when a customer
  changes. This picture is the reason the project exists.
- The questions-per-episode curve.
- A replay of one episode: the email on one side; the recall, the fills,
  and the questions on the other side.

## Anti-goals

- No hand-labeled data, ever.
- No model training and no fine-tuning. Learning means graph writes.
- No production TMS features. No real customer data.
- One flow (quote requests) done well before a second flow starts. This
  is the Freightcase scoping rule.
- The renderer is a renderer. A renderer that invents facts is a bug,
  not emergent realism.

## Known risks

- **Renderer fact-leakage.** The LLM can drop or change a fact when it
  writes the email ("12 pallets" becomes "a dozen"). Mitigations: a
  cheap validation pass that checks the email against its source facts,
  or acceptance of the leakage with grading against what the email
  communicated. Real customers change facts too.
- **Graphiti extraction quality on local models.** Graphiti runs its own
  LLM for entity and fact extraction. Small local models are a known
  weak spot for this step. Mitigation: the Graphiti model is a run
  parameter, so runs can compare a local model against an API model. The
  harness grades the result either way.
- **Same-voice syndrome.** Each email sounds like the same LLM. Persona
  style parameters and varied models or temperatures reduce this. This
  risk applies to realism-feel, not to the measurements.

## Stages

### Stage 1: the world and the stream

- Persona templates and a seeded world generator with timed changes.
- An LLM renderer with a cache and a validation pass.
- Output: an episodes.jsonl that reads plausibly and diffs cleanly
  against its ground truth.

Done when: the same seed produces the same world and the same cached
episodes, and a spot-check of 20 emails contains their source facts.

### Stage 2: the agent, without memory

- The episode loop without RECALL, FILL, and LEARN: extract, act, ask.
- The auto-answer machinery: the harness answers questions from the
  world.
- Baseline metrics: the questions-per-episode curve stays flat, by
  construction.

Done when: a full 50-episode run executes unattended and produces the
baseline curve.

### Stage 3: memory

- Graphiti as the memory engine: raw-email episode ingestion with the
  declared ontology, recall and fill with provenance, and supersession
  on contradiction.
- Compare the `--episode-body raw|extracted` modes if raw extraction is
  noisy.
- The run now produces a falling questions curve and a growing graph.

Done when: the curve falls visibly, the graph diff scores well against
the ground truth, and the agent catches at least one scripted change
(the port change) with a supersession instead of stale fills.

### Stage 4: the picture

- A 3D force graph of the memory, animated over the episode replay.
- An episode inspector: the email, the recall, the fills, and the
  questions, side by side.

Done when: a run is fun to watch. This is the same finish line as
Aviary.

## Open questions

None at design level. One implementation note for stage 1: check how
much persona vocabulary ports from the lane templates of Squawkbox
(locations, parties, references).

## Lineage

- **Squawkbox**: one state, one event, one decision, one diff. Static.
- **Freightcase**: one email, one case, one human gate. Reactive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
