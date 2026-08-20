# Corvid

**Does a freight agent learn its customers the way a human operator
does?** Corvid feeds a stream of customer emails to an agent. The agent
stores what it learns about each customer in a temporal knowledge graph.
The knowledge makes the agent better at the next email.

The name comes from the crows. Corvids remember individual humans for
years.

## Why this exists

Long-term memory is the one agentic capability that the sibling projects
(Squawkbox, Freightcase, the Aviary design) do not touch. The goal is to
learn a memory tool, not to build one. The interesting questions are not
about storage. They are about trust: when to recall a fact, when to fill
a gap from memory, and when to detect that a remembered fact is stale.

## The design

Small on purpose. Each piece earns its place only when it changes what
the memory must do. If the project is fun, it grows toward the ambitions
at the end of this document. It does not start there.

### The stack

- **graphiti-core** as a Python dependency, in-process. Graphiti is the
  open-source temporal knowledge graph engine from Zep: LLM-driven
  entity extraction, typed edges with validity intervals, and
  supersession on contradiction.
- **Neo4j** as the one container in a docker compose file: Bolt port
  7687, browser port 7474, a data volume. The Neo4j browser is the
  graph viewer; Corvid builds no UI.
- Graphiti needs an LLM and an embedder for extraction. Ollama works
  through the OpenAI-compatible endpoint. The model is a run parameter.

### The data

A minimal generator, not hand-written emails and not the full harness:

- **Two personas**, defined in one small YAML file. Each persona has a
  handful of flat facts: company, contact, origin, destination, mode,
  commodity. No pools, no weights, no style engine beyond a language
  and a one-line tone hint.
- **Omission is one rule.** Persona A states everything. Persona B
  omits its origin in most emails (a seeded coin flip per email).
- **One scripted change.** Persona B's origin changes at a fixed email
  index, and that email states the reason ("we moved our operation to
  Cartagena"). This is the supersession test.
- **A one-prompt renderer.** For each email: facts in, one LLM call,
  email out. The cache is "skip if the output file exists". No
  validation pass; eyeball the emails once.
- **Seeded throughout.** The same seed produces the same facts, the
  same omissions, and the same change index. About 15 emails per
  persona, interleaved by date.

The generator writes two files: the emails (what the agent sees) and
the facts per email (what the author knows). The second file is the
answer sheet for the milestones below; nothing is labeled by hand.

### The loop

One email at a time, in order:

```
  1. RECALL   query the graph: what do we know about this
              customer?
  2. EXTRACT  parse the email into quote request fields
              (Freightcase-style)
  3. FILL     fill gaps in the extraction from memory;
              mark the provenance "learned", never silently
  4. ASK      list the fields that are not stated and not
              known; the author answers, and the answer is
              ingested too
  5. LEARN    give Graphiti the raw email as an episode
```

### Decisions that hold at any scale

- **Raw episodes in, constrained by a declared ontology.** The LEARN
  step gives Graphiti the raw email. Corvid declares its ontology as
  custom entity and edge types (Pydantic models: Customer, Contact,
  Location; SHIPS_FROM, PREFERS_UNIT, CONTACT_IS). A pre-pass that
  feeds Graphiti pre-structured facts would reduce the tool to a
  database and defeat the goal of learning it.
- **Task extraction never feeds memory.** The quote fields are per
  episode and disposable. Graphiti reads the source email itself.
- **Answered questions become memory, tagged by source.** A direct
  answer to a direct question is the strongest fact. Edges carry a
  source tag: `email` or `answered_question`.
- **A fill from memory is a proposal with provenance.** What the email
  stated and what memory supplied are never blended silently. This is
  the Freightcase rule.
- **A recall question contains only what the stored facts contain.**
  Questions are asked one per missing field, phrased to mirror the
  ontology's fact templates ("Where does X ship from?" sits next to
  "X ships from Y" in embedding space), and keyed by field so the fill
  knows which facts answer which gap. The center node supplies the
  customer context; repeating it in the question adds nothing. Context
  the graph does not store — the known origin when asking for the
  destination — is not grounding but noise, steering retrieval toward
  the wrong edge type. If the ontology ever stores lanes, the facts
  will contain origin and destination together, and only then does the
  question grow to match. Ontology first; the question follows it.

### Milestones

1. **Generate.** The two-persona YAML, the seeded generator, the
   one-prompt renderer.
   Done when: the same seed produces the same ~30 emails, the emails
   read plausibly, and persona B's origin is absent where the coin flip
   said so.
2. **Ingest and browse.** Docker compose up, graphiti-core installed,
   the emails ingested, one graph in the Neo4j browser.
   Done when: the customers, locations, and relations are visible as a
   graph, and the change email closed the old origin edge and opened a
   new one.
3. **The loop.** The five steps above as a small script over the same
   emails.
   Done when: an email that omits the origin gets a fill from memory
   with `learned` provenance, and the questions get fewer as the
   sequence progresses.
4. **Stale-memory check.** Emails after the change email.
   Done when: a fill after the change uses the new origin, not the old
   one.

That is the project. Everything below is what it can become.

## Future ambitions

These pieces exist as designs, not commitments. Each one enters only
when the small version makes it feel worth it.

- **Richer personas and world generation.** Omission probabilities per
  fact, scripted changes as a list, style parameters, many personas,
  lane portfolios (a stable export base plus a weighted destination
  pool) so memory must learn the difference between a fact and a
  tendency. See `docs/persona-example.yaml` for the grown shape.
- **A better renderer.** A content-keyed cache (facts + prompt + model
  hashed) so re-renders after a prompt change stay cheap, and a
  validation pass that checks each email against its source facts.
- **An auto-answer oracle.** The harness answers the agent's questions
  from the ground truth, so a full run executes unattended and runs are
  comparable across memory designs.
- **Free grading.** A graph diff against the ground-truth world (facts
  learned, missed, invented). Temporal correctness: each fill checked
  against what was true at that time; a fill with the old value after a
  scripted change is a stale-memory failure. The headline chart:
  questions asked per episode, falling as memory works.
- **Richer personas.** Lane portfolios (a stable export base plus a
  weighted destination pool) instead of single defaults, so memory
  must learn the difference between a fact and a tendency.
- **A comparison lever.** `--episode-body raw|extracted`: feed Graphiti
  the raw email or the agent's validated extraction, and measure the
  difference on the same stream.
- **The picture.** A 3D force graph of the memory, animated over an
  episode replay, next to the falling questions curve.

## Anti-goals

- No hand-labeled data and no hand-written emails. The generator makes
  both the emails and the answer sheet.
- No model training and no fine-tuning. Learning means graph writes.
- No production TMS features. No real customer data.
- No harness feature before the memory needs it.
- One flow (quote requests). This is the Freightcase scoping rule.

## Known risks

- **Graphiti extraction quality on local models.** Graphiti runs its
  own LLM for entity and fact extraction. Small local models are a
  known weak spot for this step. The Graphiti model is a run parameter,
  so an API model is one flag away.

## Lineage

- **Squawkbox**: one state, one event, one decision, one diff. Static.
- **Freightcase**: one email, one case, one human gate. Reactive.
- **Aviary**: many shipments, a running clock, no agent. Alive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
