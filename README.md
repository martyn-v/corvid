# Corvid

**Does a freight agent learn its customers the way a human operator
does?** Corvid simulates customer relationships over time. Synthetic
customers send freight requests in a sequence. An agent collects
knowledge about each customer in a temporal knowledge graph. The
knowledge makes the agent better at the next request.

The name comes from the crows. Corvids remember individual humans for
years.

## Status

**Exploring.** This repo holds a complete design document, and some very rough harness code to generate emails, ingest them into Graphiti and manually verify the output. Read [DESIGN.md](DESIGN.md).

## The idea in five lines

- A seeded generator creates **ground truth first**: about 15 customer
  personas with facts, habits, omission rates, and scripted changes. No
  person labels anything, ever.
- An LLM **renders** each episode's facts into a plausible email. The
  LLM never authors facts; the answer key exists before the prose does.
- The agent handles one email at a time: **recall** what the graph knows,
  **extract** the request, **fill** gaps from memory with marked
  provenance, **ask** about the rest, and **learn** the episode into the
  graph. Answered questions become memory too.
- The memory is [Graphiti](https://github.com/getzep/graphiti), a
  temporal knowledge graph: typed edges with validity intervals, and
  supersession when a customer changes.
- Grading is free: the harness diffs the graph against the ground truth,
  checks each fill against what was true at that time, and plots the
  headline chart: **questions asked per episode, falling as memory
  works**.

## What the finished thing shows

A 3D graph that grows on screen while episodes replay. Nodes and edges
appear when the agent learns. An edge flips when a customer changes. Next
to it: the questions curve, going down.

## Roadmap

- **Anchor identity on the sender's email, not the company name.**
  The company is a *derived* fact: emails without a signature have no
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

## Lineage

- [Squawkbox](https://github.com/martyn-v/squawkbox): one state, one
  event, one decision, one diff. Static.
- [Freightcase](https://github.com/martyn-v/freightcase): one email, one
  case, one human gate. Reactive.
- [Aviary](https://github.com/martyn-v/aviary): many shipments, a
  running clock, no agent. Alive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
