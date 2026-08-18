# Corvid

**Does a freight agent learn its customers the way a human operator
does?** Corvid simulates customer relationships over time. Synthetic
customers send freight requests in a sequence. An agent collects
knowledge about each customer in a temporal knowledge graph. The
knowledge makes the agent better at the next request.

The name comes from the crows. Corvids remember individual humans for
years.

## Status

**Design phase.** This repo holds a complete design document, an example
persona, and no code. Read [DESIGN.md](DESIGN.md).

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

## Lineage

- [Squawkbox](https://github.com/martyn-v/squawkbox): one state, one
  event, one decision, one diff. Static.
- [Freightcase](https://github.com/martyn-v/freightcase): one email, one
  case, one human gate. Reactive.
- [Aviary](https://github.com/martyn-v/aviary): many shipments, a
  running clock, no agent. Alive.
- **Corvid**: many emails, many customers, a memory that grows and
  revises. Longitudinal.
