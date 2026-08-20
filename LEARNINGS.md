# Learnings

Quick notes to my future self. Details in [model-notes.md](model-notes.md).

## 2026-08-20 — higher-dimensional personas (design discussion, nothing built)

- "Graph vs vector" was a false choice — Graphiti's search is already hybrid (embeddings + BM25 + graph). Vector memory adds nothing here.
- Multiple SHIPS_TO edges coexist fine; the real bottleneck is `choose_fact`'s newest-wins, which would confidently fill whatever they shipped last.
- What I actually want: a probabilistic fill conditioned on *all* known fields of the current episode, not one hand-picked conditional.
- The simple answer is symbolic kNN at query time: score each past episode by how many of my known field-values it shares, vote on the missing field weighted by score. ~15 lines. No consolidation layer, no tendency nodes — that's the cached version I don't need at 30 episodes.
- Probabilities must be counts over still-valid edges, never LLM guesses. Counting only valid edges means supersession retracts evidence for free — staleness and statistics compose.
- Checked Graphiti source: the episode_mentions reranker counts *global* mentions, not conditioned on the current episode. The conditional vote isn't native — that's the edge of the tool, and finding it is kind of the point of this project.
- None of this is learnable until the generator has weighted lane tables — no conditional structure in the world means newest-wins looks just as good.
- No agent-memory library does conditional probabilistic fill (checked the 2026 crop: Mem0, Zep, Letta, MemoryOS, A-MEM, MAGMA — all retrieval + consolidation summaries). Classic ML solved it decades ago: pgmpy infers P(missing | known) from a fact table natively, sklearn predict_proba likewise. The glue — valid edges → dataframe — stays mine either way. At 30 rows the 15-line count still beats both.

## 2026-08-20 — renderer validation

- The renderer loves spelling numbers out — "twenty-one pieces", even
  "nine thousand four hundred thirty-six kilograms". A digits-only check
  flagged 19 of 30 perfectly good emails. num2words (per persona
  language) on both sides fixed it.
- Validate the raw render, not the .eml — the From header states the
  sender's name and email for free and would hide a missing signature.

## 2026-08-20 — anchoring identity on the sender email

- Graphiti fills `Contact.email` verbatim from the From line, first try, both personas — didn't even need the display-name fallback I built.
- Dropping the company string-match in fill felt scary but the center-node reranking already keeps other customers' facts out of the top 3.
- Questions don't need the customer name in them — the center node carries that context, exactly like DESIGN.md said all along.

## 2026-08-20 — the loop (extract, ask, learn, eval)

- Customers will get their own emails wrong, and this system will happily learn their mistakes. If I want to study that, I should script the mistakes into the world — random ones just wreck the answer key.
- Better to extract nothing than the wrong thing. Empty fields get recalled or asked about; wrong ones sneak past everything.
- "Source: email" doesn't mean it was actually in the email — the model guessed the company from the domain and it looked legit.
- Match customers by sender email, not company name. The company is often missing or spelled differently, and the string match just quietly finds nothing.
- Making ASK a langgraph interrupt was the right call — the agent has no idea who's answering, the harness just fills in ground truth.
- I need to be able to eyeball eval output — one line per field. The aggregate counts hid every bug I found today.

## 2026-08-19 — milestone 2, Graphiti ingestion (see model-notes.md)

- Thinking models don't work with Graphiti over Ollama's `/v1` — the think flag never arrives. Stick to instruct variants.
- MoE wins here: Graphiti fires 6–10 big prompts per episode, so prefill is everything.
- Better prompts beat better models — hardened docstrings fixed what a model upgrade didn't.
- Telling the model exactly how to phrase facts killed the dedup churn completely.
