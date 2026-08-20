# Learnings

Quick notes to my future self. Details in [model-notes.md](model-notes.md).

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
