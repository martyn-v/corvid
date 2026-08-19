# Model notes: Graphiti ingestion with local models

Findings from milestone 2 (ingest and browse) on a Mac Studio M3 Ultra,
96 GB. Graphiti-core with Neo4j in docker compose. Ollama serves all
local models through the OpenAI-compatible endpoint
(`OpenAIGenericClient`).

The task: ingest ~30 short generated customer emails (2 personas, one
scripted origin change at ACME email 10) and produce a clean typed
temporal graph.

## Summary

1. Reasoning models break Graphiti through the Ollama `/v1` endpoint.
   The endpoint does not pass `think: false`; output lands in the
   `reasoning` field and the content is empty, or the model echoes the
   schema back.
2. MoE beats dense for this workload. Graphiti sends 6 to 10 large
   prompts per episode; prefill dominates. `qwen3:30b-a3b-instruct-2507`
   (3.3 B active) is faster than dense ~30 B models and extracts more.
3. Extraction quality is prompt-bound more than model-bound. Hardened
   type docstrings fixed classification errors that a better model did
   not.
4. Dedup and temporal resolution are the weakest steps. Fact-phrasing
   templates in the edge docstrings removed most churn; a residual
   defect rate remains and includes one silent wrong-node wiring.

## Configuration that works

```python
llm_config = LLMConfig(
    api_key="ollama",
    model="qwen3:30b-a3b-instruct-2507-q8_0",
    small_model="qwen3:30b-a3b-instruct-2507-q8_0",  # dedup judgments are not easier than extraction
    temperature=0,                                    # Graphiti defaults to 1; wrong for extraction
    base_url="http://localhost:11434/v1",
)
llm_client = OpenAIGenericClient(config=llm_config)  # default json_schema mode
# embedder: mxbai-embed-large via Ollama, embedding_dim=1024
```

- The default `json_schema` structured output mode works with this
  model on Ollama. The `json_object` fallback (schema injected into the
  prompt) exists for models that accept the json_schema request but do
  not honor it; it was part of the gpt-oss debugging, not the final
  config.
- The `client` constructor argument of `OpenAIGenericClient` accepts a
  wrapped `AsyncOpenAI`; use it to inject `extra_body` parameters such
  as `reasoning_effort` if a reasoning model must be used.
- Changing the embedder changes the embedding dimension. Old vectors
  and old Neo4j vector indexes are invalid after a swap:
  `docker compose down -v` is the clean reset.

## Model benchmark (single-episode extraction)

| model                            | cold (s) | hot (s) | nodes | note                                  |
| -------------------------------- | -------- | ------- | ----- | ------------------------------------- |
| qwen3:30b-a3b-instruct-2507-q8_0 | 17.7     | 13.0    | 9     | chosen default                        |
| qwen3.5:27b                      | 32.7     | 19.7    | 4     | dense, slow prefill                   |
| gemma4:31b                       | 37.1     | 15.7    | 5     | dense, slow prefill                   |
| gemma4:12b                       | 13.3     | 18.5    | 6     |                                       |
| qwen3.5:9b                       | 9.3      | 4.5     | 3     | fast, shallow                         |
| llama3.1:8b                      | 11.1     | 1.8     | 1     | disqualified: extracts almost nothing |

Node count is not a quality metric. The chosen model initially produced
9 nodes that included ephemera ("14 pieces") and misclassified entities.

Not yet benchmarked: gpt-oss:120b (fits in 96 GB; Graphiti's own Ollama
example uses it), gpt-oss:20b with the reasoning_effort wrapper, an API
reference model (Haiku) for a quality ceiling.

## Failure catalog and fixes

### Reasoning models return empty content

- gpt-oss:20b: `LLM returned an empty response` in the default
  json_schema mode; switched to json_object mode as a workaround and it
  echoed the JSON schema itself back as data.
- Root cause: Ollama's `/v1` endpoint does not forward `think: false`.
  `reasoning_effort` maps to the internal think field ("none" disables;
  gpt-oss accepts only low/medium/high) but is undocumented and
  version-sensitive.
- Practical rule: use non-thinking instruct variants for Graphiti.

### Classification scrambling (typed ontology, one-line docstrings)

Observed with the chosen model at temperature 0: the commodity typed as
Customer, a LOCODE typed as Contact, the actual company and contact not
extracted at all.

Fix: hardened docstrings with a concrete example, NOT-clauses, and an
attribute rule.

```python
class Contact(BaseModel):
    """A person's name: the human who signed or wrote the email.
    Example: 'Erik Lindqvist'. NOT a code, NOT a company, NOT a place."""
```

Result: correct Customer/Contact/Location classification; ephemera
("Ocean Reefer", commodity) stopped extracting without needing
`excluded_entity_types`.

### Missing Customer entity

The company was never extracted because the rendered emails never
stated it in the body (From header only carried the contact name).
Fix in the renderer, not the ontology: a brief introduction on first
contact and a signature block with the company name. Models extract
from text; they do not parse email domains.

### Dedup churn (self-supersession and duplicate open edges)

Same durable fact restated in every email exercises the dedup judge ~14
times per persona. Two failure flavors: a duplicate not recognized
(second open edge) and a restatement judged as a contradiction
(invalidate + reopen identical fact).

- Embedder swap (nomic-embed-text-v2-moe to mxbai-embed-large) did NOT
  fix the contradiction flavor: candidate retrieval was not the
  bottleneck, judgment was.
- Fact-phrasing templates in the edge docstrings mostly fixed it:
  "State the fact exactly as: '<customer name> ships from <location
  name>'." Identical strings make the judgment trivial.
- Residual suffix drift ("Gothenburg" vs "Gothenburg, Sweden") produced
  one duplicate; fixed by normalizing Location names in the type
  docstring (city name only, LOCODE as attribute). The final run had
  zero churn.

### Silent wrong-node wiring (worst defect class)

One intermediate run wired an edge with the fact text "ACME ships from
Cartagena, Colombia" to the Rotterdam node, and its creation
invalidated the correct Cartagena edge. Fact text right, graph
structure wrong, no warning. Dangerous because recall by fact text
still looks correct. Did not recur after name normalization, but this
defect class is worth checking for in any grading step: verify edges by
endpoints, not by fact text.

### Orphan edge drops

`Source/Target entity not found in nodes for edge relation` means the
edge extraction call named an entity that the node extraction call did
not produce (cross-call name inconsistency). The edge is dropped.
Reduced by temperature 0 and uniform naming; a residual rate is
tolerable because later emails re-extract the same relationship.

## Result

The milestone 2 done-criterion passed: ACME's SHIPS_FROM Bogotá edge
closed at the scripted change email's date and a Cartagena edge opened.

Final run after all fixes: zero defective SHIPS_FROM edges (no
duplicates, no self-supersession, no wrong-node wiring). Three edges,
all matching ground truth. The earlier residual rate (roughly 2
defective edges in 5) was eliminated by the fact-phrasing templates
plus Location name normalization (city name only, LOCODE as an
attribute).

Remaining orphan-edge warnings concentrate on episodes where the
persona omitted its origin: the edge extractor proposes SHIPS_FROM, no
location node exists in that episode, and the edge is dropped. This is
correct behavior for an unstated fact; the durable edge exists from the
first email.

## Operational notes

- Full ingest of 30 emails: 4 m 45 s (about 9.5 s per episode).
  Graphiti runs parts of its pipeline concurrently, so per-episode cost
  is far below calls x hot-latency. A full wipe-and-rerun is cheap
  enough to be part of the iteration loop.
- Make ingest resumable if runs get interrupted often: skip an episode
  if its Episodic node already exists.
- Every experiment starts from a clean graph: `MATCH (n) DETACH DELETE
n`, and after an embedder change also drop the vector indexes.
- Graphiti's semantic edge names live in the `name` property of
  `RELATES_TO` edges, not in the Neo4j relationship type.
