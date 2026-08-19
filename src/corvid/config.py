from corvid.memory.graphiti import GraphitiConfig

# Local dev stack: Neo4j + Ollama (OpenAI-compatible endpoint).
graphiti_config = GraphitiConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="corvidpass",
    llm_api_key="ollama",
    llm_model="qwen3:30b-a3b-instruct-2507-q8_0",
    llm_small_model="qwen3:30b-a3b-instruct-2507-q8_0",
    llm_temperature=0,
    llm_base_url="http://localhost:11434/v1",
    embedding_model="mxbai-embed-large",
    embedding_dim=1024,
)
