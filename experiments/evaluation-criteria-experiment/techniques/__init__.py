from techniques import (
    multi_agent_supervisor,
    plain_llm,
    single_agent_react,
    single_shot_rag,
)

# Ordered from fewest to most decision points, matching the shape spectrum
# this experiment tests: plain LLM (1) -> RAG (2) -> single agent (as many
# as it takes steps) -> multi-agent (that many, times roles).
SHAPES = {
    "plain_llm": plain_llm,
    "single_shot_rag": single_shot_rag,
    "single_agent_react": single_agent_react,
    "multi_agent_supervisor": multi_agent_supervisor,
}
