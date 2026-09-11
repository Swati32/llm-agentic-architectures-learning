"""HyDE: instead of embedding the raw query, first ask the model to write a
short hypothetical passage that *would* answer the question, then embed
that hypothetical passage and search with it. The passage is never shown
to anyone and is very likely factually wrong; it doesn't need to be right,
it only needs to be phrased the way a real answer would be phrased, so its
embedding lands closer to the real supporting passage in the corpus than
the original question's embedding does.

Grounded in Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (https://arxiv.org/abs/2212.10496).
"""

import time

from retrieval import dense_search_by_vector, get_embedder
from techniques.common import (
    ANSWER_SYSTEM_PROMPT,
    HYDE_SYSTEM_PROMPT,
    RunResult,
    generate_answer,
    generate_hypothetical_passage,
    llm_step,
    retrieval_step,
)

NAME = "HyDE"
DESCRIPTION = (
    "Generates a hypothetical answer passage first, embeds that instead of the raw "
    "query, then searches with it. Two LLM calls: write the hypothetical passage, then "
    "generate the real answer."
)
TOP_K = 5
PROMPTS = {
    "hypothesis_writer": HYDE_SYSTEM_PROMPT,
    "generator": ANSWER_SYSTEM_PROMPT,
}

WHAT_IT_IS = (
    "A question and its answer are often phrased very differently: 'who criticized the "
    "merger' doesn't share much vocabulary with the sentence that actually names the critic and "
    "explains why. Dense retrieval embeds the query as-is, so it's stuck matching question-shaped "
    "text against answer-shaped text. HyDE's fix, from [Gao et al., 2022]"
    "(https://arxiv.org/abs/2212.10496): ask the model to hallucinate a plausible answer passage "
    "first, in the same style real supporting text would be written in, then embed *that* "
    "instead of the question. The hypothetical passage will often get facts wrong, that's fine "
    "and expected, it never gets shown to anyone or fact-checked; the embedding step only cares "
    "that it's phrased like a real answer, which puts it closer in embedding space to the real "
    "supporting passage than the original question's phrasing was."
)
HOW_WE_IMPLEMENTED_IT = (
    "`generate_hypothetical_passage()` asks the model for a short (3-4 sentence) passage that "
    "would answer the question, written as if it were a news snippet. That passage, not the "
    "question, is embedded with `all-MiniLM-L6-v2` and searched against the corpus with "
    "`dense_search_by_vector()`, the top 5 chunks going into the same shared `generate_answer()` "
    "prompt every architecture uses. Two LLM calls total: one to write the hypothesis, one to "
    "generate the real, grounded answer from whatever the hypothesis's embedding actually found."
)
WHEN_ITS_USEFUL = (
    "Reach for this when queries are phrased in a register that doesn't match how the corpus "
    "states its answers, questions as questions, answers as declarative statements, which is "
    "close to always true, or when the query uses different words for the same concept the "
    "source text uses. It costs one extra LLM call over naive RAG, and that call runs before "
    "retrieval, on the critical path, so it adds real latency. It's a weaker fit when the model "
    "confidently hallucinates a hypothesis that's not just factually wrong but *topically* "
    "wrong, since retrieval then searches for evidence of the wrong topic entirely; and it buys "
    "nothing extra on a query where the raw query already closely resembles the source phrasing."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> H["LLM: write hypothetical<br/>answer passage"]
    H --> E["Embed hypothetical passage"]
    E --> D["Dense search<br/>top-5 chunks"]
    D --> G["Generate answer<br/>from real passages"]
    G --> Ans["Predicted answer"]
"""


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    hyde_call, hypothetical_passage = generate_hypothetical_passage(query["question"])
    steps.append(llm_step("hypothesis_writer", "write hypothetical passage", hyde_call))

    search_text = hypothetical_passage or query["question"]
    embedder = get_embedder()
    query_embedding = embedder.encode([search_text], normalize_embeddings=True)[0]

    retrieval_started = time.perf_counter()
    chunks = dense_search_by_vector(index, query_embedding, TOP_K)
    steps.append(retrieval_step("retriever", search_text, chunks, time.perf_counter() - retrieval_started))

    call, answer, abstained = generate_answer(query["question"], chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=1,
    )
