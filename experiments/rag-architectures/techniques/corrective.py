"""Corrective RAG (CRAG): retrieve, then explicitly grade whether what came
back is actually relevant before ever generating an answer from it. If
enough chunks pass, generate normally from just those (knowledge
refinement: strip the irrelevant ones out rather than pass the whole
top-k through). If none pass, take a corrective action instead of
answering from weak evidence anyway.

Yan et al., 2024 (https://arxiv.org/abs/2401.15884) use a live web search
as that corrective action when local retrieval grades as insufficient.
This experiment has no web search (everything runs offline, against a
fixed local corpus), so the corrective action here is a broader
hybrid (dense + BM25) re-retrieval instead, the closest local analogue:
try a second, differently-mechanized search of the same corpus before
giving up. If that second pass still grades as insufficient, the run
abstains directly rather than generating from evidence it has already
graded as weak twice.
"""

import time

from retrieval import bm25_search, dense_search, reciprocal_rank_fusion
from techniques.common import (
    ANSWER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT,
    RunResult,
    generate_answer,
    grade_relevance,
    llm_step,
    retrieval_step,
)

NAME = "Corrective RAG (CRAG)"
DESCRIPTION = (
    "Retrieves, grades relevance of what came back, keeps only relevant chunks. If "
    "nothing is relevant, tries a broader hybrid re-retrieval; if that still finds "
    "nothing relevant, abstains directly instead of generating from weak evidence."
)
INITIAL_K = 5
CORRECTIVE_K = 8
PROMPTS = {
    "grader": GRADER_SYSTEM_PROMPT,
    "generator": ANSWER_SYSTEM_PROMPT,
}

WHAT_IT_IS = (
    "Every other architecture here hands the generator whatever retrieval returned and trusts "
    "the shared answer prompt's instruction to say 'Insufficient information.' when that's "
    "warranted. CRAG doesn't trust that instruction alone: it inserts an explicit grading step "
    "between retrieval and generation, an LLM call that judges each retrieved chunk as relevant "
    "or irrelevant to the question, *before* generation ever sees them. Only chunks that pass "
    "grading (Yan et al. call this 'knowledge refinement') go into the final prompt. If nothing "
    "passes, the original paper falls back to a live web search rather than trust local "
    "retrieval further; from [Yan et al., 2024](https://arxiv.org/abs/2401.15884)."
)
HOW_WE_IMPLEMENTED_IT = (
    "`dense_search()` retrieves an initial top 5. `grade_relevance()` asks the model to judge "
    "each of the 5 independently as relevant or irrelevant. If at least one is graded relevant, "
    "the run generates directly from just the relevant subset (knowledge refinement) and stops. "
    "If none are, it takes the corrective action: a broader hybrid dense+BM25 re-retrieval (top "
    "8 by reciprocal rank fusion, a different retrieval mechanism from the first pass, not just "
    "a bigger k), then grades that set too. If at least one of those passes, it generates from "
    "the relevant subset; if still none do, the run abstains directly, setting the predicted "
    "answer to 'Insufficient information.' without spending a 3rd LLM call generating from "
    "evidence it has already graded as weak twice."
)
WHEN_ITS_USEFUL = (
    "Reach for this when generating a confident-sounding wrong answer is more costly than "
    "occasionally saying 'I don't know', which is most production settings, and especially "
    "when the corpus is known to not cover every question it will be asked (MultiHop-RAG's "
    "null_query questions exist for exactly this reason). It costs the most of any architecture "
    "here on the hard path: 1 extra grading call always, and a full extra retrieval-plus-grading "
    "round on any query where the first pass grades poorly, so it's the most expensive "
    "architecture per query on average, and the one whose cost is least predictable ahead of "
    "time. It's a weaker fit under tight latency budgets, or when the corpus reliably covers "
    "the query distribution already, since the grading step then mostly confirms what "
    "generation alone would have gotten right anyway, for the price of an extra call every time."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> D["Dense search<br/>top-5"]
    D --> Gr1{"Grade relevance"}
    Gr1 -->|"≥1 relevant"| G1["Generate from<br/>relevant subset"]
    Gr1 -->|"0 relevant"| C["Corrective: hybrid<br/>re-retrieval top-8"]
    C --> Gr2{"Grade relevance"}
    Gr2 -->|"≥1 relevant"| G2["Generate from<br/>relevant subset"]
    Gr2 -->|"0 relevant"| Ab["Abstain directly:<br/>'Insufficient information.'"]
    G1 --> Ans["Predicted answer"]
    G2 --> Ans
"""


def _relevant_subset(chunks, verdicts):
    return [chunk for chunk, is_relevant in zip(chunks, verdicts) if is_relevant]


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    retrieval_started = time.perf_counter()
    initial_chunks = dense_search(index, query["question"], INITIAL_K)
    steps.append(retrieval_step("retriever", query["question"], initial_chunks, time.perf_counter() - retrieval_started))

    grade_call, verdicts = grade_relevance(query["question"], initial_chunks)
    steps.append(llm_step("grader", "grade initial retrieval", grade_call))
    relevant_chunks = _relevant_subset(initial_chunks, verdicts)
    retrieval_rounds = 1

    if not relevant_chunks:
        corrective_started = time.perf_counter()
        dense_chunks = dense_search(index, query["question"], CORRECTIVE_K)
        bm25_chunks = bm25_search(index, query["question"], CORRECTIVE_K)
        corrective_chunks = reciprocal_rank_fusion([dense_chunks, bm25_chunks], CORRECTIVE_K)
        steps.append(
            retrieval_step("corrective_retriever", query["question"], corrective_chunks, time.perf_counter() - corrective_started)
        )
        retrieval_rounds = 2

        grade_call_2, verdicts_2 = grade_relevance(query["question"], corrective_chunks)
        steps.append(llm_step("grader", "grade corrective retrieval", grade_call_2))
        relevant_chunks = _relevant_subset(corrective_chunks, verdicts_2)

    if not relevant_chunks:
        return RunResult(
            predicted_answer="Insufficient information.",
            steps=steps,
            retrieved_chunks=[],
            abstained=True,
            wall_clock_seconds=time.perf_counter() - started_at,
            retrieval_rounds=retrieval_rounds,
        )

    call, answer, abstained = generate_answer(query["question"], relevant_chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=relevant_chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=retrieval_rounds,
    )
