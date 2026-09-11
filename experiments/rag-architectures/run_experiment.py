"""Runs all 6 RAG architectures over the same sample of MultiHop-RAG
queries, against one shared corpus index (default chunking strategy:
recursive, see chunking/strategies.py), and writes one combined
results/records.json for the Streamlit app to read.

Run with `python3 -u run_experiment.py` so progress prints show up
immediately rather than buffering until the process exits.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from chunking.strategies import DEFAULT_STRATEGY, STRATEGIES
from data import load_corpus, load_queries, stratified_sample
from metrics import exact_match, faithfulness_heuristic, retrieval_mrr, retrieval_precision_at_k, retrieval_recall_at_k, token_f1
from retrieval import build_index, is_gold_chunk
from techniques import ARCHITECTURES

SAMPLE_SIZE = 80
RANDOM_SEED = 0
RESULTS_PATH = Path(__file__).parent / "results" / "records.json"


def build_record(architecture_key: str, module, query: dict, run) -> dict:
    is_gold_flags = [is_gold_chunk(chunk.text, query["evidence_facts"]) for chunk in run.retrieved_chunks]
    context_text = " ".join(chunk.text for chunk in run.retrieved_chunks)
    is_answerable = query["question_type"] != "null_query"

    return {
        "architecture": architecture_key,
        "architecture_name": module.NAME,
        "question_id": query["id"],
        "question": query["question"],
        "question_type": query["question_type"],
        "gold_answer": query["answer"],
        "predicted_answer": run.predicted_answer,
        "abstained": run.abstained,
        "exact_match": exact_match(run.predicted_answer, query["answer"]) if is_answerable else None,
        "f1": token_f1(run.predicted_answer, query["answer"]) if is_answerable else None,
        "faithfulness": faithfulness_heuristic(run.predicted_answer, context_text),
        "recall_at_k": retrieval_recall_at_k(is_gold_flags) if is_answerable else None,
        "precision_at_k": retrieval_precision_at_k(is_gold_flags) if is_answerable else None,
        "mrr": retrieval_mrr(is_gold_flags) if is_answerable else None,
        "retrieved_titles": [chunk.title for chunk in run.retrieved_chunks],
        "retrieved_chunk_count": len(run.retrieved_chunks),
        "llm_calls": run.llm_calls,
        "retrieval_calls": run.retrieval_calls,
        "retrieval_rounds": run.retrieval_rounds,
        "empty_retrievals": run.empty_retrievals,
        "empty_retrieval_rate": run.empty_retrieval_rate,
        "total_prompt_tokens": run.total_prompt_tokens,
        "total_completion_tokens": run.total_completion_tokens,
        "total_latency_seconds": run.total_latency_seconds,
        "time_to_first_token_seconds": run.time_to_first_token_seconds,
        "mean_context_payload_bytes": run.mean_context_payload_bytes,
        "wall_clock_seconds": run.wall_clock_seconds,
        "had_error": run.had_error,
        "steps": [asdict(step) for step in run.steps],
    }


def main() -> None:
    print("Loading MultiHop-RAG corpus and queries...")
    articles = load_corpus()
    all_queries = load_queries()
    sample = stratified_sample(all_queries, SAMPLE_SIZE, seed=RANDOM_SEED)
    print(f"Sampled {len(sample)} queries across {len(set(q['question_type'] for q in sample))} question types")

    print(f"Building shared corpus index ({DEFAULT_STRATEGY} chunking, {len(articles)} articles)...")
    index_started = time.time()
    index = build_index(articles, STRATEGIES[DEFAULT_STRATEGY])
    print(f"  {len(index.chunks)} chunks indexed in {time.time() - index_started:.0f}s")

    architectures_metadata = {
        key: {
            "name": module.NAME,
            "description": module.DESCRIPTION,
            "prompts": module.PROMPTS,
            "what_it_is": module.WHAT_IT_IS,
            "how_we_implemented_it": module.HOW_WE_IMPLEMENTED_IT,
            "when_its_useful": module.WHEN_ITS_USEFUL,
            "diagram": module.DIAGRAM,
        }
        for key, module in ARCHITECTURES.items()
    }

    records = []
    for architecture_key, module in ARCHITECTURES.items():
        print(f"\n{module.NAME}: {len(sample)} queries")
        started_at = time.time()
        for i, query in enumerate(sample, start=1):
            run = module.run(query, index)
            records.append(build_record(architecture_key, module, query, run))
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} ({time.time() - started_at:.0f}s elapsed)")
        print(f"  done in {time.time() - started_at:.0f}s")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "chunking_strategy": DEFAULT_STRATEGY,
                "chunk_count": len(index.chunks),
                "architectures": architectures_metadata,
                "runs": records,
            },
            indent=2,
        )
    )
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
