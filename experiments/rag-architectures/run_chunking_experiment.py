"""Secondary experiment: holds architecture fixed at Naive RAG and varies
only the chunking strategy, to isolate what chunking alone does to
retrieval and answer quality. Naive RAG is the right architecture to hold
fixed for this: it has no reranking or fusion step that could compensate
for a bad first-pass chunking choice, so whatever chunking does to
retrieval quality shows up directly in what the generator sees.

Run with `python3 -u run_chunking_experiment.py`.
"""

import json
import time
from pathlib import Path

from chunking.strategies import STRATEGIES, approx_token_count
from data import load_corpus, load_queries, stratified_sample
from metrics import exact_match, faithfulness_heuristic, retrieval_mrr, retrieval_precision_at_k, retrieval_recall_at_k, token_f1
from retrieval import build_index, is_gold_chunk
from techniques import naive

SAMPLE_SIZE = 32
RANDOM_SEED = 0
RESULTS_PATH = Path(__file__).parent / "results" / "chunking_records.json"


def build_record(strategy_key: str, query: dict, run) -> dict:
    is_gold_flags = [is_gold_chunk(chunk.text, query["evidence_facts"]) for chunk in run.retrieved_chunks]
    context_text = " ".join(chunk.text for chunk in run.retrieved_chunks)
    is_answerable = query["question_type"] != "null_query"
    return {
        "strategy": strategy_key,
        "question_id": query["id"],
        "question_type": query["question_type"],
        "predicted_answer": run.predicted_answer,
        "abstained": run.abstained,
        "exact_match": exact_match(run.predicted_answer, query["answer"]) if is_answerable else None,
        "f1": token_f1(run.predicted_answer, query["answer"]) if is_answerable else None,
        "faithfulness": faithfulness_heuristic(run.predicted_answer, context_text),
        "recall_at_k": retrieval_recall_at_k(is_gold_flags) if is_answerable else None,
        "precision_at_k": retrieval_precision_at_k(is_gold_flags) if is_answerable else None,
        "mrr": retrieval_mrr(is_gold_flags) if is_answerable else None,
    }


def main() -> None:
    print("Loading MultiHop-RAG corpus and queries...")
    articles = load_corpus()
    all_queries = load_queries()
    sample = stratified_sample(all_queries, SAMPLE_SIZE, seed=RANDOM_SEED)
    print(f"Sampled {len(sample)} queries for the chunking sweep")

    strategies_metadata = {}
    records = []
    for strategy_key, chunk_fn in STRATEGIES.items():
        print(f"\n=== {strategy_key} ===")
        index_started = time.time()
        index = build_index(articles, chunk_fn)
        chunk_lengths = [approx_token_count(c.text) for c in index.chunks]
        strategies_metadata[strategy_key] = {
            "chunk_count": len(index.chunks),
            "mean_chunk_tokens": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_tokens": min(chunk_lengths),
            "max_chunk_tokens": max(chunk_lengths),
        }
        print(
            f"  {len(index.chunks)} chunks, mean {strategies_metadata[strategy_key]['mean_chunk_tokens']:.0f} "
            f"tokens/chunk, indexed in {time.time() - index_started:.0f}s"
        )

        started_at = time.time()
        for i, query in enumerate(sample, start=1):
            run = naive.run(query, index)
            records.append(build_record(strategy_key, query, run))
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} ({time.time() - started_at:.0f}s elapsed)")
        print(f"  done in {time.time() - started_at:.0f}s")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"architecture": "naive", "strategies": strategies_metadata, "runs": records}, indent=2)
    )
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
