"""Runs every architecture over the same sample of HotpotQA questions,
against the same per-question retrieval corpus, and writes one combined
results/records.json for the Streamlit app to read. Run with `python3 -u
run_experiment.py` so progress prints show up immediately rather than
buffering until the process exits.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from data import load_pool, stratified_sample
from metrics import exact_match, token_f1
from retrieval import build_corpus
from techniques import ARCHITECTURES

SAMPLE_SIZE = 40
RANDOM_SEED = 0
RESULTS_PATH = Path(__file__).parent / "results" / "records.json"


def build_record(architecture_key: str, module, example: dict, run) -> dict:
    return {
        "architecture": architecture_key,
        "architecture_name": module.NAME,
        "question_id": example["id"],
        "question": example["question"],
        "question_type": example["type"],
        "level": example["level"],
        "gold_answer": example["answer"],
        "predicted_answer": run.predicted_answer,
        "exact_match": exact_match(run.predicted_answer, example["answer"]),
        "f1": token_f1(run.predicted_answer, example["answer"]),
        "step_count": run.step_count,
        "tool_calls": run.tool_calls,
        "tool_errors": run.tool_errors,
        "empty_retrievals": run.empty_retrievals,
        "empty_retrieval_rate": run.empty_retrieval_rate,
        "handoffs": run.handoffs,
        "early_terminated": run.early_terminated,
        "state_overhead_bytes": run.state_overhead_bytes,
        "wall_clock_seconds": run.wall_clock_seconds,
        "total_prompt_tokens": run.total_prompt_tokens,
        "total_completion_tokens": run.total_completion_tokens,
        "total_latency_seconds": run.total_latency_seconds,
        "time_to_first_token_seconds": run.time_to_first_token_seconds,
        "mean_context_payload_bytes": run.mean_context_payload_bytes,
        "had_error": run.had_error,
        "steps": [asdict(step) for step in run.steps],
    }


def main() -> None:
    pool = load_pool("validation")
    sample = stratified_sample(pool, SAMPLE_SIZE, seed=RANDOM_SEED)
    print(f"Sampled {len(sample)} questions "
          f"({sum(1 for e in sample if e['type'] == 'bridge')} bridge, "
          f"{sum(1 for e in sample if e['type'] == 'comparison')} comparison)")

    print("Embedding per-question corpora...")
    corpora = {example["id"]: build_corpus(example) for example in sample}

    architectures_metadata = {
        key: {"name": module.NAME, "description": module.DESCRIPTION, "prompts": module.PROMPTS}
        for key, module in ARCHITECTURES.items()
    }

    records = []
    for architecture_key, module in ARCHITECTURES.items():
        print(f"\n{module.NAME}: {len(sample)} questions")
        started_at = time.time()
        for i, example in enumerate(sample, start=1):
            run = module.run(example, corpora[example["id"]])
            records.append(build_record(architecture_key, module, example, run))
            if i % 10 == 0:
                print(f"  {i}/{len(sample)} ({time.time() - started_at:.0f}s elapsed)")
        print(f"  done in {time.time() - started_at:.0f}s")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"architectures": architectures_metadata, "runs": records}, indent=2))
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
