"""Runs every technique over the same stratified test sample and writes
per-query records to results/records.json for the Streamlit app to read."""

import json
from pathlib import Path

from data import load_examples, stratified_sample
from resources import build_resources
from techniques import TECHNIQUES

TEST_SAMPLE_SIZE = 200
RESULTS_PATH = Path(__file__).parent / "results" / "records.json"


def build_record(technique_key: str, example: dict, run) -> dict:
    return {
        "technique": technique_key,
        "text": example["text"],
        "gold_intent": example["intent"],
        "gold_group": example["coarse_group"],
        "predicted_intent": run.predicted_intent,
        "predicted_group": run.predicted_group_stated,
        "is_valid": run.is_valid,
        "is_hierarchy_consistent": run.is_hierarchy_consistent,
        "had_error": run.had_error,
        "prompt_tokens": run.total_prompt_tokens,
        "completion_tokens": run.total_completion_tokens,
        "latency_seconds": run.total_latency_seconds,
        "time_to_first_token_seconds": run.time_to_first_token_seconds,
        "context_payload_bytes": sum(call.context_payload_bytes for call in run.calls),
        "cache_hit": run.extra.get("cache_hit", False),
        "retrieval_empty": run.extra.get("retrieval_empty", False),
        "raw_response": run.raw_response,
        "prompts_used": [call.messages_sent for call in run.calls],
    }


def main() -> None:
    train_pool = load_examples("train")
    test_pool = load_examples("test")
    test_sample = stratified_sample(test_pool, TEST_SAMPLE_SIZE)
    resources = build_resources(train_pool)

    records = []
    for technique_key, technique in TECHNIQUES.items():
        print(f"{technique.NAME}: {len(test_sample)} queries")
        for i, example in enumerate(test_sample, start=1):
            run = technique.run(example["text"], resources)
            records.append(build_record(technique_key, example, run))
            if i % 50 == 0:
                print(f"  {i}/{len(test_sample)}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
