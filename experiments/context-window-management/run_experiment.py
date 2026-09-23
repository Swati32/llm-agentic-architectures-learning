"""Builds a length x position x needle grid of synthetic haystack
documents from SQuAD, runs all 4 context-management techniques over
every document, and writes one combined results/records.json for the
Streamlit app to read.

Run with `python3 -u run_experiment.py` so progress prints show up
immediately rather than buffering until the process exits.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from data import build_document, load_examples, sample_needles_and_haystack
from metrics import exact_match, token_f1
from techniques import TECHNIQUES

NEEDLE_COUNT = 8
LENGTHS = {"short": 500, "medium": 2000, "long": 6000}
POSITIONS = ["start", "middle", "end"]
RANDOM_SEED = 0
RESULTS_PATH = Path(__file__).parent / "results" / "records.json"


def build_record(technique_key: str, module, doc_id: str, needle: dict, length_name: str, position: str, run) -> dict:
    return {
        "technique": technique_key,
        "technique_name": module.NAME,
        "document_id": doc_id,
        "length": length_name,
        "position": position,
        "question": needle["question"],
        "gold_answer": needle["answer"],
        "predicted_answer": run.predicted_answer,
        "exact_match": exact_match(run.predicted_answer, needle["answer"]),
        "f1": token_f1(run.predicted_answer, needle["answer"]),
        "llm_calls": run.llm_calls,
        "total_prompt_tokens": run.total_prompt_tokens,
        "total_completion_tokens": run.total_completion_tokens,
        "total_latency_seconds": run.total_latency_seconds,
        "time_to_first_token_seconds": run.time_to_first_token_seconds,
        "mean_context_payload_bytes": run.mean_context_payload_bytes,
        "wall_clock_seconds": run.wall_clock_seconds,
        "had_error": run.had_error,
        "context_sent_to_generator": run.context_sent_to_generator,
        "steps": [asdict(step) for step in run.steps],
    }


def main() -> None:
    print("Loading SQuAD and sampling needles...")
    examples = load_examples("train")
    needles, distractor_pool = sample_needles_and_haystack(examples, NEEDLE_COUNT, seed=RANDOM_SEED)
    print(f"Sampled {len(needles)} needles, {len(distractor_pool)} distractor paragraphs")

    techniques_metadata = {
        key: {
            "name": module.NAME,
            "description": module.DESCRIPTION,
            "prompts": module.PROMPTS,
            "what_it_is": module.WHAT_IT_IS,
            "how_we_implemented_it": module.HOW_WE_IMPLEMENTED_IT,
            "when_its_useful": module.WHEN_ITS_USEFUL,
            "diagram": module.DIAGRAM,
        }
        for key, module in TECHNIQUES.items()
    }

    total_documents = len(needles) * len(LENGTHS) * len(POSITIONS)
    print(f"Building {total_documents} documents ({len(needles)} needles x {len(LENGTHS)} lengths x {len(POSITIONS)} positions)")

    records = []
    doc_index = 0
    for needle_i, needle in enumerate(needles):
        for length_name, length_tokens in LENGTHS.items():
            for position in POSITIONS:
                doc_index += 1
                doc_id = f"needle{needle_i}_{length_name}_{position}"
                document = build_document(
                    needle["context"], distractor_pool, length_tokens, position, seed=RANDOM_SEED + doc_index
                )

                for technique_key, module in TECHNIQUES.items():
                    run = module.run(document, needle["question"])
                    records.append(build_record(technique_key, module, doc_id, needle, length_name, position, run))

                if doc_index % 10 == 0:
                    print(f"  {doc_index}/{total_documents} documents done")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "needle_count": len(needles),
                "lengths": LENGTHS,
                "positions": POSITIONS,
                "techniques": techniques_metadata,
                "runs": records,
            },
            indent=2,
        )
    )
    print(f"\nWrote {len(records)} records to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
