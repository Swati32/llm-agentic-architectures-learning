"""
No live model calls and no dataset loading here, unlike the other
experiments in this repo. This experiment compares parameter-efficient
fine-tuning methods by their mechanics, not by running them, so
"running the experiment" means computing real trainable-parameter counts
and memory footprints from llama3.1:8b's architecture (see
model_config.py), for the scenario of fine-tuning it to summarize SAMSum
dialogues. The dashboard reads the resulting results/records.json, same as
every other experiment.
"""

import json

import model_config as mc
from techniques import adapters, bitfit, full_finetuning, lora, prefix_tuning, qlora


def main():
    records = [
        full_finetuning.compute(),
        lora.compute(rank=8),
        qlora.compute(rank=8),
        adapters.compute(),
        prefix_tuning.compute(),
        bitfit.compute(),
    ]

    output = {
        "reference_model": "llama3.1:8b",
        "total_params": mc.total_params(),
        "scenario": "Fine-tuning llama3.1:8b to summarize SAMSum dialogues",
        "records": records,
        # How trainable-parameter count scales with rank, holding the technique
        # fixed. Shown as its own chart rather than extra rows in the family table.
        "lora_rank_sweep": [lora.compute(rank=r) for r in (4, 8, 16, 32, 64, 128)],
    }

    with open("results/records.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Reference model: llama3.1:8b, {mc.total_params():,} params")
    for r in records:
        print(
            f"{r['technique']:<20} trainable={r['trainable_params']:>12,} "
            f"({r['trainable_pct']:.4f}%)  memory={r['memory_gb']:.2f} GB"
        )


if __name__ == "__main__":
    main()
