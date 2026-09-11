"""Orchestrates the full comparison: load data, train SFT, train RLHF's
reward model and PPO stage on top of it, train DPO on top of it too,
generate each technique's responses to the same held-out prompts, judge
them pairwise, and write results/records.json.

Trained adapters and merged checkpoints are written under checkpoints/,
which is gitignored: only the resulting metrics and generations in
results/records.json are committed, same as every other experiment in
this repo. Re-running this script retrains everything from scratch.
"""

import json
import sys
import time
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import merge_lora_and_save
from data import build_experiment_data
from generate import generate_responses
from judge import judge_pair
from metrics import overall_win_rates, pairwise_summary
from techniques import dpo, reward_model, rlhf_ppo, sft

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CHECKPOINTS_DIR = Path("checkpoints")
RESULTS_PATH = Path("results/records.json")

N_TRAIN_PAIRS = 600
N_EVAL_PROMPTS = 60
N_REWARD_EVAL_PAIRS = 80
SFT_EPOCHS = 3
REWARD_EPOCHS = 2
DPO_EPOCHS = 3
PPO_EPOCHS = 1  # PPO's rollout+generation cost dominates wall-clock; see README for why
MAX_NEW_TOKENS = 96

TECHNIQUES = ["SFT", "RLHF (PPO)", "DPO"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log("Loading and filtering Anthropic/hh-rlhf (helpful-base)...")
    sft_ds, pref_ds, eval_prompts, reward_eval_ds, dpo_eval_ds = build_experiment_data(
        tokenizer,
        n_train_pairs=N_TRAIN_PAIRS,
        n_eval_prompts=N_EVAL_PROMPTS,
        n_reward_eval_pairs=N_REWARD_EVAL_PAIRS,
    )
    log(f"SFT examples: {len(sft_ds)}, preference pairs: {len(pref_ds)}, eval prompts: {len(eval_prompts)}")

    training_metrics = {}

    log("Training SFT...")
    training_metrics["sft"] = sft.run(
        BASE_MODEL, sft_ds, str(CHECKPOINTS_DIR / "sft_adapter"), num_train_epochs=SFT_EPOCHS
    )
    log(f"SFT done: {training_metrics['sft']}")

    log("Merging SFT adapter into a full checkpoint (the shared starting point for RLHF and DPO)...")
    merge_lora_and_save(BASE_MODEL, str(CHECKPOINTS_DIR / "sft_adapter"), str(CHECKPOINTS_DIR / "sft_merged"))
    sft_merged = str(CHECKPOINTS_DIR / "sft_merged")

    log("Training the reward model...")
    training_metrics["reward_model"] = reward_model.run(
        sft_merged,
        pref_ds,
        str(CHECKPOINTS_DIR / "reward_adapter"),
        num_train_epochs=REWARD_EPOCHS,
        eval_dataset=reward_eval_ds,
    )
    log(f"Reward model done: {training_metrics['reward_model']}")

    log("Merging reward model adapter...")
    merge_lora_and_save(
        sft_merged,
        str(CHECKPOINTS_DIR / "reward_adapter"),
        str(CHECKPOINTS_DIR / "reward_merged"),
        model_class=AutoModelForSequenceClassification,
        num_labels=1,
    )

    log("Training RLHF's PPO stage...")
    training_metrics["rlhf_ppo"] = rlhf_ppo.run(
        sft_merged,
        str(CHECKPOINTS_DIR / "reward_merged"),
        list(pref_ds["prompt"]),
        str(CHECKPOINTS_DIR / "ppo_adapter"),
        num_train_epochs=PPO_EPOCHS,
    )
    log(f"RLHF (PPO) done: {training_metrics['rlhf_ppo']}")

    log("Training DPO...")
    training_metrics["dpo"] = dpo.run(
        sft_merged,
        pref_ds,
        str(CHECKPOINTS_DIR / "dpo_adapter"),
        num_train_epochs=DPO_EPOCHS,
        eval_dataset=dpo_eval_ds,
    )
    log(f"DPO done: {training_metrics['dpo']}")

    log("Generating held-out responses for all three techniques...")
    generations = {
        "SFT": generate_responses(sft_merged, None, eval_prompts, max_new_tokens=MAX_NEW_TOKENS),
        "RLHF (PPO)": generate_responses(
            sft_merged, str(CHECKPOINTS_DIR / "ppo_adapter"), eval_prompts, max_new_tokens=MAX_NEW_TOKENS
        ),
        "DPO": generate_responses(
            sft_merged, str(CHECKPOINTS_DIR / "dpo_adapter"), eval_prompts, max_new_tokens=MAX_NEW_TOKENS
        ),
    }
    log("Generation done.")

    log("Judging every technique pair on every held-out prompt...")
    technique_pairs = [
        ("SFT", "RLHF (PPO)"),
        ("SFT", "DPO"),
        ("RLHF (PPO)", "DPO"),
    ]
    judgments = []
    for i, prompt in enumerate(eval_prompts):
        for technique_a, technique_b in technique_pairs:
            response_a = generations[technique_a][i]["response"]
            response_b = generations[technique_b][i]["response"]
            verdict, call_result = judge_pair(prompt, response_a, response_b, seed=i)
            winner = {"left": "a", "right": "b", "tie": "tie"}.get(verdict)
            judgments.append(
                {
                    "prompt": prompt,
                    "technique_a": technique_a,
                    "technique_b": technique_b,
                    "response_a": response_a,
                    "response_b": response_b,
                    "winner": winner,
                    "judge_error": None if call_result.succeeded else call_result.error,
                }
            )
        if (i + 1) % 10 == 0:
            log(f"  judged {i + 1}/{len(eval_prompts)} prompts")

    valid_judgments = [j for j in judgments if j["winner"] is not None]
    log(f"Judging done: {len(valid_judgments)}/{len(judgments)} calls produced a usable verdict.")

    output = {
        "base_model": BASE_MODEL,
        "dataset": "Anthropic/hh-rlhf (helpful-base subset, single-turn)",
        "n_train_pairs": len(pref_ds),
        "n_eval_prompts": len(eval_prompts),
        "training_metrics": training_metrics,
        "generations": generations,
        "judgments": judgments,
        "pairwise_summary": pairwise_summary(valid_judgments),
        "overall_win_rates": overall_win_rates(valid_judgments, TECHNIQUES),
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
