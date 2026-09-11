"""Loads Anthropic/hh-rlhf's helpful-base subset and turns its raw
"\n\nHuman: ...\n\nAssistant: ..." transcripts into the three shapes each
technique needs: SFT demonstrations, preference pairs, and a held-out
prompt set for evaluation generations.

Restricted to single-turn conversations only (one Human turn, one
Assistant turn). Multi-turn transcripts in this dataset can run long, and
keeping prompts short is what makes training tractable on a laptop.
Restricted to helpful-base rather than the full hh-rlhf mix: the other
subsets (harmless-base, red-team-attempts) contain adversarial and
explicit content, which doesn't belong in a public dashboard's example
outputs, and isn't needed to compare how SFT, RLHF, and DPO work as
training mechanisms.
"""

import json
import re
from pathlib import Path

from datasets import Dataset, load_dataset

TURN_SPLIT = re.compile(r"\n\n(Human|Assistant): ")
MAX_PROMPT_CHARS = 400
MAX_RESPONSE_CHARS = 600
CACHE_DIR = Path(__file__).parent / ".cache"


def _parse_turns(transcript):
    parts = TURN_SPLIT.split(transcript)
    # parts[0] is "" (text before the first turn marker); after that,
    # role/content pairs alternate.
    turns = []
    for i in range(1, len(parts), 2):
        role = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append((role, content))
    return turns


def _extract_single_turn_pair(example):
    chosen_turns = _parse_turns(example["chosen"])
    rejected_turns = _parse_turns(example["rejected"])
    if len(chosen_turns) != 2 or len(rejected_turns) != 2:
        return None
    if chosen_turns[0] != rejected_turns[0]:
        return None  # the two transcripts should share the same Human turn
    (human_role, prompt), (assistant_role, chosen_response) = chosen_turns
    _, rejected_response = rejected_turns[1]
    if human_role != "Human" or assistant_role != "Assistant":
        return None
    if not (0 < len(prompt) <= MAX_PROMPT_CHARS):
        return None
    if not (0 < len(chosen_response) <= MAX_RESPONSE_CHARS):
        return None
    if not (0 < len(rejected_response) <= MAX_RESPONSE_CHARS):
        return None
    if chosen_response == rejected_response:
        return None
    return {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response}


def load_filtered_pairs(split, limit=None):
    # Filtering hh-rlhf's ~44k-row train split down to single-turn pairs is
    # the slow part of this pipeline (a full Python-level regex pass), and
    # every technique in run_experiment.py needs the same filtered set, so
    # it's cached to disk after the first scan rather than re-run each time.
    cache_path = CACHE_DIR / f"{split}_pairs.jsonl"
    if cache_path.exists():
        with open(cache_path) as f:
            pairs = [json.loads(line) for line in f]
    else:
        raw = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
        pairs = []
        for example in raw:
            pair = _extract_single_turn_pair(example)
            if pair is not None:
                pairs.append(pair)
        CACHE_DIR.mkdir(exist_ok=True)
        with open(cache_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
    return pairs[:limit] if limit is not None else pairs


def build_experiment_data(tokenizer, n_train_pairs=800, n_eval_prompts=80, n_reward_eval_pairs=100, seed=42):
    """Returns (sft_dataset, pref_dataset, eval_prompts, reward_eval_dataset, dpo_eval_dataset).

    sft_dataset and pref_dataset are built from hh-rlhf's train split, and
    are what SFT, the reward model, RLHF's PPO stage, and DPO all train on.

    eval_prompts, reward_eval_dataset, and dpo_eval_dataset all come from
    hh-rlhf's test split, so nothing used for evaluation was ever seen
    during training. eval_prompts is prompt-only, used to generate fresh
    responses for the win-rate judging. reward_eval_dataset and
    dpo_eval_dataset are the same held-out pairs in each trainer's own
    expected shape (RewardTrainer wants full chosen/rejected text,
    DPOTrainer wants prompt separated out), used to check how well the
    reward model and DPO each learned the preference signal.
    """
    train_pairs = load_filtered_pairs("train", limit=n_train_pairs * 3)
    rng_pairs = list(train_pairs)
    import random

    random.Random(seed).shuffle(rng_pairs)
    train_pairs = rng_pairs[:n_train_pairs]

    def format_prompt(prompt):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    sft_records = [
        {"text": format_prompt(p["prompt"]) + p["chosen"] + tokenizer.eos_token} for p in train_pairs
    ]
    pref_records = [
        {
            "prompt": format_prompt(p["prompt"]),
            "chosen": p["chosen"] + tokenizer.eos_token,
            "rejected": p["rejected"] + tokenizer.eos_token,
        }
        for p in train_pairs
    ]

    eval_pairs = load_filtered_pairs("test", limit=(n_eval_prompts + n_reward_eval_pairs) * 2)
    random.Random(seed + 1).shuffle(eval_pairs)
    eval_prompts = [p["prompt"] for p in eval_pairs[:n_eval_prompts]]
    reward_eval_pairs = eval_pairs[n_eval_prompts : n_eval_prompts + n_reward_eval_pairs]
    reward_eval_records = [
        {
            "chosen": format_prompt(p["prompt"]) + p["chosen"] + tokenizer.eos_token,
            "rejected": format_prompt(p["prompt"]) + p["rejected"] + tokenizer.eos_token,
        }
        for p in reward_eval_pairs
    ]
    dpo_eval_records = [
        {
            "prompt": format_prompt(p["prompt"]),
            "chosen": p["chosen"] + tokenizer.eos_token,
            "rejected": p["rejected"] + tokenizer.eos_token,
        }
        for p in reward_eval_pairs
    ]

    return (
        Dataset.from_list(sft_records),
        Dataset.from_list(pref_records),
        eval_prompts,
        Dataset.from_list(reward_eval_records),
        Dataset.from_list(dpo_eval_records),
    )
