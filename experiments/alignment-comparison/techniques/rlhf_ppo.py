"""RLHF's second stage: PPO optimizes the SFT policy against the trained
reward model.

Unlike SFT, the policy is never shown a "correct" response to copy here.
Instead, it generates its own response to a prompt, the reward model
scores that response, and PPO nudges the policy's weights toward whatever
tended to score higher, while a KL penalty keeps it from drifting too far
from the SFT policy in the process (see kl_mean in the returned metrics).
train_reward_model.py's reward model has to be trained first; this
function only runs the PPO loop on top of it.
"""

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from common import MPSCacheClearingCallback, device, measure_training_run
from lora_config import causal_lm_lora


def _tokenize_prompts(tokenizer, prompts, max_prompt_length=256):
    tokenized = tokenizer(list(prompts), truncation=True, max_length=max_prompt_length)
    return Dataset.from_dict({"input_ids": tokenized["input_ids"]})


def run(
    sft_merged_path,
    reward_model_merged_path,
    formatted_prompts,
    output_dir,
    num_train_epochs=1,
    response_length=64,
):
    """formatted_prompts must already carry the chat template (e.g.
    pref_dataset["prompt"] from data.py), the same formatting SFT and DPO
    trained on. Tokenizing a raw, un-templated prompt here would show the
    policy an input shape it never learned to expect."""
    tokenizer = AutoTokenizer.from_pretrained(sft_merged_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation during rollouts

    prompt_dataset = _tokenize_prompts(tokenizer, formatted_prompts)

    policy_model = AutoModelForCausalLM.from_pretrained(sft_merged_path).to(device())
    value_model = AutoModelForSequenceClassification.from_pretrained(
        sft_merged_path, num_labels=1
    ).to(device())
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_merged_path, num_labels=1
    ).to(device())
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    value_model.config.pad_token_id = tokenizer.pad_token_id
    for param in reward_model.parameters():
        param.requires_grad_(False)

    config = PPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_mini_batches=1,
        num_ppo_epochs=2,
        learning_rate=1e-5,
        response_length=response_length,
        kl_coef=0.05,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        missing_eos_penalty=1.0,
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=prompt_dataset,
        eval_dataset=prompt_dataset,  # PPOTrainer samples a few completions from this after training finishes
        peft_config=causal_lm_lora(),
        callbacks=[MPSCacheClearingCallback()],
    )

    with measure_training_run() as run_metrics:
        trainer.train()

    history = trainer.state.log_history
    mean_reward_logs = [row.get("objective/rlhf_reward") for row in history if "objective/rlhf_reward" in row]
    kl_logs = [row.get("objective/kl") for row in history if "objective/kl" in row]

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "technique": "RLHF (PPO)",
        "num_examples": len(formatted_prompts),
        "mean_rlhf_reward": mean_reward_logs[-1] if mean_reward_logs else None,
        "mean_kl_from_reference": kl_logs[-1] if kl_logs else None,
        **run_metrics,
    }
