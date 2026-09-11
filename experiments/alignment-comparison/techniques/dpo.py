"""DPO: skip the reward model and the RL loop entirely, and optimize the
SFT policy directly on the same chosen/rejected pairs RLHF's reward model
was trained on.

DPO's loss ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))
pushes the policy to make a chosen response more likely than a rejected
one, relative to a frozen reference copy of the same SFT policy, in one
supervised-style training pass. There's no generation step during
training and nothing being sampled from the policy, which is the whole
reason it needs no reward model and no RL loop: it never has to score a
response the policy hasn't been directly shown.
"""

from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import MPSCacheClearingCallback, measure_training_run
from lora_config import causal_lm_lora


def run(sft_merged_path, pref_dataset, output_dir, num_train_epochs=1, eval_dataset=None):
    tokenizer = AutoTokenizer.from_pretrained(sft_merged_path)

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=sft_merged_path,
        ref_model=None,  # peft_config below means the reference is the same weights with the adapter disabled
        args=config,
        train_dataset=pref_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=causal_lm_lora(),
        callbacks=[MPSCacheClearingCallback()],
    )

    with measure_training_run() as run_metrics:
        train_output = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # DPOTrainer has no built-in reference-KL metric the way PPOTrainer does.
    # The closest signals for "how much did the policy move, and in the right
    # direction" are its implicit reward margin (how much more likely the
    # chosen response is made than the rejected one) and accuracy (how often
    # that ranking comes out correct), measured on held-out pairs so this
    # isn't just reporting how well it fit its own training batch.
    margin = accuracy = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        margin = eval_metrics.get("eval_rewards/margins")
        accuracy = eval_metrics.get("eval_rewards/accuracies")

    return {
        "technique": "DPO",
        "train_loss": train_output.training_loss,
        "held_out_reward_margin": margin,
        "held_out_reward_accuracy": accuracy,
        "num_examples": len(pref_dataset),
        **run_metrics,
    }
