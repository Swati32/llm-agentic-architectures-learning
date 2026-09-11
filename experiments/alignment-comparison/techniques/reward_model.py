"""Trains the reward model that RLHF's PPO stage optimizes against.

This is the step that makes RLHF different from DPO: instead of using the
chosen/rejected pairs to update the policy directly, they're first used to
train a separate scorer, a model whose only job is to take a (prompt,
response) pair and output a single number saying how good that response
is. PPO then never looks at the original pairs again, only at this
scorer's opinion of whatever the policy generates.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from common import MPSCacheClearingCallback, device, measure_training_run
from lora_config import sequence_classification_lora


def run(base_model_id, pref_dataset, output_dir, num_train_epochs=1, eval_dataset=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(base_model_id, num_labels=1).to(device())
    model.config.pad_token_id = tokenizer.pad_token_id

    config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        max_length=512,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=pref_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=sequence_classification_lora(),
        callbacks=[MPSCacheClearingCallback()],
    )

    with measure_training_run() as run_metrics:
        train_output = trainer.train()

    accuracy = None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        accuracy = eval_metrics.get("eval_accuracy")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "technique": "Reward model (for RLHF)",
        "train_loss": train_output.training_loss,
        "held_out_accuracy": accuracy,
        "num_examples": len(pref_dataset),
        **run_metrics,
    }
