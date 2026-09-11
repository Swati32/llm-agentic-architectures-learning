"""SFT: train the model to imitate the chosen responses directly.

No preference signal is used here at all, only the `chosen` half of each
pair, treated as "here is the correct response to this prompt." This is
the starting point RLHF and DPO both build on: neither technique in this
experiment ever touches a base model directly, both continue training
from this SFT checkpoint.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from common import MPSCacheClearingCallback, device, measure_training_run
from lora_config import causal_lm_lora


def run(base_model_id, sft_dataset, output_dir, num_train_epochs=1):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device())

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        dataset_text_field="text",
        max_length=512,
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        peft_config=causal_lm_lora(),
        callbacks=[MPSCacheClearingCallback()],
    )

    with measure_training_run() as run_metrics:
        train_output = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "technique": "SFT",
        "train_loss": train_output.training_loss,
        "num_examples": len(sft_dataset),
        **run_metrics,
    }
