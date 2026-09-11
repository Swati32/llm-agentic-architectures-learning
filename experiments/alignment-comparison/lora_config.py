"""One LoRA configuration, shared by all three techniques.

This experiment isn't about how many parameters move (see
[peft-comparison](../peft-comparison/README.md) for that question, worked
out on this same q_proj/v_proj target choice). It's about what each
training objective does with that same, fixed-size trainable slice. Using
one LoRA config everywhere keeps that controlled: SFT, the reward model,
RLHF's PPO policy, and DPO all train exactly the same shape of update, so
any difference between them comes from the objective, not from one
technique quietly getting a bigger budget than another.
"""

from peft import LoraConfig

RANK = 8
TARGET_MODULES = ["q_proj", "v_proj"]


def causal_lm_lora():
    return LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def sequence_classification_lora():
    return LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
