import model_config as mc


def compute():
    total = mc.total_params()
    trainable = total
    frozen = 0
    return {
        "technique": "Full fine-tuning",
        "family": "Baseline",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100.0,
        "memory_gb": mc.memory_gb(trainable, frozen),
        "config": {},
        "mechanism": (
            "Every parameter in the model is trainable. The optimizer keeps a gradient "
            "and Adam state for all 8.03B of them."
        ),
        "why": (
            "This was the only option before PEFT existed: to adapt a pretrained model "
            "to a new task, keep training it, letting every parameter move."
        ),
        "tradeoffs": (
            "Gives the model the most freedom to change, at the highest possible cost: "
            "roughly 120GB of memory to train, and a full new copy of the model to "
            "store per task, since nothing is shared with the original."
        ),
        "diagram": "full_finetuning",
    }
