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
        "when_to_use": (
            "Reach for this when a PEFT technique's constrained update genuinely isn't "
            "enough: teaching the model a large amount of new domain knowledge, a very "
            "different output format, or a capability far from what it already does, "
            "and you have both the compute and a large enough dataset to support "
            "changing every parameter without overfitting."
        ),
        "deep_dive": [
            "Let's start with the simplest case: what if you just kept training the "
            "model the exact same way it learned everything in the first place? "
            "That's full fine-tuning. Nothing is off-limits. Every single number "
            "inside the model, all 8.03 billion of them, gets a chance to change a "
            "little with each example it sees. There's no decision to make about "
            "which parts to touch, because the answer is simply: all of them.",
            "That freedom comes at a real cost. To update a number during training, "
            "the computer has to remember which direction it should move and a bit "
            "of its recent history too (see Optimizer state, in Terminology). Do "
            "that for 8 billion numbers, and you need roughly 120GB of memory, far "
            "more than most people have on their own computer, plus a full new copy "
            "of the model for every task you fine-tune it for. This makes sense "
            "mainly when the cheaper techniques on this page genuinely aren't "
            "enough, for example when teaching the model a large amount of "
            "information it has never seen, and you have both the computer and the "
            "data to support it.",
        ],
        "diagram": "full_finetuning",
    }
