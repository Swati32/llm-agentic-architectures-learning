import model_config as mc

NUM_VIRTUAL_TOKENS = 20


def trainable_params_for_tokens(num_virtual_tokens):
    # A learned key vector and value vector, per virtual token, per layer.
    return num_virtual_tokens * mc.NUM_LAYERS * 2 * mc.HIDDEN_SIZE


def compute(num_virtual_tokens=NUM_VIRTUAL_TOKENS):
    total = mc.total_params()
    trainable = trainable_params_for_tokens(num_virtual_tokens)
    frozen = total - trainable
    return {
        "technique": "Prefix-tuning",
        "family": "Soft prompt",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / total,
        "memory_gb": mc.memory_gb(trainable, frozen),
        "config": {"num_virtual_tokens": num_virtual_tokens},
        "mechanism": (
            f"Base weights are frozen and untouched. {num_virtual_tokens} trainable "
            "\"virtual token\" key/value vectors are learned for every layer and "
            "prepended to that layer's real keys and values during attention, "
            "steering what the frozen model attends to without changing any of its "
            "weights. Not captured by the parameter count: these virtual tokens also "
            f"consume {num_virtual_tokens} positions of the model's context window on "
            "every forward pass, a cost this table doesn't show."
        ),
        "why": (
            "Li & Liang, 2021, built this for generation tasks, on the idea that "
            "steering a frozen model's attention with a learned prompt can work "
            "almost as well as changing its weights, while touching no weights at all."
        ),
        "tradeoffs": (
            "No weights are ever touched, which makes it easy to swap between many "
            "tasks on the same frozen model. The real cost is context window space, "
            "not parameters: every virtual token eats a position that a real token "
            "could otherwise use, on every single call."
        ),
        "when_to_use": (
            "Reach for this when you need to serve many tasks off the exact same "
            "frozen weights with nothing to merge or swap at all, and your prompts "
            "are short enough that losing a few tens of context positions to virtual "
            "tokens doesn't matter. Less attractive for long-context use cases, where "
            "that lost space is worth more."
        ),
        "diagram": "prefix_tuning",
    }
