import model_config as mc


def compute():
    total = mc.total_params()
    # Llama-family models have no bias terms: linear projections are
    # bias-free, and RMSNorm (unlike LayerNorm) has only a weight/gain
    # parameter, no bias. BitFit was designed against BERT-style
    # architectures, where every linear layer and every LayerNorm has a
    # bias term to unfreeze. On this architecture there is nothing for it
    # to select. See Terminology and "What we learned" for why.
    trainable = 0
    frozen = total
    return {
        "technique": "BitFit",
        "family": "Selective",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 0.0,
        "memory_gb": mc.memory_gb(trainable, frozen),
        "config": {},
        "mechanism": (
            "Freeze every weight except the model's existing bias terms, and train "
            "only those. Nothing new is added. On llama3.1:8b this selects zero "
            "parameters: Llama's linear layers have no bias terms, and its RMSNorm "
            "layers have only a weight, no bias, so there is nothing for BitFit to "
            "unfreeze on this architecture."
        ),
        "why": (
            "Zaken et al., 2021, asked how little of a model you could train and "
            "still adapt it to a new task, and found that just its existing bias "
            "terms, already tiny, already there, were often enough."
        ),
        "tradeoffs": (
            "Extremely cheap when it applies, since nothing new is added and almost "
            "nothing is unfrozen. Whether it applies at all depends entirely on the "
            "base model's design: bias-free architectures like Llama give it nothing "
            "to work with, which is not a flaw in the technique, just a mismatch with "
            "this particular model family."
        ),
        "diagram": "bitfit",
    }
