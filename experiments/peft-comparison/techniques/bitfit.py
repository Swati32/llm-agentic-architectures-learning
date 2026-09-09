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
        "when_to_use": (
            "Worth trying first, before anything heavier, but only on an architecture "
            "that actually has bias terms to unfreeze, typically BERT-style encoders "
            "rather than modern bias-free decoder models like Llama. Check this before "
            "reaching for it: on the wrong architecture it silently trains nothing."
        ),
        "deep_dive": [
            "This is the one technique here that actually selects from parameters "
            "that already exist, and it does it with the simplest possible rule: "
            "freeze everything, then unfreeze every parameter whose role is a bias "
            "term, identified by name and position in the architecture, not by any "
            "importance score. On `llama3.1:8b` that rule fires on nothing: Llama's "
            "linear layers have no bias terms, and its RMSNorm layers have only a "
            "weight, no bias, so there is nothing for BitFit to unfreeze on this "
            "architecture.",
            "[Zaken et al., 2021](https://arxiv.org/abs/2106.10199) asked how "
            "little of a model you could train and still adapt it, and found that "
            "just its existing bias terms, already tiny, already there, were often "
            "enough on the BERT-style models they tested. It's extremely cheap when "
            "it applies, since nothing new is added and almost nothing is "
            "unfrozen. It's worth trying first, before anything heavier, but only "
            "on an architecture that actually has bias terms to unfreeze; check "
            "this before reaching for it, since on the wrong architecture, like "
            "this one, it silently trains nothing.",
        ],
        "diagram": "bitfit",
    }
