"""
Architecture constants for llama3.1:8b, the same model used in the
intent-classification-prompting experiment. These are the published values
from Meta's Llama 3.1 model config (hidden size, layer count, GQA head
counts), not something loaded from a live model, since this experiment
computes parameter counts and memory footprints, not model outputs.

Reusing this model keeps the comparison grounded in something concrete
instead of an abstract "a large language model".
"""

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_LAYERS = 32
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8  # grouped-query attention
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 128
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 1024
VOCAB_SIZE = 128256

# Bytes per parameter for the frozen base weights, at standard fp16/bf16.
# QLoRA overrides this to 4-bit (0.5 bytes) for its own technique.
BASE_WEIGHT_BYTES = 2

# Standard mixed-precision Adam rule of thumb: a trainable parameter costs
# ~16 bytes total (2 bytes weight + 2 bytes gradient + 12 bytes Adam state,
# fp32 master weight + two fp32 moments). See "Terminology" in the README,
# and https://blog.eleuther.ai/transformer-math/ for the derivation.
TRAINABLE_PARAM_BYTES = 16


def attention_params_per_layer():
    q_proj = HIDDEN_SIZE * HIDDEN_SIZE
    k_proj = HIDDEN_SIZE * KV_DIM
    v_proj = HIDDEN_SIZE * KV_DIM
    o_proj = HIDDEN_SIZE * HIDDEN_SIZE
    return q_proj + k_proj + v_proj + o_proj


def mlp_params_per_layer():
    # SwiGLU MLP: gate, up, down projections
    gate = HIDDEN_SIZE * INTERMEDIATE_SIZE
    up = HIDDEN_SIZE * INTERMEDIATE_SIZE
    down = INTERMEDIATE_SIZE * HIDDEN_SIZE
    return gate + up + down


def norm_params_per_layer():
    return 2 * HIDDEN_SIZE  # input norm + post-attention norm, RMSNorm (weight only, no bias)


def params_per_layer():
    return attention_params_per_layer() + mlp_params_per_layer() + norm_params_per_layer()


def total_params():
    embedding = VOCAB_SIZE * HIDDEN_SIZE
    lm_head = VOCAB_SIZE * HIDDEN_SIZE  # untied in Llama 3.1
    final_norm = HIDDEN_SIZE
    return embedding + lm_head + final_norm + NUM_LAYERS * params_per_layer()


def memory_gb(trainable_params, frozen_params, base_weight_bytes=BASE_WEIGHT_BYTES):
    bytes_total = frozen_params * base_weight_bytes + trainable_params * TRAINABLE_PARAM_BYTES
    return bytes_total / (1024 ** 3)
