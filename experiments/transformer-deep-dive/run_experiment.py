"""Computes the worked attention example from Attention Is All You Need
(https://arxiv.org/abs/1706.03762) and writes it to results/records.json.

There is no model here and nothing to train. The sentence "cats chase mice"
gets small, hand-picked embeddings and hand-picked Q/K/V projection weights,
chosen only to keep every number small enough to read in a table, not
learned from data. The arithmetic that turns those numbers into an
attention output is exactly the mechanism the paper describes: scaled
dot-product attention (Section 3.2.1) run once for a single head, then
again split across two heads to show multi-head attention (Section 3.2.2).
Everything downstream (the app, the README's numbers) reads this file
rather than recomputing it, the same pattern every other experiment in
this repo follows.
"""

import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

TOKENS = ["cats", "chase", "mice"]

# d_model = 4. Hand-picked one-hot-ish embeddings, not learned.
X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # cats
    [0.0, 1.0, 0.0, 1.0],   # chase
    [1.0, 1.0, 0.0, 0.0],   # mice
])

# Hand-picked projection weights, d_model=4 -> d_k=d_v=4 (single-head pass).
W_Q = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.3, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
])
W_K = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.6, 0.0, 0.0, 1.0],
])
W_V = np.array([
    [0.5, 0.0, 0.0, 1.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.5],
])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> dict:
    d_k = q.shape[-1]
    raw_scores = q @ k.T
    scaled_scores = raw_scores / np.sqrt(d_k)
    weights = softmax(scaled_scores)
    output = weights @ v
    return {
        "raw_scores": raw_scores.tolist(),
        "scaled_scores": scaled_scores.tolist(),
        "weights": weights.tolist(),
        "output": output.tolist(),
    }


def single_head_pass() -> dict:
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    attn = scaled_dot_product_attention(Q, K, V)
    return {
        "d_model": 4,
        "d_k": 4,
        "tokens": TOKENS,
        "embeddings": X.tolist(),
        "W_Q": W_Q.tolist(),
        "W_K": W_K.tolist(),
        "W_V": W_V.tolist(),
        "Q": Q.tolist(),
        "K": K.tolist(),
        "V": V.tolist(),
        **attn,
    }


def multi_head_pass(n_heads: int = 2) -> dict:
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    d_head = Q.shape[1] // n_heads
    heads = []
    for h in range(n_heads):
        sl = slice(h * d_head, (h + 1) * d_head)
        attn = scaled_dot_product_attention(Q[:, sl], K[:, sl], V[:, sl])
        heads.append({"head": h, "d_head": d_head, **attn})
    concat = np.concatenate([np.array(h["output"]) for h in heads], axis=1)
    W_O = np.eye(concat.shape[1])  # identity, so concat and final output match for readability
    final_output = concat @ W_O
    return {
        "n_heads": n_heads,
        "d_head": d_head,
        "heads": heads,
        "concat": concat.tolist(),
        "W_O": W_O.tolist(),
        "output": final_output.tolist(),
    }


def main() -> None:
    records = {
        "paper": "Attention Is All You Need (Vaswani et al., 2017)",
        "sentence": " ".join(TOKENS),
        "single_head": single_head_pass(),
        "multi_head": multi_head_pass(n_heads=2),
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
