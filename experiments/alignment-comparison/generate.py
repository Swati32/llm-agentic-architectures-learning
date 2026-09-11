"""Generates responses from a fine-tuned technique's adapter, for the
held-out eval prompts. Used the same way for all three techniques' final
checkpoints, so the win-rate judging in judge.py is comparing responses
produced under identical decoding settings, not an artifact of one
technique being sampled differently than another.
"""

import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import device


def generate_responses(sft_merged_path, adapter_dir, prompts, max_new_tokens=96, batch_size=8):
    """adapter_dir=None generates from sft_merged_path directly (used for
    SFT itself, which has nothing further to layer on top of); otherwise
    loads the given technique's LoRA adapter over that same base, so every
    technique is generated from an identical starting point."""
    tokenizer = AutoTokenizer.from_pretrained(sft_merged_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(sft_merged_path).to(device())
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base_model, adapter_dir).to(device())
    else:
        model = base_model
    model.eval()

    formatted = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]

    responses, latencies = [], []
    for start in range(0, len(formatted), batch_size):
        batch = formatted[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device())
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start_time
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        responses.extend(text.strip() for text in decoded)
        latencies.extend([elapsed / len(batch)] * len(batch))

    del model, base_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return [
        {"prompt": prompt, "response": response, "latency_seconds": latency}
        for prompt, response, latency in zip(prompts, responses, latencies)
    ]
