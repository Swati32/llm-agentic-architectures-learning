"""Shared helpers used by every technique module: which device to train
on, and a way to measure wall-clock time and peak memory around a training
run. Kept in one place so the metrics in results/records.json are measured
the same way for SFT, RLHF, and DPO.
"""

import threading
import time
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

CHECKPOINTS_DIR_NAME = "checkpoints"


class MPSCacheClearingCallback(TrainerCallback):
    """Calls torch.mps.empty_cache() every few steps.

    Without this, a real run here (hundreds of steps, batches padded to
    whatever the longest sequence in that batch happens to be) grew to
    8GB+ RSS and started swapping heavily on a 16GB Mac, slowing from
    roughly 1 step/second to over 10 seconds/step by step 18. PyTorch's
    MPS allocator caches freed blocks for reuse rather than returning them
    to the OS, and variable batch shapes here mean many differently-sized
    blocks accumulate rather than getting reused. Passed to every trainer
    in this experiment via its `callbacks` argument.
    """

    def __init__(self, every_n_steps=5):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % self.every_n_steps == 0:
            torch.mps.empty_cache()


def merge_lora_and_save(base_model_id, adapter_dir, output_dir, model_class=AutoModelForCausalLM, **model_kwargs):
    """Folds a trained LoRA adapter into its base model's weights and saves
    the result as an ordinary full checkpoint.

    Used once, right after SFT: RLHF's reward model and PPO stage, and DPO,
    all start from the SFT policy, per the standard RLHF recipe (the reward
    model is initialized from the SFT model too, not the raw base model).
    Each of those then trains its own fresh LoRA adapter on top of this
    merged checkpoint, rather than stacking a second adapter on the first,
    so every technique's *own* trainable slice stays the same size and
    shape, set once in lora_config.py.

    Also used to fold RLHF's reward model adapter into a plain scorer
    (model_class=AutoModelForSequenceClassification), since PPO's reward
    scoring is simplest against an ordinary forward pass, not a PEFT-wrapped
    one.
    """
    base_model = model_class.from_pretrained(base_model_id, **model_kwargs)
    merged = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(output_dir)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _PeakMemorySampler:
    """Polls MPS's allocated-memory counter on a background thread.

    torch.mps has no built-in max_memory_allocated the way torch.cuda does,
    so this samples current_allocated_memory every 200ms while training
    runs and keeps the largest value seen. That means a very short spike
    between samples can be missed; good enough for comparing techniques
    against each other, not a precise peak.
    """

    def __init__(self, interval_seconds=0.2):
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            if torch.backends.mps.is_available():
                self.peak_bytes = max(self.peak_bytes, torch.mps.current_allocated_memory())
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()


@contextmanager
def measure_training_run():
    """Yields a dict that's filled in with wall_clock_seconds and
    peak_memory_mb once the `with` block exits."""
    result = {}
    start = time.time()
    with _PeakMemorySampler() as sampler:
        yield result
    result["wall_clock_seconds"] = time.time() - start
    result["peak_memory_mb"] = sampler.peak_bytes / (1024 ** 2)
