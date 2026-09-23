"""Thin wrapper around a local Ollama model that turns every call into a
`CallResult` carrying both the model's answer and the operational metrics
this project tracks for it.

Runs against Ollama (http://localhost:11434) rather than a hosted
provider, same reasoning as the other experiments in this repo: it's
free, and this experiment runs hundreds of calls across a length x
position x technique grid, so cost matters more here, not less. Start the
server with `ollama serve` and pull the model with `ollama pull
llama3.1:8b` first.
"""

import json
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_ID = "llama3.1:8b"
NANOSECONDS_PER_SECOND = 1_000_000_000

# Ollama's `/api/tags` reports llama3.1:8b's *maximum supported* context as
# 131,072 tokens, but the server process it actually launches to serve a
# request defaults to a much smaller runtime context window (4,096 tokens)
# unless told otherwise. This experiment's whole point is testing what
# happens at document lengths well past that default (a "long" document
# here runs ~6,000 words, roughly 8,000+ tokens), so leaving this unset
# would silently truncate/context-shift exactly the documents this
# experiment exists to measure, the "don't silently truncate" pitfall
# this repo explicitly calls out. Set well above the longest case with
# margin for the system prompt, question, and (for Hierarchical
# Summarization) the accumulated running summary.
NUM_CTX = 16384


@dataclass
class CallResult:
    messages_sent: list[dict]
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    context_payload_bytes: int
    time_to_first_token_seconds: float | None
    latency_seconds: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def call_model(messages: list[dict], temperature: float = 0.0) -> CallResult:
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }
    context_payload_bytes = len(json.dumps(messages).encode("utf-8"))

    try:
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - any local server/network failure counts as a call error
        return CallResult(
            messages_sent=messages,
            content=None,
            prompt_tokens=0,
            completion_tokens=0,
            context_payload_bytes=context_payload_bytes,
            time_to_first_token_seconds=None,
            latency_seconds=0.0,
            error=str(exc),
        )

    # Ollama reports prefill (prompt_eval_duration) and generation
    # (eval_duration) separately, server-side — more accurate than timing
    # the HTTP round trip ourselves, and free of the network/parsing
    # overhead that measurement adds. Because Ollama reuses its KV cache
    # across calls that share a long system prompt, prompt_eval_duration
    # (used here as time-to-first-token) will look artificially small
    # relative to total latency once the cache is warm — a property of
    # local single-user serving, not of any technique. See the README.
    return CallResult(
        messages_sent=messages,
        content=result["message"]["content"],
        prompt_tokens=result.get("prompt_eval_count", 0),
        completion_tokens=result.get("eval_count", 0),
        context_payload_bytes=context_payload_bytes,
        time_to_first_token_seconds=result.get("prompt_eval_duration", 0) / NANOSECONDS_PER_SECOND,
        latency_seconds=result.get("total_duration", 0) / NANOSECONDS_PER_SECOND,
    )
