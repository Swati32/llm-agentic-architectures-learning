"""Thin wrapper around a local Ollama model that turns every call into a
`CallResult` carrying both the model's answer and the operational metrics
this project tracks for it. Same pattern as the other experiments in this
repo (see intent-classification-prompting/llm_client.py); here it's used
only for judging, not for the fine-tuning itself, which runs through
transformers/trl/peft directly (see techniques/).

Runs against Ollama (http://localhost:11434) rather than a hosted
provider, keeping the pipeline fully free and reproducible on any machine
with Ollama installed, at the cost of running on local compute instead of
dedicated inference hardware. Start the server with `ollama serve` and
pull the model with `ollama pull llama3.1:8b` before running this
experiment's judging step.
"""

import json
import time
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_ID = "llama3.1:8b"
NANOSECONDS_PER_SECOND = 1_000_000_000


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
        "options": {"temperature": temperature},
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

    return CallResult(
        messages_sent=messages,
        content=result["message"]["content"],
        prompt_tokens=result.get("prompt_eval_count", 0),
        completion_tokens=result.get("eval_count", 0),
        context_payload_bytes=context_payload_bytes,
        time_to_first_token_seconds=result.get("prompt_eval_duration", 0) / NANOSECONDS_PER_SECOND,
        latency_seconds=result.get("total_duration", 0) / NANOSECONDS_PER_SECOND,
    )
