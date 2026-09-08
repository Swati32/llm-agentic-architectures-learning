"""The shape every technique's output is normalized into, so metrics.py and
the Streamlit app can treat all seven techniques identically."""

from dataclasses import dataclass, field

from llm_client import CallResult
from prompt_common import parse_json_response
from taxonomy import group_for_intent, resolve_group, resolve_intent


@dataclass
class TechniqueRun:
    predicted_intent: str | None
    predicted_group_stated: str | None
    raw_response: str | None
    calls: list[CallResult]
    extra: dict = field(default_factory=dict)  # technique-specific flags: cache_hit, retrieval_empty, votes, ...

    @property
    def is_valid(self) -> bool:
        """True when the response named an intent that actually exists,
        independent of whether that intent is the correct one."""
        return self.predicted_intent is not None

    @property
    def is_hierarchy_consistent(self) -> bool:
        """True when the model's stated group actually contains its stated
        intent — catches self-contradictory answers regardless of correctness."""
        if self.predicted_intent is None or self.predicted_group_stated is None:
            return False
        return group_for_intent(self.predicted_intent) == self.predicted_group_stated

    @property
    def total_prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(call.completion_tokens for call in self.calls)

    @property
    def total_latency_seconds(self) -> float:
        return sum(call.latency_seconds for call in self.calls)

    @property
    def time_to_first_token_seconds(self) -> float | None:
        """TTFT of the first call — later calls (self-consistency, hierarchical's
        second stage) are follow-on work, not part of the user's initial wait."""
        return self.calls[0].time_to_first_token_seconds if self.calls else None

    @property
    def had_error(self) -> bool:
        return any(not call.succeeded for call in self.calls)


def resolve_from_model_output(raw_response: str | None, calls: list[CallResult], extra: dict | None = None) -> TechniqueRun:
    parsed = parse_json_response(raw_response)
    predicted_intent = resolve_intent(parsed["intent"]) if parsed and "intent" in parsed else None
    predicted_group = resolve_group(parsed["coarse_group"]) if parsed and "coarse_group" in parsed else None
    return TechniqueRun(
        predicted_intent=predicted_intent,
        predicted_group_stated=predicted_group,
        raw_response=raw_response,
        calls=calls,
        extra=extra or {},
    )
