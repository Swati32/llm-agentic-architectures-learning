from techniques import (
    chain_of_thought,
    few_shot_fixed,
    few_shot_retrieval,
    hierarchical,
    self_consistency,
    zero_shot,
    zero_shot_schema,
)

TECHNIQUES = {
    "zero_shot": zero_shot,
    "zero_shot_schema": zero_shot_schema,
    "few_shot_fixed": few_shot_fixed,
    "few_shot_retrieval": few_shot_retrieval,
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
    "hierarchical": hierarchical,
}
