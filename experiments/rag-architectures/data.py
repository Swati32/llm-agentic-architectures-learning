"""MultiHop-RAG loading and sampling.

MultiHop-RAG (https://arxiv.org/abs/2401.15391) ships two files: a corpus
of 609 news articles, and 2,556 queries, each labeled with a question_type
(inference_query, comparison_query, temporal_query, or null_query) and an
evidence_list of the exact source sentences ("facts") a correct answer
needs. null_query questions have an empty evidence_list on purpose: they
are not answerable from the corpus, and a system that keeps answering them
anyway is hallucinating.

The two files ship as plain JSON on the Hub, not a Parquet dataset config,
so this loads them directly with huggingface_hub rather than through
`datasets.load_dataset`.
"""

import json
import random

from huggingface_hub import hf_hub_download

REPO_ID = "yixuantt/MultiHopRAG"
QUESTION_TYPES = ["inference_query", "comparison_query", "temporal_query", "null_query"]


def load_corpus() -> list[dict]:
    """Returns the 609 source articles: [{"title", "body", "source", ...}, ...]."""
    path = hf_hub_download(REPO_ID, "corpus.json", repo_type="dataset")
    return json.loads(open(path).read())


def load_queries() -> list[dict]:
    """Returns the 2,556 queries: [{"query", "answer", "question_type", "evidence_list"}, ...]."""
    path = hf_hub_download(REPO_ID, "MultiHopRAG.json", repo_type="dataset")
    raw = json.loads(open(path).read())
    queries = []
    for i, row in enumerate(raw):
        queries.append(
            {
                "id": f"q{i}",
                "question": row["query"],
                "answer": row["answer"],
                "question_type": row["question_type"],
                "evidence_facts": [ev["fact"] for ev in row["evidence_list"]],
                "evidence_titles": list({ev["title"] for ev in row["evidence_list"]}),
            }
        )
    return queries


def stratified_sample(queries: list[dict], target_size: int, seed: int = 0) -> list[dict]:
    """Splits evenly across the 4 question types, since they stress a RAG
    architecture differently: inference and comparison need the right
    chunks combined correctly, temporal needs date-aware reasoning over
    those chunks, and null needs the system to recognize the corpus
    doesn't have the answer at all rather than confabulate one."""
    by_type: dict[str, list[dict]] = {qtype: [] for qtype in QUESTION_TYPES}
    for query in queries:
        by_type[query["question_type"]].append(query)

    rng = random.Random(seed)
    for examples in by_type.values():
        rng.shuffle(examples)

    per_type = target_size // len(QUESTION_TYPES)
    sample = []
    for qtype in QUESTION_TYPES:
        sample.extend(by_type[qtype][:per_type])
    rng.shuffle(sample)
    return sample
