"""Aggregates the raw pairwise judgments from judge.py into the win-rate
numbers the README and dashboard actually report.
"""

from collections import defaultdict


def pairwise_summary(judgments):
    """judgments: list of {technique_a, technique_b, winner} where winner
    is 'a', 'b', or 'tie'. Returns, for each unordered technique pair, win
    counts and rates for both techniques plus the tie count."""
    pairs = defaultdict(lambda: {"a_wins": 0, "b_wins": 0, "ties": 0, "total": 0})
    for j in judgments:
        key = (j["technique_a"], j["technique_b"])
        bucket = pairs[key]
        bucket["total"] += 1
        if j["winner"] == "a":
            bucket["a_wins"] += 1
        elif j["winner"] == "b":
            bucket["b_wins"] += 1
        elif j["winner"] == "tie":
            bucket["ties"] += 1

    summary = []
    for (technique_a, technique_b), bucket in pairs.items():
        total = bucket["total"]
        if total == 0:
            continue
        summary.append(
            {
                "technique_a": technique_a,
                "technique_b": technique_b,
                "a_win_rate": bucket["a_wins"] / total,
                "b_win_rate": bucket["b_wins"] / total,
                "tie_rate": bucket["ties"] / total,
                "n": total,
            }
        )
    return summary


def overall_win_rates(judgments, techniques):
    """Each technique's share of non-tied pairwise judgments it won,
    across every pair it appeared in. This is the single number the
    Comparison tab leads with."""
    wins = defaultdict(int)
    decided = defaultdict(int)
    for j in judgments:
        if j["winner"] == "tie":
            continue
        winner_technique = j["technique_a"] if j["winner"] == "a" else j["technique_b"]
        loser_technique = j["technique_b"] if j["winner"] == "a" else j["technique_a"]
        wins[winner_technique] += 1
        decided[winner_technique] += 1
        decided[loser_technique] += 1

    return {
        technique: (wins[technique] / decided[technique] if decided[technique] else None)
        for technique in techniques
    }
