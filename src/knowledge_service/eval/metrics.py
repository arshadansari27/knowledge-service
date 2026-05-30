"""Pure retrieval metrics. No I/O. Inputs: ranked retrieved ids + a relevant set."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant ids found in the top-k retrieved ids."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant."""
    relevant_set = set(relevant)
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant_set) / len(top)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none)."""
    relevant_set = set(relevant)
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / idx
    return 0.0


def dcg(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Discounted cumulative gain with binary gains (1 if relevant else 0)."""
    relevant_set = set(relevant)
    total = 0.0
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            total += 1.0 / math.log2(idx + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized DCG at k with binary relevance."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    actual = dcg(retrieved[:k], relevant_set)
    ideal_hits = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return actual / ideal if ideal else 0.0
