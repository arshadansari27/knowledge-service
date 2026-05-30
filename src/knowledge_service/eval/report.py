"""Report types + pure aggregation and summary-table formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

_METRIC_KEYS = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "correctness",
)


@dataclass
class QueryResult:
    query_id: str
    mode: str
    query_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    answer: str = ""
    rationale: str = ""


def aggregate(results: list[QueryResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Average each metric within every (mode, query_type) bucket.

    Returns a mapping keyed by (mode, query_type) to averaged metrics plus a
    "count" of contributing queries.
    """
    buckets: dict[tuple[str, str], list[QueryResult]] = {}
    for r in results:
        buckets.setdefault((r.mode, r.query_type), []).append(r)

    agg: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in buckets.items():
        cell: dict[str, float] = {}
        for mk in _METRIC_KEYS:
            vals = [row.metrics.get(mk, 0.0) for row in rows]
            cell[mk] = sum(vals) / len(vals) if vals else 0.0
        cell["count"] = len(rows)
        agg[key] = cell
    return agg


def format_summary_table(agg: dict[tuple[str, str], dict[str, float]]) -> str:
    """Render the aggregated metrics as a fixed-width text table."""
    header = (
        f"{'mode':<12} {'query_type':<10} "
        f"{'recall@k':>9} {'prec@k':>8} {'mrr':>6} {'ndcg':>6} "
        f"{'faithful':>9} {'correct':>8} {'n':>4}"
    )
    lines = [header, "-" * len(header)]
    for mode, qtype in sorted(agg.keys()):
        cell = agg[(mode, qtype)]
        lines.append(
            f"{mode:<12} {qtype:<10} "
            f"{cell['recall_at_k']:>9.3f} {cell['precision_at_k']:>8.3f} "
            f"{cell['mrr']:>6.3f} {cell['ndcg_at_k']:>6.3f} "
            f"{cell['faithfulness']:>9.3f} {cell['correctness']:>8.3f} "
            f"{int(cell['count']):>4}"
        )
    return "\n".join(lines)
