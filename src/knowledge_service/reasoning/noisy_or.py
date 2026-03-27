"""Evidence combination via Noisy-OR. Replaces the 332-line ReasoningEngine."""

from math import prod


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    return 1.0 - prod(1.0 - c for c in confidences)
