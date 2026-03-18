"""Reasoning engine for probabilistic inference over knowledge claims."""

from knowledge_service.reasoning.engine import (
    ContradictionResult,
    InferenceResult,
    ReasoningEngine,
)

__all__ = ["ReasoningEngine", "ContradictionResult", "InferenceResult"]
