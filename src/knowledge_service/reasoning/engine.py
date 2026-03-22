"""ReasoningEngine: probabilistic inference over knowledge claims using ProbLog."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence

from problog import get_evaluatable
from problog.program import PrologString

logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    """Result of a contradiction check between claims."""

    probability: float
    involved_claims: list = field(default_factory=list)


@dataclass
class InferenceResult:
    """Result of a ProbLog inference query."""

    query: str
    probability: float


class ReasoningEngine:
    """Probabilistic reasoning engine backed by ProbLog.

    Loads Prolog rule files from ``rules_dir`` and exposes high-level methods
    for combining evidence, detecting contradictions, and running arbitrary
    ProbLog queries over a set of probabilistic claims.
    """

    def __init__(self, rules_dir: str | Path) -> None:
        self._rules_dir = Path(rules_dir)
        parts = []
        for pl_file in sorted(self._rules_dir.glob("*.pl")):
            parts.append(pl_file.read_text(encoding="utf-8"))
        self._all_rules: str = "\n".join(parts)
        self._base_rules: str = self._load_rules("base.pl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combine_evidence(self, confidences: Sequence[float]) -> float:
        """Combine independent evidence sources using the Noisy-OR formula.

        Noisy-OR: P = 1 - product(1 - ci)

        Args:
            confidences: Individual confidence scores in [0, 1].

        Returns:
            Combined probability in [0, 1].
        """
        if not confidences:
            return 0.0
        # product of failure probabilities
        failure_product = math.prod(1.0 - c for c in confidences)
        return 1.0 - failure_product

    def check_contradiction(
        self,
        new_claim: tuple,
        existing_claims: list[tuple],
        opposites: list[tuple[str, str]],
    ) -> ContradictionResult:
        """Check whether ``new_claim`` contradicts any of ``existing_claims``.

        A contradiction arises when a new claim and an existing claim share the
        same subject and object but have predicates that are declared opposites.

        The returned probability is the ProbLog-computed joint probability that
        both conflicting claims are simultaneously true, i.e. the product of
        their individual confidences (conjunction).

        Args:
            new_claim: ``(subject, predicate, object, confidence)``.
            existing_claims: List of ``(subject, predicate, object, confidence)``.
            opposites: Pairs ``(pred_a, pred_b)`` that are mutually exclusive.

        Returns:
            :class:`ContradictionResult` with computed probability.
        """
        all_claims = [new_claim] + list(existing_claims)

        # If no opposites defined, no contradiction is possible.
        if not opposites:
            return ContradictionResult(probability=0.0, involved_claims=[])

        program_parts = self._program_preamble()

        # Emit probabilistic claim facts (4-arity to keep sources independent)
        for idx, claim in enumerate(all_claims):
            self._emit_claim_facts(claim, idx, program_parts)

        program_parts.append("")

        # Emit opposite/2 facts
        for pred_a, pred_b in opposites:
            a_atom = _to_atom(pred_a)
            b_atom = _to_atom(pred_b)
            program_parts.append(f"opposite({a_atom}, {b_atom}).")
            program_parts.append(f"opposite({b_atom}, {a_atom}).")

        program_parts.append("")

        # Build contradiction query for the new claim against each existing claim
        contradiction_prob = 0.0
        involved: list[tuple] = []

        new_subj, new_pred, new_obj = new_claim[0], new_claim[1], new_claim[2]
        ns = _to_atom(new_subj)
        np_ = _to_atom(new_pred)
        no = _to_atom(new_obj)

        # Find any opposite predicate that appears in existing claims
        relevant_queries: list[str] = []
        for ec in existing_claims:
            subj, pred, obj = ec[0], ec[1], ec[2]
            s = _to_atom(subj)
            p = _to_atom(pred)
            o = _to_atom(obj)
            # Check if (new_pred, pred) or (pred, new_pred) is an opposite pair
            for pred_a, pred_b in opposites:
                a = _to_atom(pred_a)
                b = _to_atom(pred_b)
                if (np_ == a and p == b) or (np_ == b and p == a):
                    if s == ns and o == no:
                        query_atom = f"contradicts({ns}, {np_}, {no}, {p})"
                        relevant_queries.append(query_atom)
                        involved.append((subj, pred, obj))

        if not relevant_queries:
            return ContradictionResult(probability=0.0, involved_claims=[])

        # Add query directives
        for q in relevant_queries:
            program_parts.append(f"query({q}).")

        program = "\n".join(program_parts)

        try:
            db = PrologString(program)
            results = get_evaluatable().create_from(db).evaluate()
            # Take the maximum contradiction probability across all queries
            for term, prob in results.items():
                if prob > contradiction_prob:
                    contradiction_prob = prob
        except Exception as exc:
            logger.warning("ProbLog contradiction check failed, using product fallback: %s", exc)
            # Fall back to product of confidences if ProbLog fails
            new_conf = new_claim[3]
            for ec in existing_claims:
                subj, pred, obj, conf = ec[0], ec[1], ec[2], ec[3]
                for pred_a, pred_b in opposites:
                    if (new_pred in (pred_a, pred_b)) and (pred in (pred_a, pred_b)):
                        if new_pred != pred and subj == new_subj and obj == new_obj:
                            contradiction_prob = max(contradiction_prob, new_conf * conf)
                            involved.append((subj, pred, obj))

        return ContradictionResult(
            probability=contradiction_prob,
            involved_claims=involved,
        )

    def infer(
        self,
        query: str,
        claims: list[tuple],
    ) -> list[InferenceResult]:
        """Run a ProbLog query over a set of probabilistic claims.

        Each claim is represented as a 4-arity fact
        ``confidence::claims(S, P, O, sourceN).`` so that ProbLog treats
        repeated (S, P, O) triples from different sources as independent
        random variables and computes Noisy-OR combination automatically.

        Args:
            query: A Prolog goal string, e.g.
                   ``"supported(cold_exposure, increases, dopamine)"``.
            claims: List of 4-tuples ``(subject, predicate, object, confidence)``
                    or 5-tuples ``(subject, predicate, object, confidence, meta)``
                    where *meta* is a dict with optional keys ``knowledge_type``,
                    ``valid_from``, ``valid_until``.

        Returns:
            List of :class:`InferenceResult` ordered by probability descending.
            Returns an empty list when the query is provably false (prob 0).
        """
        program_parts = self._program_preamble()

        for idx, claim in enumerate(claims):
            self._emit_claim_facts(claim, idx, program_parts)

        program_parts.append("")
        program_parts.append(f"query({query}).")

        program = "\n".join(program_parts)

        try:
            db = PrologString(program)
            raw_results = get_evaluatable().create_from(db).evaluate()
        except Exception as exc:
            logger.warning("ProbLog inference failed, using Python fallback: %s", exc)
            return self._fallback_infer(query, claims)

        results: list[InferenceResult] = []
        for term, prob in raw_results.items():
            results.append(InferenceResult(query=str(term), probability=float(prob)))

        # Sort descending by probability, filter out zero if any unsupported
        results.sort(key=lambda r: r.probability, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _program_preamble(self) -> list[str]:
        """Return the common program header: rules, current_date, and stub facts.

        ProbLog errors if a predicate used in a rule body has zero clauses.
        We emit sentinel facts with a dummy sentinel value so the predicates
        are always defined.
        """
        parts: list[str] = [self._all_rules, ""]
        # Inject current_date/1 (integer YYYYMMDD) for temporal reasoning
        parts.append(f"current_date({_date_to_int(date.today().isoformat())}).")
        # Stub facts so ProbLog never sees an undefined predicate
        parts.append("valid_from(__stub, __stub, __stub, 0).")
        parts.append("valid_until(__stub, __stub, __stub, 0).")
        parts.append("claim_type(__stub, __stub, __stub, __stub).")
        parts.append("inverse(__stub, __stub).")
        parts.append("")
        return parts

    def _emit_claim_facts(self, claim: tuple, idx: int, program_parts: list[str]) -> None:
        """Emit ProbLog facts for a single claim tuple (4 or 5 elements)."""
        subj, pred, obj, conf = claim[0], claim[1], claim[2], claim[3]
        meta: dict = claim[4] if len(claim) > 4 else {}
        s_atom = _to_atom(subj)
        p_atom = _to_atom(pred)
        o_atom = _to_atom(obj)
        program_parts.append(f"{conf}::claims({s_atom}, {p_atom}, {o_atom}, source{idx}).")

        # Emit metadata facts when present
        if meta.get("knowledge_type"):
            program_parts.append(
                f"claim_type({s_atom}, {p_atom}, {o_atom}, {meta['knowledge_type']})."
            )
        if meta.get("valid_from"):
            program_parts.append(
                f"valid_from({s_atom}, {p_atom}, {o_atom}, {_date_to_int(meta['valid_from'])})."
            )
        if meta.get("valid_until"):
            program_parts.append(
                f"valid_until({s_atom}, {p_atom}, {o_atom}, {_date_to_int(meta['valid_until'])})."
            )

    def _load_rules(self, filename: str) -> str:
        """Load a Prolog rules file from the rules directory."""
        path = self._rules_dir / filename
        return path.read_text(encoding="utf-8")

    def _fallback_infer(
        self,
        query: str,
        claims: list[tuple],
    ) -> list[InferenceResult]:
        """Pure-Python fallback for supported/3 queries when ProbLog is unavailable."""
        import re

        m = re.match(r"supported\((\w+),\s*(\w+),\s*(\w+)\)", query.strip())
        if not m:
            return []

        qs, qp, qo = m.group(1), m.group(2), m.group(3)
        matching_confs = [
            claim[3] for claim in claims if claim[0] == qs and claim[1] == qp and claim[2] == qo
        ]

        if not matching_confs:
            return []

        prob = self.combine_evidence(matching_confs)
        return [InferenceResult(query=query, probability=prob)]


def _date_to_int(iso_date: str) -> int:
    """Convert an ISO date string (YYYY-MM-DD) to an integer YYYYMMDD for ProbLog arithmetic."""
    return int(iso_date.replace("-", ""))


def _to_atom(value: str) -> str:
    """Convert a string value to a valid Prolog atom.

    Atoms containing only alphanumeric characters and underscores that start
    with a lowercase letter are bare atoms.  Everything else is quoted with
    single quotes.
    """
    import re

    if re.match(r"^[a-z][a-zA-Z0-9_]*$", value):
        return value
    # Escape single quotes inside the value
    escaped = value.replace("'", "\\'")
    return f"'{escaped}'"
