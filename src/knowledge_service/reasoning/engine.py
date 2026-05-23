"""Forward-chaining inference engine. Ontology-declared rules, Python-executed."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from knowledge_service._utils import compute_triple_hash
from knowledge_service.ontology.namespaces import (
    KS_GRAPH_ONTOLOGY,
    KS_INVERSE_PREDICATE,
    KS_TRANSITIVE_PREDICATE,
)
from knowledge_service.ontology.uri import KS as KS_PREFIX
from knowledge_service.ontology.uri import is_uri

logger = logging.getLogger(__name__)


def _triple_dict_hash(triple: dict) -> str:
    """Canonical hash of a trigger triple. Tolerates ``object`` vs ``object_``."""
    obj = triple.get("object") or triple.get("object_", "")
    return compute_triple_hash(triple["subject"], triple["predicate"], obj)


@dataclass
class DerivedTriple:
    subject: str
    predicate: str
    object_: str
    confidence: float
    derived_from: list[str]
    inference_method: str
    depth: int

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object_,
            "confidence": self.confidence,
            "knowledge_type": "inferred",
            "derived_from": self.derived_from,
            "inference_method": self.inference_method,
            "valid_from": None,
            "valid_until": None,
        }

    def compute_hash(self) -> str:
        return compute_triple_hash(self.subject, self.predicate, self.object_)


class InferenceRule:
    """Base class. Subclasses implement configure() and discover()."""

    name: str = "base"

    def configure(self, triple_store) -> None:
        pass

    def discover(self, triple: dict, triple_store, depth: int) -> list[DerivedTriple]:
        return []


class InverseRule(InferenceRule):
    """Materializes inverse triples. A p B → B p_inv A when p has ks:inversePredicate."""

    name = "inverse"

    def __init__(self):
        self._inverse_map: dict[str, str] = {}

    def configure(self, triple_store) -> None:
        rows = triple_store.query(f"""
            SELECT ?p ?inv WHERE {{
                GRAPH <{KS_GRAPH_ONTOLOGY}> {{
                    ?p <{KS_INVERSE_PREDICATE.value}> ?inv .
                }}
            }}
        """)
        for row in rows:
            p_uri = row["p"].value if hasattr(row["p"], "value") else str(row["p"])
            inv_uri = row["inv"].value if hasattr(row["inv"], "value") else str(row["inv"])
            self._inverse_map[p_uri] = inv_uri
            self._inverse_map[inv_uri] = p_uri
        logger.info("InverseRule loaded %d inverse pairs", len(rows))

    def discover(self, triple: dict, triple_store, depth: int) -> list[DerivedTriple]:
        pred = triple["predicate"]
        if pred not in self._inverse_map:
            return []
        # Object becomes subject in inverse — skip if object is a literal (non-URI)
        obj = triple.get("object") or triple.get("object_", "")
        if not is_uri(obj):
            return []
        inv_pred = self._inverse_map[pred]
        source_hash = _triple_dict_hash(triple)
        return [
            DerivedTriple(
                subject=triple["object"],
                predicate=inv_pred,
                object_=triple["subject"],
                confidence=triple["confidence"],
                derived_from=[source_hash],
                inference_method="inverse",
                depth=depth,
            )
        ]


class TransitiveRule(InferenceRule):
    """Closes transitive predicates. A p B, B p C → A p C."""

    name = "transitive"

    def __init__(self):
        self._transitive_predicates: set[str] = set()

    def configure(self, triple_store) -> None:
        rows = triple_store.query(f"""
            SELECT ?p WHERE {{
                GRAPH <{KS_GRAPH_ONTOLOGY}> {{
                    ?p <{KS_TRANSITIVE_PREDICATE.value}> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
                }}
            }}
        """)
        for row in rows:
            uri = row["p"].value if hasattr(row["p"], "value") else str(row["p"])
            self._transitive_predicates.add(uri)
        logger.info(
            "TransitiveRule loaded %d transitive predicates", len(self._transitive_predicates)
        )

    def discover(self, triple: dict, triple_store, depth: int) -> list[DerivedTriple]:
        pred = triple["predicate"]
        if pred not in self._transitive_predicates:
            return []
        subj = triple["subject"]
        obj = triple.get("object") or triple.get("object_", "")
        # Object becomes subject in transitive chain — skip if it's a literal (non-URI)
        if not is_uri(obj):
            return []
        conf = triple["confidence"]
        source_hash = _triple_dict_hash(triple)
        results = []

        # Forward: (A, p, B) + existing (B, p, ?C) → (A, p, C)
        forward = triple_store.get_triples(subject=obj, predicate=pred)
        for existing in forward:
            c_obj = existing["object"]
            if c_obj == subj:
                continue
            existing_conf = existing.get("confidence") or 0.0
            existing_hash = compute_triple_hash(obj, pred, c_obj)
            results.append(
                DerivedTriple(
                    subject=subj,
                    predicate=pred,
                    object_=c_obj,
                    confidence=conf * existing_conf,
                    derived_from=[source_hash, existing_hash],
                    inference_method="transitive",
                    depth=depth,
                )
            )

        # Backward: existing (?Z, p, A) + (A, p, B) → (Z, p, B)
        backward = triple_store.get_triples(object_=subj, predicate=pred)
        for existing in backward:
            z_subj = existing["subject"]
            if z_subj == obj:
                continue
            existing_conf = existing.get("confidence") or 0.0
            existing_hash = compute_triple_hash(z_subj, pred, subj)
            results.append(
                DerivedTriple(
                    subject=z_subj,
                    predicate=pred,
                    object_=obj,
                    confidence=existing_conf * conf,
                    derived_from=[existing_hash, source_hash],
                    inference_method="transitive",
                    depth=depth,
                )
            )
        return results


class TypeInheritanceRule(InferenceRule):
    """Propagates has_property through is_a chains."""

    name = "type_inheritance"

    def __init__(self):
        self._is_a_uri = f"{KS_PREFIX}is_a"
        self._has_property_uri = f"{KS_PREFIX}has_property"

    def configure(self, triple_store) -> None:
        pass

    def discover(self, triple: dict, triple_store, depth: int) -> list[DerivedTriple]:
        pred = triple["predicate"]
        subj = triple["subject"]
        obj = triple.get("object") or triple.get("object_", "")
        conf = triple["confidence"]
        source_hash = _triple_dict_hash(triple)
        results = []

        if pred == self._is_a_uri:
            # A is_a B → inherit B's properties — skip if B is a literal (non-URI)
            if not is_uri(obj):
                return []
            properties = triple_store.get_triples(subject=obj, predicate=self._has_property_uri)
            for prop in properties:
                prop_obj = prop["object"]
                prop_conf = prop.get("confidence") or 0.0
                prop_hash = compute_triple_hash(obj, self._has_property_uri, prop_obj)
                results.append(
                    DerivedTriple(
                        subject=subj,
                        predicate=self._has_property_uri,
                        object_=prop_obj,
                        confidence=conf * prop_conf,
                        derived_from=[source_hash, prop_hash],
                        inference_method="type_inheritance",
                        depth=depth,
                    )
                )
        elif pred == self._has_property_uri:
            # B has_property X → propagate to instances
            instances = triple_store.get_triples(object_=subj, predicate=self._is_a_uri)
            for inst in instances:
                inst_subj = inst["subject"]
                inst_conf = inst.get("confidence") or 0.0
                inst_hash = compute_triple_hash(inst_subj, self._is_a_uri, subj)
                results.append(
                    DerivedTriple(
                        subject=inst_subj,
                        predicate=self._has_property_uri,
                        object_=obj,
                        confidence=inst_conf * conf,
                        derived_from=[inst_hash, source_hash],
                        inference_method="type_inheritance",
                        depth=depth,
                    )
                )
        return results


class InferenceEngine:
    """BFS forward-chaining engine with depth cap and cycle detection."""

    def __init__(self, triple_store, rules: list[InferenceRule], max_depth: int = 3):
        self._store = triple_store
        self._rules = rules
        self._max_depth = max_depth

    def configure(self) -> None:
        for rule in self._rules:
            rule.configure(self._store)

    def run(self, trigger_triple: dict) -> list[DerivedTriple]:
        all_derived: list[DerivedTriple] = []
        seen_hashes: set[str] = set()

        trigger_hash = _triple_dict_hash(trigger_triple)
        seen_hashes.add(trigger_hash)

        queue: deque[tuple[dict, int]] = deque()
        queue.append((trigger_triple, 0))

        while queue:
            current_triple, current_depth = queue.popleft()
            if current_depth >= self._max_depth:
                continue

            for rule in self._rules:
                derived_list = rule.discover(current_triple, self._store, depth=current_depth + 1)
                for derived in derived_list:
                    h = derived.compute_hash()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    existing = self._store.get_triples(
                        subject=derived.subject,
                        predicate=derived.predicate,
                        object_=derived.object_,
                    )
                    if existing:
                        continue

                    all_derived.append(derived)
                    queue.append((derived.to_dict(), current_depth + 1))

        if all_derived:
            logger.info(
                "InferenceEngine: %d new triples derived from (%s, %s, ...)",
                len(all_derived),
                trigger_triple.get("subject", "?")[-40:],
                trigger_triple.get("predicate", "?")[-30:],
            )
        return all_derived


