"""Forward-chaining inference engine. Ontology-declared rules, Python-executed."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from pyoxigraph import Literal, NamedNode, Triple

from knowledge_service.ontology.namespaces import KS_GRAPH_ONTOLOGY, KS_INVERSE_PREDICATE
from knowledge_service.ontology.uri import is_uri

logger = logging.getLogger(__name__)


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
        s = NamedNode(self.subject)
        p = NamedNode(self.predicate)
        o = NamedNode(self.object_) if is_uri(self.object_) else Literal(self.object_)
        return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()


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
        inv_pred = self._inverse_map[pred]
        source_hash = _compute_trigger_hash(triple)
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


def _compute_trigger_hash(triple: dict) -> str:
    """Compute the SHA-256 hash of a trigger triple dict."""
    s = NamedNode(triple["subject"])
    p = NamedNode(triple["predicate"])
    o_val = triple.get("object") or triple.get("object_", "")
    o = NamedNode(o_val) if is_uri(o_val) else Literal(o_val)
    return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()


def _compute_triple_hash_from_parts(subject: str, predicate: str, object_: str) -> str:
    s = NamedNode(subject)
    p = NamedNode(predicate)
    o = NamedNode(object_) if is_uri(object_) else Literal(object_)
    return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()
