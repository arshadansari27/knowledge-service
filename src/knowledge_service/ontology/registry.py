"""Domain registry. Reads predicates, synonyms, and materiality from ontology."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from knowledge_service.ontology.uri import KS

logger = logging.getLogger(__name__)

DEFAULT_MATERIALITY = 0.5


@dataclass
class PredicateInfo:
    uri: str
    label: str
    domain: str
    materiality_weight: float = DEFAULT_MATERIALITY
    synonyms: list[str] = field(default_factory=list)


class DomainRegistry:
    def __init__(self, triple_store, prompts_dir: Path):
        self._store = triple_store
        self._prompts_dir = prompts_dir
        self._predicates: dict[str, list[PredicateInfo]] = {}  # domain -> list
        self._synonyms: dict[str, str] = {}  # synonym_label -> canonical URI
        self._materiality: dict[str, float] = {}  # uri -> weight
        self._prompts: dict[str, str] = {}  # name -> text

    def load(self):
        """Load predicate metadata from ontology graph via SPARQL."""
        rows = self._store.query(f"""
            SELECT ?p ?label ?domain ?weight WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    ?p a <{KS}Predicate> ;
                       <http://www.w3.org/2000/01/rdf-schema#label> ?label ;
                       <{KS}domain> ?domain .
                    OPTIONAL {{ ?p <{KS}materialityWeight> ?weight . }}
                }}
            }}
        """)
        for row in rows:
            uri = row["p"].value if hasattr(row["p"], "value") else str(row["p"])
            label_val = row["label"].value if hasattr(row["label"], "value") else str(row["label"])
            domain_val = (
                row["domain"].value if hasattr(row["domain"], "value") else str(row["domain"])
            )
            weight_val = (
                float(row["weight"].value)
                if row.get("weight") and row["weight"]
                else DEFAULT_MATERIALITY
            )

            syns = self._load_synonyms(uri)
            info = PredicateInfo(
                uri=uri,
                label=label_val,
                domain=domain_val,
                materiality_weight=weight_val,
                synonyms=syns,
            )
            self._predicates.setdefault(domain_val, []).append(info)
            self._materiality[uri] = weight_val
            # Register canonical label (slugified) as mapping to URI
            canonical_slug = label_val.lower().strip().replace(" ", "_")
            self._synonyms[canonical_slug] = uri
            for syn in syns:
                slug = syn.lower().strip().replace(" ", "_")
                self._synonyms[slug] = uri

        # Load prompt overrides
        if self._prompts_dir.exists():
            for f in self._prompts_dir.glob("*.txt"):
                self._prompts[f.stem] = f.read_text()

        logger.info(
            "DomainRegistry loaded: %d predicates, %d synonyms, %d prompts",
            sum(len(v) for v in self._predicates.values()),
            len(self._synonyms),
            len(self._prompts),
        )

    def _load_synonyms(self, predicate_uri: str) -> list[str]:
        rows = self._store.query(f"""
            SELECT ?syn WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    <{predicate_uri}> <{KS}synonym> ?syn .
                }}
            }}
        """)
        return [r["syn"].value if hasattr(r["syn"], "value") else str(r["syn"]) for r in rows]

    def get_predicates(self, domains: list[str] | None = None) -> list[PredicateInfo]:
        if domains is None:
            return [p for preds in self._predicates.values() for p in preds]
        result = []
        for d in domains:
            result.extend(self._predicates.get(d, []))
        return result

    def resolve_synonym(self, label: str) -> str:
        slug = label.lower().strip().replace(" ", "_")
        if slug in self._synonyms:
            return self._synonyms[slug]
        return label

    def get_materiality(self, predicate_uri: str) -> float:
        return self._materiality.get(predicate_uri, DEFAULT_MATERIALITY)

    def get_prompt(self, name: str) -> str | None:
        return self._prompts.get(name)

    def get_domains_for_entity_types(self, rdf_types: list[str]) -> list[str]:
        return list(self._predicates.keys())
