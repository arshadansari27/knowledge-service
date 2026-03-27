from dataclasses import dataclass

from knowledge_service.stores.triples import TripleStore
from knowledge_service.stores.content import ContentStore
from knowledge_service.stores.entities import EntityStore
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.theses import ThesisStore


@dataclass
class Stores:
    triples: TripleStore
    content: ContentStore
    entities: EntityStore
    provenance: ProvenanceStore
    theses: ThesisStore
    pg_pool: object  # asyncpg.Pool
