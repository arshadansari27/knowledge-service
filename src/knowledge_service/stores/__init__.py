from dataclasses import dataclass

from knowledge_service.stores.triples import TripleStore
from knowledge_service.stores.content import ContentStore
from knowledge_service.stores.entities import EntityStore
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.theses import ThesisStore
from knowledge_service.ingestion.outbox import OutboxStore


@dataclass
class Stores:
    triples: TripleStore
    content: ContentStore
    entities: EntityStore
    provenance: ProvenanceStore
    theses: ThesisStore
    outbox: OutboxStore
    pg_pool: object  # asyncpg.Pool
