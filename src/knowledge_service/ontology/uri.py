"""Canonical URI normalization. Single source of truth for all URI construction."""

import re

KS = "http://knowledge.local/schema/"
KS_DATA = "http://knowledge.local/data/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def is_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "urn:"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "_", value.lower().strip())
    return re.sub(r"_+", "_", slug).strip("_")


def to_entity_uri(value: str) -> str:
    if is_uri(value):
        return value
    return f"{KS_DATA}{slugify(value)}"


def to_predicate_uri(value: str) -> str:
    if is_uri(value):
        return value
    return f"{KS}{slugify(value)}"
