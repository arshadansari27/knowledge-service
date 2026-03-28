"""Bootstrap spaCy NLP pipeline with optional entity linker."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_spacy_nlp(spacy_data_dir: str) -> Any | None:
    """Load spaCy model with entity linker. Returns None if unavailable."""
    try:
        import spacy  # noqa: PLC0415
    except ImportError:
        logger.warning("spaCy is not installed — NLP phase will be skipped")
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found — NLP phase will be skipped. "
            "Install with: python -m spacy download en_core_web_sm"
        )
        return None

    try:
        nlp.add_pipe(
            "entityLinker",
            last=True,
            config={"resolve_pronouns": False, "data_dir": spacy_data_dir},
        )
        logger.info("spaCy entity linker loaded from %s", spacy_data_dir)
    except Exception:
        logger.warning(
            "spaCy entity linker unavailable — proceeding without Wikidata linking",
            exc_info=True,
        )

    return nlp
