"""Test community detection auto-trigger conditions."""

import time

from knowledge_service.ingestion.worker import _should_rebuild_communities


class TestCommunityTrigger:
    def test_triggers_when_conditions_met(self):
        """Should trigger when triples > min and cooldown elapsed."""
        assert _should_rebuild_communities(
            triples_created=5,
            total_triples=100,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_no_triples_created(self):
        assert not _should_rebuild_communities(
            triples_created=0,
            total_triples=100,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_below_threshold(self):
        assert not _should_rebuild_communities(
            triples_created=5,
            total_triples=30,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_cooldown_not_elapsed(self):
        assert not _should_rebuild_communities(
            triples_created=5,
            total_triples=100,
            min_triples=50,
            last_rebuild=time.time(),  # just now
            cooldown=3600,
        )
