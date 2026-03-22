import pytest
from knowledge_service.reasoning.engine import ReasoningEngine


@pytest.fixture
def engine():
    rules_dir = "src/knowledge_service/reasoning/rules"
    return ReasoningEngine(rules_dir)


class TestCombineEvidence:
    def test_single_source(self, engine):
        result = engine.combine_evidence([0.7])
        assert result == pytest.approx(0.7)

    def test_two_independent_sources(self, engine):
        # Noisy-OR: 1 - (0.3 * 0.4) = 0.88
        result = engine.combine_evidence([0.7, 0.6])
        assert result == pytest.approx(0.88)

    def test_empty_sources(self, engine):
        result = engine.combine_evidence([])
        assert result == pytest.approx(0.0)


class TestCheckContradiction:
    def test_detects_contradiction_with_opposite_predicates(self, engine):
        result = engine.check_contradiction(
            new_claim=("cold_exposure", "increases", "dopamine", 0.7),
            existing_claims=[("cold_exposure", "does_not_increase", "dopamine", 0.6)],
            opposites=[("increases", "does_not_increase")],
        )
        assert result.probability > 0
        assert result.probability < 1

    def test_no_contradiction_with_same_claim(self, engine):
        result = engine.check_contradiction(
            new_claim=("cold_exposure", "increases", "dopamine", 0.7),
            existing_claims=[("cold_exposure", "increases", "dopamine", 0.6)],
            opposites=[],
        )
        assert result.probability == pytest.approx(0.0)


class TestInfer:
    def test_derives_conclusion_from_multiple_claims(self, engine):
        """When multiple independent claims support the same assertion,
        ProbLog should derive a Conclusion with combined probability."""
        claims = [
            ("cold_exposure", "increases", "dopamine", 0.7),
            ("cold_exposure", "increases", "dopamine", 0.6),
        ]
        results = engine.infer(
            query="supported(cold_exposure, increases, dopamine)",
            claims=claims,
        )
        assert len(results) >= 1
        # Combined confidence should be higher than either individual source
        assert results[0].probability > 0.7
        assert results[0].probability == pytest.approx(0.88)  # Noisy-OR

    def test_infer_returns_empty_for_unsupported_query(self, engine):
        claims = [("a", "b", "c", 0.5)]
        results = engine.infer(
            query="supported(x, y, z)",
            claims=claims,
        )
        assert len(results) == 0 or results[0].probability == pytest.approx(0.0)


class TestRulesLoading:
    def test_base_rules_loaded(self, engine):
        """Engine must have base rules loaded."""
        assert "supported" in engine._base_rules

    def test_all_rule_files_loaded(self, engine):
        """Engine should load knowledge_types and temporal rules too."""
        assert hasattr(engine, "_all_rules")
        assert "knowledge_type" in engine._all_rules  # from knowledge_types.pl
        assert "expired" in engine._all_rules  # from temporal.pl
        # Must not crash with all rules loaded
        try:
            engine.infer("supported(a, b, c)", claims=[("a", "b", "c", 0.5)])
        except Exception as exc:
            pytest.fail(f"infer() raised unexpectedly with all rules loaded: {exc}")


class TestGlobLoading:
    """Task 5: Verify glob-loading picks up all .pl files including new ones."""

    def test_loads_all_pl_files_including_new(self, engine):
        assert "indirect_link" in engine._all_rules
        assert "causal_propagation" in engine._all_rules
        assert "high_confidence" in engine._all_rules
        assert "authoritative" in engine._all_rules
        assert "currently_valid" in engine._all_rules
        assert "inverse_holds" in engine._all_rules
        assert "corroborated" in engine._all_rules


class TestInferWithMetadata:
    """Task 5: Verify 5-tuple claim support with metadata dicts."""

    def test_infer_accepts_5_tuple_claims(self, engine):
        claims = [("cold_exposure", "increases", "dopamine", 0.7, {"knowledge_type": "fact"})]
        results = engine.infer(
            query="authoritative(cold_exposure, increases, dopamine)", claims=claims
        )
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_infer_backward_compat_4_tuple(self, engine):
        claims = [("a", "b", "c", 0.5)]
        results = engine.infer(query="supported(a, b, c)", claims=claims)
        assert len(results) >= 1


class TestInferenceChains:
    """Task 5: Test inference_chains.pl rules."""

    def test_indirect_link_2hop(self, engine):
        claims = [("a", "causes", "b", 0.8), ("b", "causes", "c", 0.7)]
        results = engine.infer("indirect_link(a, causes, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_causal_propagation_increases(self, engine):
        claims = [
            ("stress", "causes", "cortisol", 0.8),
            ("cortisol", "increases", "inflammation", 0.7),
        ]
        results = engine.infer("causal_propagation(stress, inflammation)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_no_indirect_link_self(self, engine):
        claims = [("a", "causes", "a", 0.8)]
        results = engine.infer("indirect_link(a, causes, a)", claims=claims)
        assert len(results) == 0 or results[0].probability == pytest.approx(0.0)


class TestConfidenceRules:
    """Task 5: Test confidence.pl rules."""

    def test_high_confidence_no_conflict(self, engine):
        claims = [("a", "b", "c", 0.9)]
        results = engine.infer("high_confidence(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_contested_with_conflict(self, engine):
        claims = [("a", "b", "c", 0.9), ("a", "b", "d", 0.7)]
        results = engine.infer("contested(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_authoritative_fact(self, engine):
        claims = [("a", "b", "c", 0.95, {"knowledge_type": "fact"})]
        results = engine.infer("authoritative(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0


class TestTemporalRules:
    """Task 5: Test temporal.pl rules."""

    def test_expired_claim(self, engine):
        claims = [("btc", "has_property", "50000", 0.9, {"valid_until": "2020-01-01"})]
        results = engine.infer("expired(btc, has_property, '50000')", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_currently_valid(self, engine):
        claims = [("btc", "has_property", "100000", 0.9, {"valid_from": "2020-01-01"})]
        results = engine.infer("currently_valid(btc, has_property, '100000')", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0


class TestInverseHolds:
    """Task 5: Test inverse_holds rule from base.pl."""

    def test_inverse_holds_contains_part_of(self, engine):
        claims = [("body", "contains", "heart", 0.95)]
        original_all_rules = engine._all_rules
        engine._all_rules = original_all_rules + "\ninverse(contains, part_of).\n"
        try:
            results = engine.infer("inverse_holds(heart, part_of, body)", claims=claims)
            assert len(results) >= 1
            assert results[0].probability > 0
        finally:
            engine._all_rules = original_all_rules


class TestFallbackLogging:
    def test_fallback_logs_warning_on_problog_failure(self, engine, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="knowledge_service.reasoning.engine"):
            # Force fallback by passing a query ProbLog can't parse
            engine.infer("unsupported_predicate(??invalid syntax??)", claims=[("a", "b", "c", 0.5)])
        assert any(
            "fallback" in r.message.lower() or "problog" in r.message.lower()
            for r in caplog.records
        )
