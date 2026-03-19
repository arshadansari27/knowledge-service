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


class TestFallbackLogging:
    def test_fallback_logs_warning_on_problog_failure(self, engine, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="knowledge_service.reasoning.engine"):
            # Force fallback by passing a query ProbLog can't parse
            engine.infer("unsupported_predicate(??invalid syntax??)", claims=[("a","b","c",0.5)])
        assert any("fallback" in r.message.lower() or "problog" in r.message.lower()
                   for r in caplog.records)
