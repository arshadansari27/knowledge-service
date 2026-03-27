import pytest
from knowledge_service.reasoning.noisy_or import noisy_or


class TestNoisyOr:
    def test_single_source(self):
        assert noisy_or([0.8]) == pytest.approx(0.8)

    def test_two_sources(self):
        # 1 - (1-0.7)(1-0.8) = 1 - 0.06 = 0.94
        assert noisy_or([0.7, 0.8]) == pytest.approx(0.94)

    def test_three_sources(self):
        # 1 - (1-0.5)(1-0.6)(1-0.7) = 1 - 0.06 = 0.94
        assert noisy_or([0.5, 0.6, 0.7]) == pytest.approx(0.94)

    def test_empty(self):
        assert noisy_or([]) == pytest.approx(0.0)

    def test_zero_confidence(self):
        assert noisy_or([0.0, 0.8]) == pytest.approx(0.8)

    def test_full_confidence(self):
        assert noisy_or([1.0, 0.5]) == pytest.approx(1.0)
