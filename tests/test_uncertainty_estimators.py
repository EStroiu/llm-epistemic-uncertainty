import unittest

from uncertainty_estimators import (
    estimate_nli_consistency,
    estimate_sampling_variance,
    estimate_self_disagreement,
)
from uncertainty_prompting import evaluate_clarity_interpretability


class TestEstimators(unittest.TestCase):
    def test_sampling_variance_output_range(self) -> None:
        prob_samples = [
            {"SUPPORTED": 0.7, "REFUTED": 0.2, "NOT_ENOUGH_INFO": 0.1},
            {"SUPPORTED": 0.6, "REFUTED": 0.3, "NOT_ENOUGH_INFO": 0.1},
            {"SUPPORTED": 0.2, "REFUTED": 0.5, "NOT_ENOUGH_INFO": 0.3},
        ]
        out = estimate_sampling_variance(prob_samples)
        self.assertGreaterEqual(out.uncertainty, 0.0)
        self.assertLessEqual(out.uncertainty, 1.0)
        self.assertEqual(out.name, "sampling_variance")

    def test_self_disagreement_increases_for_different_answers(self) -> None:
        similar = estimate_self_disagreement(
            ["SUPPORTED because Paris is in France.", "SUPPORTED because Paris is in France."],
            ["SUPPORTED", "SUPPORTED"],
        )
        diverse = estimate_self_disagreement(
            ["SUPPORTED due to evidence.", "REFUTED because the claim is false."],
            ["SUPPORTED", "REFUTED"],
        )
        self.assertGreater(diverse.uncertainty, similar.uncertainty)

    def test_nli_consistency(self) -> None:
        consistent = estimate_nli_consistency(["entailment", "entailment", "entailment"])
        inconsistent = estimate_nli_consistency(["contradiction", "neutral", "contradiction"])
        self.assertGreater(inconsistent.uncertainty, consistent.uncertainty)

    def test_prompt_clarity_scoring(self) -> None:
        plain = evaluate_clarity_interpretability("SUPPORTED. The claim is correct.")
        numeric = evaluate_clarity_interpretability(
            "SUPPORTED. The claim is correct. Confidence: 82%. Uncertainty reason: limited evidence."
        )
        self.assertGreaterEqual(numeric["clarity_score"], plain["clarity_score"])


if __name__ == "__main__":
    unittest.main()

