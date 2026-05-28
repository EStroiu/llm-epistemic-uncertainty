import unittest

from claim_judging import CONTRADICTS, NOT_ADDRESSED, SUPPORTS
from claim_uncertainty import (
    build_claim_consistency_estimator,
    combine_semantic_and_logit_uncertainty,
    summarize_claim_uncertainties,
)


class TestClaimUncertainty(unittest.TestCase):
    def test_combines_semantic_and_logit_uncertainty(self) -> None:
        combined = combine_semantic_and_logit_uncertainty(
            semantic_uncertainty=0.5,
            logit_uncertainty=0.2,
            semantic_weight=0.65,
            logit_weight=0.35,
        )

        self.assertAlmostEqual(combined, 0.395)

    def test_combination_falls_back_to_semantic_only(self) -> None:
        combined = combine_semantic_and_logit_uncertainty(
            semantic_uncertainty=0.25,
            logit_uncertainty=None,
        )

        self.assertAlmostEqual(combined, 0.25)

    def test_combination_returns_none_without_signals(self) -> None:
        combined = combine_semantic_and_logit_uncertainty(
            semantic_uncertainty=None,
            logit_uncertainty=None,
        )

        self.assertIsNone(combined)

    def test_summarizes_claim_judgment_rates(self) -> None:
        claims = [
            {"claim_id": "c1", "text": "Paris is the capital of France.", "logit_uncertainty": 0.2}
        ]
        judgments = [
            {"claim_id": "c1", "judgment": SUPPORTS},
            {"claim_id": "c1", "judgment": CONTRADICTS},
            {"claim_id": "c1", "judgment": NOT_ADDRESSED},
            {"claim_id": "c1", "judgment": SUPPORTS},
        ]

        summaries = summarize_claim_uncertainties(claims, judgments)
        summary = summaries[0]

        self.assertEqual(summary["support_count"], 2)
        self.assertEqual(summary["contradiction_count"], 1)
        self.assertEqual(summary["not_addressed_count"], 1)
        self.assertAlmostEqual(summary["semantic_uncertainty"], 0.375)
        self.assertAlmostEqual(summary["final_uncertainty"], 0.31375)
        self.assertAlmostEqual(summary["final_certainty"], 0.68625)
        self.assertIn("model_disagreement", summary["uncertainty_type"])
        self.assertIn("not_consistently_addressed", summary["uncertainty_type"])

    def test_summarizes_claim_without_judgments(self) -> None:
        claims = [
            {"claim_id": "c1", "text": "Paris is the capital of France.", "logit_uncertainty": None}
        ]

        summaries = summarize_claim_uncertainties(claims, [])
        summary = summaries[0]

        self.assertEqual(summary["judgment_count"], 0)
        self.assertIsNone(summary["semantic_uncertainty"])
        self.assertIsNone(summary["final_uncertainty"])
        self.assertEqual(summary["severity"], "unknown")

    def test_builds_claim_consistency_estimator(self) -> None:
        estimator = build_claim_consistency_estimator(
            [
                {"final_uncertainty": 0.2, "severity": "low"},
                {"final_uncertainty": 0.8, "severity": "high"},
            ]
        )

        self.assertEqual(estimator.name, "claim_semantic_consistency")
        self.assertGreater(estimator.uncertainty, 0.0)
        self.assertEqual(estimator.details["num_claims"], 2)
        self.assertEqual(estimator.details["num_high_claims"], 1)


if __name__ == "__main__":
    unittest.main()
