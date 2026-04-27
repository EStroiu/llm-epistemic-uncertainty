import unittest

from uncertainty_logit_gap import analyze_response_for_logit_gap_claims
from uncertainty_schema import EstimatorOutput, build_uncertainty_payload, combine_estimators


class TestUncertaintySchema(unittest.TestCase):
    def test_weighted_combination(self) -> None:
        estimators = [
            EstimatorOutput(name="a", uncertainty=0.2, confidence=0.8, weight=2.0),
            EstimatorOutput(name="b", uncertainty=0.8, confidence=0.2, weight=1.0),
        ]
        result = combine_estimators(estimators)
        self.assertAlmostEqual(result["overall_uncertainty"], 0.4, places=6)
        self.assertAlmostEqual(result["overall_confidence"], 0.6, places=6)
        self.assertEqual(result["reliability"], "high_reliability")

    def test_payload_shape(self) -> None:
        payload = build_uncertainty_payload(
            content="Some answer",
            estimators=[EstimatorOutput(name="demo", uncertainty=0.55, confidence=0.45)],
            prompt_variant="v1",
            expression="short_confidence",
            metadata={"sample_id": "x-1"},
        )
        self.assertIn("schema_version", payload)
        self.assertIn("uncertainty", payload)
        self.assertEqual(payload["content"], "Some answer")
        self.assertEqual(payload["uncertainty"]["prompt_variant"], "v1")
        self.assertEqual(payload["metadata"]["sample_id"], "x-1")

    def test_logit_gap_adapter_includes_unified_payload(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "Paris is the capital of France."},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Paris",
                                "logprob": -0.05,
                                "top_logprobs": [
                                    {"token": "Paris", "logprob": -0.05},
                                    {"token": "London", "logprob": -2.00},
                                ],
                            },
                            {
                                "token": " is",
                                "logprob": -0.10,
                                "top_logprobs": [
                                    {"token": " is", "logprob": -0.10},
                                    {"token": " was", "logprob": -1.80},
                                ],
                            },
                        ]
                    },
                }
            ]
        }
        result = analyze_response_for_logit_gap_claims(response, requested_top_logprobs=5)
        self.assertIn("uncertainty_payload", result)
        self.assertIn("uncertainty", result["uncertainty_payload"])
        self.assertEqual(
            result["uncertainty_payload"]["uncertainty"]["estimators"][0]["name"],
            "logit_gap_fragility",
        )


if __name__ == "__main__":
    unittest.main()

