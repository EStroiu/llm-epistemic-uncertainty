import unittest

from uncertainty_logit_gap import analyze_response_for_logit_gap_claims


def _token(token: str, logprob: float, alt_token: str, alt_logprob: float):
    return {
        "token": token,
        "logprob": logprob,
        "top_logprobs": [
            {"token": token, "logprob": logprob},
            {"token": alt_token, "logprob": alt_logprob},
        ],
    }


class TestLogitAtomicClaims(unittest.TestCase):
    def test_atomic_claims_include_logit_metrics(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "Paris is the capital of France."},
                    "logprobs": {
                        "content": [
                            _token("Paris", -0.05, "London", -3.0),
                            _token(" is", -0.05, " was", -3.0),
                            _token(" the", -0.05, " a", -2.5),
                            _token(" capital", -0.05, " city", -2.5),
                            _token(" of", -0.05, " in", -2.5),
                            _token(" France", -0.20, " Germany", -0.21),
                            _token(".", -0.01, ",", -3.0),
                        ]
                    },
                }
            ]
        }

        result = analyze_response_for_logit_gap_claims(response)
        atomic_claims = result["atomic_claims"]

        self.assertEqual(len(atomic_claims), 1)
        claim = atomic_claims[0]
        self.assertEqual(claim["text"], "Paris is the capital of France.")
        self.assertEqual(claim["token_count"], 7)
        self.assertIsNotNone(claim["mean_prob_gap"])
        self.assertIsNotNone(claim["mean_entropy_topk"])
        self.assertIsNotNone(claim["logit_uncertainty"])
        self.assertIsNotNone(claim["logit_confidence"])
        self.assertEqual(claim["severity"], "high")

    def test_atomic_claims_exist_without_logprobs(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Marie Curie won two Nobel Prizes and was born in Warsaw."
                    }
                }
            ]
        }

        result = analyze_response_for_logit_gap_claims(response)
        atomic_claims = result["atomic_claims"]

        self.assertEqual([c["text"] for c in atomic_claims], [
            "Marie Curie won two Nobel Prizes.",
            "She was born in Warsaw.",
            "She conducted pioneering research on radioactivity.",
        ])
        self.assertEqual(atomic_claims[0]["severity"], "unknown")
        self.assertIsNone(atomic_claims[0]["logit_uncertainty"])
        self.assertTrue(result["provider_capabilities"]["fallback_recommended"])

    def test_uncertainty_payload_metadata_includes_atomic_claims(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "Paris is the capital of France."},
                    "logprobs": {"content": [_token("Paris is the capital of France.", -0.1, "London", -2.0)]},
                }
            ]
        }

        result = analyze_response_for_logit_gap_claims(response)
        metadata_claims = result["uncertainty_payload"]["metadata"]["atomic_claims"]

        self.assertEqual(metadata_claims[0]["text"], "Paris is the capital of France.")
        self.assertIn("logit_uncertainty", metadata_claims[0])


if __name__ == "__main__":
    unittest.main()
