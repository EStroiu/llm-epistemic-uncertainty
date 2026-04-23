import unittest

from evals.fever_benchmark import (
    LABEL_NEI,
    LABEL_REFUTES,
    LABEL_SUPPORTS,
    _extract_predicted_label,
    _normalize_label,
    _sample_uncertainty,
)


class TestFeverHelpers(unittest.TestCase):
    def test_normalize_label_supported(self):
        self.assertEqual(_normalize_label("supports"), LABEL_SUPPORTS)

    def test_normalize_label_refuted(self):
        self.assertEqual(_normalize_label("Refutes"), LABEL_REFUTES)

    def test_normalize_label_nei(self):
        self.assertEqual(_normalize_label("not_enough_info"), LABEL_NEI)
        self.assertEqual(_normalize_label("NEI"), LABEL_NEI)

    def test_extract_predicted_label_from_first_line(self):
        answer = "LABEL: SUPPORTED\nREASON: evidence is present"
        self.assertEqual(_extract_predicted_label(answer), LABEL_SUPPORTS)

    def test_extract_predicted_label_when_unknown(self):
        answer = "I am not sure."
        self.assertEqual(_extract_predicted_label(answer), "UNKNOWN")

    def test_sample_uncertainty_prefers_claim_fragility(self):
        result = {
            "claim_spans": [
                {"fragility_score": 0.2},
                {"fragility_score": 0.4},
                {"fragility_score": None},
            ],
            "token_signals": [{"prob_gap": 0.9}],
        }
        self.assertAlmostEqual(_sample_uncertainty(result), 0.3)

    def test_sample_uncertainty_uses_token_gap_fallback(self):
        result = {
            "claim_spans": [],
            "token_signals": [{"prob_gap": 0.9}, {"prob_gap": 0.5}],
        }
        # mean gap is 0.7, so uncertainty should be 0.3
        self.assertAlmostEqual(_sample_uncertainty(result), 0.3)

    def test_sample_uncertainty_none_when_no_signals(self):
        result = {"claim_spans": [], "token_signals": []}
        self.assertIsNone(_sample_uncertainty(result))


if __name__ == "__main__":
    unittest.main()
