import unittest

from claim_judging import (
    CONTRADICTS,
    NOT_ADDRESSED,
    SUPPORTS,
    mock_judge_claim_against_answer,
    parse_judge_response,
    parse_judgment_label,
)


class TestClaimJudging(unittest.TestCase):
    def test_parse_judgment_label(self) -> None:
        self.assertEqual(parse_judgment_label("supports"), SUPPORTS)
        self.assertEqual(parse_judgment_label("contradicts"), CONTRADICTS)
        self.assertEqual(parse_judgment_label("not addressed"), NOT_ADDRESSED)
        self.assertEqual(parse_judgment_label("neutral"), NOT_ADDRESSED)

    def test_parse_json_judge_response(self) -> None:
        parsed = parse_judge_response(
            '{"judgment":"CONTRADICTS","confidence":0.82,"reason":"The answer gives a different date."}'
        )

        self.assertEqual(parsed.judgment, CONTRADICTS)
        self.assertAlmostEqual(parsed.confidence, 0.82)
        self.assertEqual(parsed.reason, "The answer gives a different date.")

    def test_mock_judge_supports_overlap(self) -> None:
        judgment = mock_judge_claim_against_answer(
            "Paris is the capital of France.",
            "SUPPORTED. Paris is the capital of France. Confidence: 90%.",
        )

        self.assertEqual(judgment.judgment, SUPPORTS)
        self.assertEqual(judgment.source, "mock_judge")

    def test_mock_judge_detects_negation_contradiction(self) -> None:
        judgment = mock_judge_claim_against_answer(
            "Paris is the capital of France.",
            "Paris is not the capital of France.",
        )

        self.assertEqual(judgment.judgment, CONTRADICTS)

    def test_mock_judge_not_addressed_for_low_overlap(self) -> None:
        judgment = mock_judge_claim_against_answer(
            "Paris is the capital of France.",
            "The Pacific Ocean is large.",
        )

        self.assertEqual(judgment.judgment, NOT_ADDRESSED)


if __name__ == "__main__":
    unittest.main()
