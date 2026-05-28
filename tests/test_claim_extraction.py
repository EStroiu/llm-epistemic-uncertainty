import unittest

from claim_extraction import (
    AtomicClaim,
    claims_from_texts,
    mock_extract_atomic_claims,
    parse_claim_extraction_response,
)


class TestClaimExtraction(unittest.TestCase):
    def test_parses_llm_json_claims(self) -> None:
        answer = "Marie Curie won two Nobel Prizes. She was born in Warsaw."
        response = {
            "claims": [
                {"text": "Marie Curie won two Nobel Prizes."},
                {"text": "Marie Curie was born in Warsaw."},
            ]
        }

        claims = parse_claim_extraction_response(answer, str(response).replace("'", '"'))

        self.assertEqual([c.text for c in claims], [
            "Marie Curie won two Nobel Prizes.",
            "Marie Curie was born in Warsaw.",
        ])
        self.assertEqual([c.claim_id for c in claims], ["c1", "c2"])
        self.assertEqual(claims[0].extraction_method, "llm_claim_extraction")

    def test_does_not_split_at_every_dot_when_llm_groups_claim(self) -> None:
        answer = "Paris is the capital of France. It is located in Europe."
        response = '{"claims":[{"text":"Paris is the capital of France and is located in Europe."}]}'

        claims = parse_claim_extraction_response(answer, response)

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Paris is the capital of France and is located in Europe.")

    def test_claims_from_texts_cleans_bullets_and_deduplicates(self) -> None:
        claims = claims_from_texts(
            "1. Paris is the capital of France.\n2. Paris is the capital of France.",
            ["1. Paris is the capital of France", "- Paris is the capital of France."],
            method="llm_claim_extraction",
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].text, "Paris is the capital of France.")

    def test_mock_extractor_is_only_for_dry_run(self) -> None:
        claims = mock_extract_atomic_claims("Marie Curie won two Nobel Prizes.")

        self.assertGreaterEqual(len(claims), 1)
        self.assertEqual(claims[0].extraction_method, "mock_llm_claim_extraction")

    def test_atomic_claim_to_dict(self) -> None:
        claim = AtomicClaim(
            claim_id="c1",
            text="Paris is in France.",
            source_text="Paris is in France.",
            start=0,
            end=19,
            sentence_start=0,
            sentence_end=19,
            extraction_method="llm_claim_extraction",
        )

        payload = claim.to_dict()
        self.assertEqual(payload["claim_id"], "c1")
        self.assertEqual(payload["text"], "Paris is in France.")


if __name__ == "__main__":
    unittest.main()
