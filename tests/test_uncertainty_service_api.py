import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from uncertainty_runtime import infer_uncertainty
from uncertainty_service_api import app


def _fake_token(token, logprob, alt_token, alt_logprob):
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[
            SimpleNamespace(token=token, logprob=logprob),
            SimpleNamespace(token=alt_token, logprob=alt_logprob),
        ],
    )


def _fake_response(content, logprob_content=None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=SimpleNamespace(content=logprob_content or []),
    )
    return SimpleNamespace(choices=[choice], usage=None)


class FakeChatCompletions:
    def __init__(self):
        self.freeform_calls = 0
        self.claim_extraction_calls = 0

    def create(self, **kwargs):
        if kwargs.get("logprobs"):
            return _fake_response(
                "Paris is the capital of France.",
                [
                    _fake_token("Paris", -0.05, "London", -2.0),
                    _fake_token(" is", -0.05, " was", -2.0),
                    _fake_token(" the", -0.05, " a", -2.0),
                    _fake_token(" capital", -0.20, " city", -0.21),
                    _fake_token(" of", -0.05, " in", -2.0),
                    _fake_token(" France", -0.05, " Germany", -2.0),
                    _fake_token(".", -0.01, ",", -2.0),
                ],
            )
        if kwargs.get("max_tokens") == 260:
            self.freeform_calls += 1
            return _fake_response("Paris is the capital of France.")
        messages = kwargs.get("messages", [])
        if messages and "extract atomic factual claims" in str(messages[-1].get("content", "")).lower():
            self.claim_extraction_calls += 1
            return _fake_response('{"claims":[{"text":"Paris is the capital of France."}]}')
        if kwargs.get("max_tokens") == 120:
            return _fake_response('{"judgment":"SUPPORTS","confidence":0.9,"reason":"same claim"}')
        return _fake_response("neutral")


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class TestUncertaintyServiceAPI(unittest.TestCase):
    def test_runtime_dry_run_payload(self) -> None:
        payload = infer_uncertainty(
            query="Paris is the capital of France.",
            variant="numeric",
            n_samples=3,
            nli_pairs=2,
            dry_run=True,
        )
        self.assertIn("content", payload)
        self.assertIn("claims", payload)
        self.assertIn("uncertainty", payload)
        self.assertIn("metadata", payload)
        self.assertIn("estimators", payload["uncertainty"])
        self.assertEqual(payload["metadata"]["dry_run"], True)
        self.assertEqual(payload["metadata"]["num_alternative_answers"], 2)
        self.assertEqual(payload["metadata"]["main_answer"]["role"], "main")
        self.assertEqual(len(payload["metadata"]["alternative_answers"]), 2)
        self.assertEqual(payload["metadata"]["alternative_answers"][0]["role"], "alternative")
        self.assertEqual(len(payload["metadata"]["answer_samples"]), 3)
        self.assertEqual(payload["content"], payload["metadata"]["main_answer"]["answer_text"])
        self.assertIn("atomic_claims", payload["metadata"])
        self.assertIn("claim_judgments", payload["metadata"])
        self.assertGreaterEqual(len(payload["metadata"]["atomic_claims"]), 1)
        self.assertEqual(
            len(payload["metadata"]["claim_judgments"]),
            len(payload["metadata"]["atomic_claims"]) * payload["metadata"]["num_alternative_answers"],
        )
        self.assertIn("judgment", payload["metadata"]["claim_judgments"][0])
        self.assertIn("claim_uncertainties", payload["metadata"])
        self.assertEqual(len(payload["metadata"]["claim_uncertainties"]), len(payload["metadata"]["atomic_claims"]))
        self.assertEqual(payload["claims"], payload["metadata"]["claim_uncertainties"])
        self.assertIn("semantic_uncertainty", payload["metadata"]["claim_uncertainties"][0])
        self.assertIn("final_uncertainty", payload["metadata"]["claim_uncertainties"][0])
        estimator_names = [e["name"] for e in payload["uncertainty"]["estimators"]]
        self.assertIn("claim_semantic_consistency", estimator_names)

    def test_runtime_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            infer_uncertainty(query="  ", dry_run=True)

    def test_runtime_single_sample_has_claims_but_no_judgments(self) -> None:
        payload = infer_uncertainty(
            query="Paris is the capital of France.",
            variant="numeric",
            n_samples=1,
            nli_pairs=0,
            dry_run=True,
        )

        self.assertEqual(payload["metadata"]["num_alternative_answers"], 0)
        self.assertEqual(payload["metadata"]["claim_judgments"], [])
        self.assertGreaterEqual(len(payload["claims"]), 1)
        self.assertIsNone(payload["claims"][0]["semantic_uncertainty"])
        self.assertIsNone(payload["claims"][0]["final_uncertainty"])

    def test_runtime_freeform_mode_answers_prompt_and_scores_claims(self) -> None:
        payload = infer_uncertainty(
            query="Give me three facts about Marie Curie.",
            mode="freeform",
            variant="numeric",
            n_samples=3,
            nli_pairs=2,
            dry_run=True,
        )

        self.assertEqual(payload["metadata"]["mode"], "freeform")
        self.assertIn("Marie Curie", payload["content"])
        self.assertGreaterEqual(len(payload["claims"]), 2)
        self.assertIn("final_uncertainty", payload["claims"][0])
        estimator_names = [e["name"] for e in payload["uncertainty"]["estimators"]]
        self.assertIn("claim_semantic_consistency", estimator_names)

    def test_runtime_freeform_live_path_uses_main_answer_logprobs(self) -> None:
        with patch("uncertainty_runtime.get_nebula_client", return_value=FakeClient()):
            payload = infer_uncertainty(
                query="Tell me about Paris.",
                mode="freeform",
                variant="numeric",
                n_samples=3,
                nli_pairs=0,
                dry_run=False,
                top_logprobs=5,
            )

        self.assertTrue(payload["metadata"]["token_logprobs_available"])
        self.assertIsNone(payload["metadata"]["logprob_error"])
        self.assertIn("logit_analysis", payload["metadata"])
        self.assertIsNotNone(payload["claims"][0]["logit_uncertainty"])
        self.assertIsNotNone(payload["claims"][0]["semantic_uncertainty"])
        self.assertIsNotNone(payload["claims"][0]["final_uncertainty"])

    def test_api_health_and_infer(self) -> None:
        client = TestClient(app)
        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        dashboard = client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn("LLM Uncertainty Dashboard", dashboard.text)

        res = client.post(
            "/infer",
            json={
                "query": "The Earth has two moons.",
                "mode": "freeform",
                "variant": "numeric",
                "n_samples": 3,
                "nli_pairs": 2,
                "dry_run": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("schema_version", body)
        self.assertIn("content", body)
        self.assertIn("claims", body)
        self.assertEqual(body["metadata"]["mode"], "freeform")
        self.assertIn("uncertainty", body)
        self.assertIn("metadata", body)

    def test_api_bad_request(self) -> None:
        client = TestClient(app)
        res = client.post("/infer", json={"query": "", "dry_run": True})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()

