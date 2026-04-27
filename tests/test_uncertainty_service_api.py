import unittest

from fastapi.testclient import TestClient

from uncertainty_runtime import infer_uncertainty
from uncertainty_service_api import app


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
        self.assertIn("uncertainty", payload)
        self.assertIn("metadata", payload)
        self.assertIn("estimators", payload["uncertainty"])
        self.assertEqual(payload["metadata"]["dry_run"], True)

    def test_runtime_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            infer_uncertainty(query="  ", dry_run=True)

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
        self.assertIn("uncertainty", body)
        self.assertIn("metadata", body)

    def test_api_bad_request(self) -> None:
        client = TestClient(app)
        res = client.post("/infer", json={"query": "", "dry_run": True})
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()

