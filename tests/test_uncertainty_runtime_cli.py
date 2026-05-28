import json
import os
import shutil
import tempfile
import unittest

from uncertainty_runtime import save_payload


class TestUncertaintyRuntimeCLI(unittest.TestCase):
    def test_save_payload_writes_json_file(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="uncertainty_runtime_")
        try:
            payload = {
                "content": "Paris is the capital of France.",
                "claims": [{"claim_id": "c1", "text": "Paris is the capital of France."}],
            }
            path = save_payload(payload, output_dir=temp_dir, prefix="my run")

            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.basename(path).startswith("my_run_"))
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["content"], payload["content"])
            self.assertEqual(saved["claims"][0]["claim_id"], "c1")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
