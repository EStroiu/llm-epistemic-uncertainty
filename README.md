# LLM Epistemic Uncertainty

This repository estimates uncertainty in LLM answers at the claim level. The main workflow is:

1. Prompt a model normally.
2. Collect the model's main answer plus alternative sampled answers.
3. Extract atomic factual claims from the main answer with an LLM-based extractor.
4. Score each claim with semantic consistency across sampled answers.
5. For the main answer, request Nebula token logprobs and attach logit-gap features to each extracted claim.
6. Return a structured payload with the answer, per-claim certainty/uncertainty, and aggregate uncertainty scores.

The system is designed for no-finetuning settings where we only have API access to an OpenAI-compatible endpoint such as Nebula.

## Requirements

- Python 3.9+
- Nebula API key in `NEBULA_API_KEY`

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the repo root:

```env
NEBULA_API_KEY=your_key_here
```

You can start from:

```powershell
Copy-Item .env.example .env
```

## Quick Smoke Test

Check Nebula access:

```powershell
python nebula_prompting.py
```

Run tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Free-Form Prompting

Use `uncertainty_runtime.py` when you want to prompt a model normally and inspect uncertainty for the claims it makes.

Dry run, no API call:

```powershell
python uncertainty_runtime.py --query "Give me three facts about Marie Curie." --mode freeform --n-samples 3 --nli-pairs 2 --dry-run
```

Live Nebula run:

```powershell
python uncertainty_runtime.py --query "Give me three facts about Marie Curie." --mode freeform --model FAST.gpt-oss:120b --n-samples 4 --nli-pairs 2 --top-logprobs 5
```

Print the full JSON payload:

```powershell
python uncertainty_runtime.py --query "Give me three facts about Marie Curie." --mode freeform --model FAST.gpt-oss:120b --n-samples 4 --nli-pairs 2 --top-logprobs 5 --json
```

Save the full JSON payload:

```powershell
python uncertainty_runtime.py --query "Give me three facts about Marie Curie." --mode freeform --model FAST.gpt-oss:120b --n-samples 4 --nli-pairs 2 --top-logprobs 5 --save
```

Saved files go to `outputs/` by default:

```text
outputs/freeform_YYYYMMDD_HHMMSS.json
```

Customize the output folder/name:

```powershell
python uncertainty_runtime.py --query "Give me three facts about Marie Curie." --mode freeform --save --output-dir outputs --output-prefix marie_curie_test
```

## Output Shape

The top-level response looks like:

```json
{
  "content": "The model answer...",
  "claims": [
    {
      "claim_id": "c1",
      "text": "Marie Curie won two Nobel Prizes.",
      "certainty": 0.82,
      "uncertainty": 0.18,
      "semantic_uncertainty": 0.25,
      "logit_uncertainty": 0.05,
      "support_count": 3,
      "contradiction_count": 0,
      "not_addressed_count": 1,
      "uncertainty_type": ["not_consistently_addressed"]
    }
  ],
  "uncertainty": {
    "overall_uncertainty": 0.31,
    "overall_confidence": 0.69,
    "reliability": "high_reliability",
    "estimators": []
  },
  "metadata": {}
}
```

Important per-claim fields:

- `semantic_uncertainty`: based on whether alternative sampled answers support, contradict, or do not address the claim.
- `logit_uncertainty`: based on token logprob gaps and top-k entropy for the claim span in the main answer.
- `final_uncertainty`: combined semantic + logit score.
- `certainty`: `1 - final_uncertainty`.
- `support_count`, `contradiction_count`, `not_addressed_count`: judge results across alternative answers.
- `uncertainty_type`: rough explanation tags such as `stable_across_samples`, `model_disagreement`, `not_consistently_addressed`, or `low_logit_margin`.

## Architecture

### `uncertainty_runtime.py`

Main runtime entry point. It supports two modes:

- `freeform`: prompt the model normally, extract claims from the model answer, score claim uncertainty.
- `claim_verification`: older FEVER-style mode where the input is treated as a claim to classify.

In free-form mode, it:

1. Asks Nebula for the main answer with `logprobs=True`.
2. Samples additional alternative answers.
3. Extracts atomic claims from the main answer with an LLM-based extractor.
4. Judges each claim against each alternative answer.
5. Combines semantic consistency with logit-gap uncertainty.

### `claim_extraction.py`

LLM-based atomic claim extraction.

The live extractor asks the model to return JSON:

```json
{"claims":[{"text":"..."}]}
```

It explicitly avoids punctuation-only splitting. This matters because one factual claim may contain dates, abbreviations, parentheticals, bullet points, or several punctuation marks.

Dry-run/tests use a mock extractor so tests do not require network access.

### `claim_judging.py`

Judges whether an alternative sampled answer:

- `SUPPORTS` a claim
- `CONTRADICTS` a claim
- `NOT_ADDRESSED` the claim

Live mode uses a Nebula judge prompt. Dry-run mode uses deterministic lexical heuristics.

### `claim_uncertainty.py`

Aggregates judgments into claim-level scores:

```text
semantic_uncertainty = contradiction_rate + 0.5 * not_addressed_rate
```

Then combines semantic and logit uncertainty:

```text
final_uncertainty =
  0.65 * semantic_uncertainty
  + 0.35 * logit_uncertainty
```

If logit scores are unavailable, it falls back to semantic uncertainty.

### `uncertainty_logit_gap.py`

Analyzes OpenAI-compatible chat responses with token logprobs. It computes:

- top-1/top-2 probability gap
- logprob gap
- top-k entropy
- claim-level logit fragility

The runtime uses this for the main free-form answer when Nebula returns token logprobs.

### `uncertainty_schema.py`

Defines the shared payload shape:

- `content`
- `claims`
- `uncertainty`
- `metadata`

### `uncertainty_service_api.py`

FastAPI wrapper around `infer_uncertainty`.

Start the service:

```powershell
uvicorn uncertainty_service_api:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /infer`
- `GET /dashboard`

Example PowerShell request:

```powershell
$body = @{
  query = "Give me three facts about Marie Curie."
  mode = "freeform"
  model = "FAST.gpt-oss:120b"
  variant = "numeric"
  n_samples = 4
  nli_pairs = 2
  top_logprobs = 5
  dry_run = $false
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/infer" -ContentType "application/json" -Body $body
```

Dashboard:

```text
http://127.0.0.1:8000/dashboard
```

## Benchmarking

`uncertainty_benchmark.py` runs the benchmark harness on FEVER-style JSONL data.

Dry run:

```powershell
python uncertainty_benchmark.py --dataset-path tests/data/tiny_fever.jsonl --max-examples 3 --n-samples 3 --nli-pairs 2 --variant numeric --dry-run
```

Live run:

```powershell
python uncertainty_benchmark.py --dataset-path tests/data/tiny_fever.jsonl --model FAST.gpt-oss:120b --max-examples 3 --n-samples 3 --nli-pairs 2 --variant numeric
```

Benchmark outputs go to:

```text
experiments/<run_id>/
```

Outputs include:

- `summary.csv`
- `metrics.json`
- `examples/example_*.json`

The benchmark now includes claim-level columns such as:

- `num_claims`
- `mean_claim_uncertainty`
- `max_claim_uncertainty`
- `min_claim_certainty`
- `total_claim_supports`
- `total_claim_contradictions`
- `total_claim_not_addressed`

## Notes And Limitations

- This system estimates uncertainty; it does not verify truth against external evidence.
- Multi-sample semantic consistency can miss cases where all samples repeat the same hallucination.
- Logit uncertainty is based on available Nebula/OpenAI-compatible top-logprobs, not raw model logits.
- The LLM claim extractor is better than punctuation splitting, but it can still miss or merge claims.
- Live free-form runs may use several API calls: main answer, alternative answers, claim extraction, claim judging, and NLI comparison.

## Useful Commands

Run all tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run only API/runtime tests:

```powershell
python -m unittest tests.test_uncertainty_service_api
```

Run a saved free-form experiment:

```powershell
python uncertainty_runtime.py --query "Give me three facts about CRISPR." --mode freeform --model FAST.gpt-oss:120b --n-samples 4 --nli-pairs 2 --top-logprobs 5 --save --output-prefix crispr
```
