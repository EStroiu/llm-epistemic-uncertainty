## Prerequisites

- Python 3.9+ (recommended)
- A Nebula API key

## Setup (venv + requirements)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure your API key (`.env`)

This project reads the API key from an environment variable named `NEBULA_API_KEY`.

1. Create a file named `.env` in the project root (or edit the existing one).
2. Add your key:

```env
NEBULA_API_KEY=YOUR_KEY_HERE
```

Notes:

- `.env` should stay local and **must not be committed**. It is already ignored via `.gitignore`.
- If you want a template, copy `.env.example` to `.env` and fill in your key.

## Run the demo script

```bash
python nebula_prompting.py
```

What it does:

- Lists available models from the Nebula API
- Sends a simple chat completion request
- Prints the model response and usage stats

## Unified uncertainty payload

`uncertainty_logit_gap.py` now also returns a standardized `uncertainty_payload` block with:

- `content`
- `uncertainty.overall_uncertainty`
- `uncertainty.overall_confidence`
- `uncertainty.reliability`
- `uncertainty.estimators[]`
- lightweight metadata for downstream consumers (for example embodied agents)

## Quick tests

Run a short smoke test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Multi-estimator benchmark (points 2-4)

`uncertainty_benchmark.py` runs one unified pipeline with:

- sampling variance estimator
- self-disagreement estimator
- NLI consistency estimator
- prompt variants (`none`, `brief`, `numeric`, `calibrated`)
- uncertainty + clarity/interpretability metrics

### 1) Very fast smoke benchmark (no API calls)

```bash
python uncertainty_benchmark.py \
  --dataset-path tests/data/tiny_fever.jsonl \
  --max-examples 3 \
  --n-samples 3 \
  --nli-pairs 2 \
  --variant numeric \
  --dry-run
```

### 2) Short live run (about 1-2 minutes depending on model queue)

```bash
python uncertainty_benchmark.py \
  --dataset-path path/to/fever_subset.jsonl \
  --model FAST.gpt-oss:120b \
  --max-examples 5 \
  --n-samples 3 \
  --nli-pairs 2 \
  --variant calibrated
```

### 3) Final variant comparison

Run 4 short jobs with variants `none`, `brief`, `numeric`, `calibrated` and compare:

- `metrics.json` (`error_detection_auroc`, `nei_detection_auroc`, `spearman_uncertainty_error`, `ece_confidence`)
- `avg_clarity_score`
- `avg_interpretability_score`

## Local API for Student B handoff

This project now includes a minimal local API service:

- `GET /health`
- `POST /infer` -> returns `content + uncertainty metadata` payload

Start the server:

```bash
uvicorn uncertainty_service_api:app --host 127.0.0.1 --port 8000
```

Quick dry-run request (no Nebula call):

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Paris is the capital of France.\",\"variant\":\"numeric\",\"n_samples\":3,\"nli_pairs\":2,\"dry_run\":true}"
```

## API smoke tests

```bash
python -m unittest tests/test_uncertainty_service_api.py
```

