# LLM epistemic uncertainty (FEVER-style)

Python utilities for claim verification with several uncertainty signals, a small benchmark harness, and a local HTTP service that returns structured scores for another component to consume.

## Requirements

Python 3.9+ and a Nebula API key (`NEBULA_API_KEY`).

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## API key

Create `.env` in the repo root (gitignored):

```env
NEBULA_API_KEY=your_key_here
```

You can start from `.env.example` if present.

## Nebula smoke script

```bash
python nebula_prompting.py
```

Loads models, sends one completion, prints the reply and usage.

## Uncertainty payload

`uncertainty_logit_gap.py` attaches an `uncertainty_payload` object: main text plus `overall_uncertainty`, `overall_confidence`, `reliability`, per-estimator entries, and optional metadata. Shape is defined in `uncertainty_schema.py`.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

API-only:

```bash
python -m unittest tests/test_uncertainty_service_api.py
```

## Benchmark

`uncertainty_benchmark.py` runs sampling variance, self-disagreement, and NLI consistency estimators, optional prompt variants (`none`, `brief`, `numeric`, `calibrated`), and writes metrics plus per-example logs under `experiments/` (gitignored by default).

Dry run — no network, uses tiny bundled data:

```bash
python uncertainty_benchmark.py --dataset-path tests/data/tiny_fever.jsonl --max-examples 3 --n-samples 3 --nli-pairs 2 --variant numeric --dry-run
```

Live run — point `--dataset-path` at your JSONL (see `scripts/build_fever_jsonl.py`). Runtime and cost scale with `max_examples`, `n_samples`, and `nli_pairs` because each example triggers multiple API calls.

```bash
python uncertainty_benchmark.py --dataset-path data/fever_labelled_dev_100.jsonl --model FAST.gpt-oss:120b --max-examples 5 --n-samples 3 --nli-pairs 2 --variant calibrated
```

Outputs include `metrics.json` (AUROC-style error detection, Spearman vs errors, ECE, clarity / interpretability proxies). To merge many runs into one table:

```bash
python aggregate_hyperparam_results.py --experiments-dir experiments --output-csv experiments/hyperparam_comparison.csv
```

Plotting helpers: `plot_hyperparam_results.py`, `plot_report_figures.py` (read paths inside those files).

## Local HTTP service

FastAPI app in `uncertainty_service_api.py`.

```bash
uvicorn uncertainty_service_api:app --host 127.0.0.1 --port 8000
```

- `GET /health` — liveness
- `POST /infer` — body: claim text, variant, sample counts, `dry_run`; response matches the shared payload shape from `uncertainty_runtime.infer_uncertainty`
- `GET /dashboard` — static page that calls `/infer` in the browser; useful for manual checks, not required for integration

Example `curl` (dry run, no Nebula):

```bash
curl -X POST "http://127.0.0.1:8000/infer" -H "Content-Type: application/json" -d "{\"query\":\"Paris is the capital of France.\",\"variant\":\"numeric\",\"n_samples\":3,\"nli_pairs\":2,\"dry_run\":true}"
```

PowerShell:

```powershell
Invoke-RestMethod -Method GET -Uri "http://127.0.0.1:8000/health"
$body = @{ query = "Paris is the capital of France."; variant = "numeric"; n_samples = 3; nli_pairs = 2; dry_run = $true } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/infer" -ContentType "application/json" -Body $body
```

Dashboard URL: `http://127.0.0.1:8000/dashboard`
