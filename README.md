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

## FEVER Benchmark Integration

This repository includes the FEVER dataset definition as a git submodule at `external/fever`.
Benchmark execution loads FEVER data from Hugging Face parquet conversion files (Hub), which is compatible with newer `datasets` versions.

Initialize submodules (optional, once per clone):

```bash
git submodule update --init --recursive
```

Run the FEVER uncertainty benchmark:

```bash
chmod +x run_fever_benchmark.sh
./run_fever_benchmark.sh
```

Useful overrides:

```bash
MODEL="FAST.gpt-oss:120b" FEVER_SPLIT="paper_dev" MAX_EXAMPLES=500 NUM_RUNS=3 ./run_fever_benchmark.sh
```

Outputs are saved under `experiments/fever_<config>_<split>_<timestamp>/`:

- `summary.csv`: per-example predictions + uncertainty
- `metrics.json`: informativeness/calibration metrics (AUROC, ECE, Spearman)
- `result_jsons/`: per-example raw analysis records
- `sentence_confidence/`: HTML confidence maps with original prompts
- PDF plots:
	- `uncertainty_by_label.pdf`
	- `uncertainty_correct_vs_incorrect.pdf`
	- `calibration_reliability.pdf`

## Run unit tests

I added unit tests for FEVER helper functions.
Run this from the project root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Label-Ambiguity Uncertainty Benchmark

This method asks the model for probabilities of the three FEVER labels and computes uncertainty as normalized entropy over those probabilities.

Quick pilot run (~1 minute depending on API latency):

```bash
QUICK_RUN=1 ./run_label_ambiguity_benchmark.sh
```

Full run (same scale as Elena's FEVER setup):

```bash
MAX_EXAMPLES=200 NUM_RUNS=3 ./run_label_ambiguity_benchmark.sh
```

