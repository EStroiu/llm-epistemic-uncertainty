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
chmod +x scripts/run_fever_benchmark.sh
./scripts/run_fever_benchmark.sh
```

By default this uses a 256-token generation budget and a composite uncertainty score that combines claim fragility, REASON-line fragility, and a format penalty.

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

## Logit-Gap Experiments

The main uncertainty experiments live in `uncertainty_logit_gap.py` and are usually launched through `run_experiments.sh`.

Run the default token-level version:

```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

This keeps uncertainty scoring on tokens, but the confidence map is rendered at word-level by default for easier reading.

If you want to override the display granularity, set `DISPLAY_GRANULARITY`:

```bash
DISPLAY_GRANULARITY=token ./scripts/run_experiments.sh
```

You can still force word scoring if you want to compare against the older behavior:

```bash
GRANULARITY=word DISPLAY_GRANULARITY=word ./scripts/run_experiments.sh
```

The same split applies to FEVER:

```bash
chmod +x scripts/run_fever_benchmark.sh
./scripts/run_fever_benchmark.sh
```

Recommended hybrid mode for interpretability:

```bash
GRANULARITY=token DISPLAY_GRANULARITY=word ./scripts/run_fever_benchmark.sh
```

The dedicated `*_word.sh` wrappers now mean word-level visualization with token-level scoring under the hood.

If you want to try the simpler experimental scorer, use:

```bash
chmod +x scripts/run_fever_benchmark_simple.sh
./scripts/run_fever_benchmark_simple.sh
```

This variant keeps the same FEVER setup but uses a more interpretable max-based uncertainty score and reports filtered metrics as well.

