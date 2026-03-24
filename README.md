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

