from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from uncertainty_runtime import infer_uncertainty


app = FastAPI(
    title="LLM Uncertainty Service",
    description="Student A service endpoint for uncertainty-aware outputs.",
    version="0.1.0",
)


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Uncertainty Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f8fa; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    label { display: block; margin-top: 10px; font-weight: 600; }
    input, select, textarea { width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 6px; }
    textarea { min-height: 90px; }
    button { margin-top: 14px; padding: 10px 14px; border: none; border-radius: 8px; background: #1f6feb; color: white; cursor: pointer; }
    button:hover { background: #1559bd; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .big-score { font-size: 34px; font-weight: 700; color: #bc3800; }
    .muted { color: #666; font-size: 13px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111; color: #f8f8f2; padding: 10px; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>LLM Uncertainty Dashboard</h2>
  <p class="muted">This dashboard is an optional client. Canonical interface for integration is still POST /infer.</p>

  <div class="card">
    <label for="query" title="Claim or prompt to evaluate for factual support and uncertainty.">Claim / Query</label>
    <textarea id="query">Paris is the capital of France.</textarea>

    <div class="grid">
      <div>
        <label for="mode" title="Use freeform for normal prompts, claim_verification for FEVER-style labels.">Mode</label>
        <select id="mode">
          <option>freeform</option>
          <option>claim_verification</option>
        </select>
      </div>
      <div>
        <label for="variant" title="Prompt style for uncertainty expression (none, brief, numeric, calibrated).">Variant</label>
        <select id="variant">
          <option>numeric</option>
          <option>none</option>
          <option>brief</option>
          <option>calibrated</option>
        </select>
      </div>
      <div>
        <label for="model" title="Nebula model ID used for inference.">Model</label>
        <input id="model" value="FAST.gpt-oss:120b" />
      </div>
      <div>
        <label for="n_samples" title="Number of repeated model samples used to estimate uncertainty.">n_samples</label>
        <input id="n_samples" type="number" min="1" max="10" value="4" />
      </div>
      <div>
        <label for="nli_pairs" title="How many sample pairs are compared with NLI consistency checks.">nli_pairs</label>
        <input id="nli_pairs" type="number" min="0" max="20" value="2" />
      </div>
      <div>
        <label for="sample_temperature" title="Sampling randomness for generated answers (higher means more diverse outputs).">sample_temperature</label>
        <input id="sample_temperature" type="number" min="0" max="2" step="0.1" value="0.7" />
      </div>
      <div>
        <label for="prob_temperature" title="Randomness for label-probability estimation (usually keep at 0.0).">prob_temperature</label>
        <input id="prob_temperature" type="number" min="0" max="2" step="0.1" value="0.0" />
      </div>
      <div>
        <label for="top_logprobs" title="Number of token alternatives requested for logit-gap scoring.">top_logprobs</label>
        <input id="top_logprobs" type="number" min="1" max="20" value="5" />
      </div>
    </div>
    <label title="Uses mock data only; no real Nebula call and no API cost."><input id="dry_run" type="checkbox" checked /> dry_run (no Nebula call)</label>
    <button onclick="runInfer()">Run /infer</button>
  </div>

  <div class="card">
    <div class="muted">Overall uncertainty (main score)</div>
    <div id="unc_score" class="big-score">-</div>
    <div id="summary" class="muted">No run yet.</div>
  </div>

  <div class="card">
    <h3>Answer</h3>
    <div id="answer_text">-</div>
  </div>

  <div class="card">
    <h3>Estimator breakdown</h3>
    <pre id="estimators">-</pre>
  </div>

  <div class="card">
    <h3>Raw response JSON</h3>
    <pre id="raw_json">-</pre>
  </div>

  <script>
    async function runInfer() {
      const payload = {
        query: document.getElementById("query").value,
        model: document.getElementById("model").value,
        mode: document.getElementById("mode").value,
        variant: document.getElementById("variant").value,
        n_samples: Number(document.getElementById("n_samples").value),
        nli_pairs: Number(document.getElementById("nli_pairs").value),
        sample_temperature: Number(document.getElementById("sample_temperature").value),
        prob_temperature: Number(document.getElementById("prob_temperature").value),
        top_logprobs: Number(document.getElementById("top_logprobs").value),
        dry_run: document.getElementById("dry_run").checked
      };

      document.getElementById("summary").textContent = "Running...";
      try {
        const response = await fetch("/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const body = await response.json();
        if (!response.ok) {
          document.getElementById("summary").textContent = "Request failed";
          document.getElementById("raw_json").textContent = JSON.stringify(body, null, 2);
          return;
        }

        const u = body.uncertainty.overall_uncertainty;
        const c = body.uncertainty.overall_confidence;
        const rel = body.uncertainty.reliability;
        document.getElementById("unc_score").textContent = u.toFixed(3);
        document.getElementById("summary").textContent = `confidence=${c.toFixed(3)} | reliability=${rel}`;
        document.getElementById("answer_text").textContent = body.content || "-";
        document.getElementById("estimators").textContent = JSON.stringify(body.uncertainty.estimators, null, 2);
        document.getElementById("raw_json").textContent = JSON.stringify(body, null, 2);
      } catch (err) {
        document.getElementById("summary").textContent = "Request error";
        document.getElementById("raw_json").textContent = String(err);
      }
    }
  </script>
</body>
</html>
"""


class InferRequest(BaseModel):
    query: str = Field(..., description="Claim or user query to evaluate.")
    model: str = Field(default="FAST.gpt-oss:120b")
    mode: str = Field(default="claim_verification", description="claim_verification|freeform")
    variant: str = Field(default="numeric", description="none|brief|numeric|calibrated")
    n_samples: int = Field(default=4, ge=1, le=10)
    nli_pairs: int = Field(default=2, ge=0, le=20)
    sample_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    prob_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_logprobs: int = Field(default=5, ge=1, le=20)
    dry_run: bool = Field(default=False, description="True = do not call Nebula API")


class InferResponse(BaseModel):
    schema_version: str
    timestamp_utc: str
    content: str
    claims: List[Dict[str, Any]]
    uncertainty: Dict[str, Any]
    metadata: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> Dict[str, Any]:
    try:
        return infer_uncertainty(
            query=request.query,
            model=request.model,
            mode=request.mode,
            variant=request.variant,
            n_samples=request.n_samples,
            nli_pairs=request.nli_pairs,
            sample_temperature=request.sample_temperature,
            prob_temperature=request.prob_temperature,
            top_logprobs=request.top_logprobs,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

