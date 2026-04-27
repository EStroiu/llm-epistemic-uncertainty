from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from uncertainty_runtime import infer_uncertainty


app = FastAPI(
    title="LLM Uncertainty Service",
    description="Student A service endpoint for uncertainty-aware outputs.",
    version="0.1.0",
)


class InferRequest(BaseModel):
    query: str = Field(..., description="Claim or user query to evaluate.")
    model: str = Field(default="FAST.gpt-oss:120b")
    variant: str = Field(default="numeric", description="none|brief|numeric|calibrated")
    n_samples: int = Field(default=4, ge=1, le=10)
    nli_pairs: int = Field(default=2, ge=0, le=20)
    sample_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    prob_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    dry_run: bool = Field(default=False, description="True = do not call Nebula API")


class InferResponse(BaseModel):
    schema_version: str
    timestamp_utc: str
    content: str
    uncertainty: Dict[str, Any]
    metadata: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> Dict[str, Any]:
    try:
        return infer_uncertainty(
            query=request.query,
            model=request.model,
            variant=request.variant,
            n_samples=request.n_samples,
            nli_pairs=request.nli_pairs,
            sample_temperature=request.sample_temperature,
            prob_temperature=request.prob_temperature,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

