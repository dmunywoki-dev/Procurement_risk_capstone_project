#!/usr/bin/env python3
"""FastAPI service for prepared, point-in-time TGEE feature records.

The endpoint returns a relative review-priority percentile. It is not a fraud
probability, an allegation, or a substitute for an independent human review.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "artifacts" / "fy2025_2026"
    / "tgee_fy2025_2026_model_bundle.joblib"
)


class PreparedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: Optional[str] = None
    ocid: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    tabular: dict[str, Any]
    temporal: dict[str, Any]
    graph: dict[str, Any]
    text: str = ""
    hypergraph: Optional[dict[str, Any]] = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PreparedRecord] = Field(min_length=1, max_length=1000)


def _load_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}. Run the FY2025/2026 "
            "training pipeline first."
        )
    bundle = joblib.load(bundle_path)
    required = {
        "tabular_model", "tabular_feature_cols", "tabular_reference",
        "temporal_model", "temporal_feature_cols", "temporal_reference",
        "graph_model", "graph_feature_cols", "graph_reference",
        "weights", "review_threshold", "high_threshold",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")
    return bundle


def _percentile(reference: Any, value: float) -> float:
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(value):
        raise ValueError("The model score reference is empty or non-finite.")
    return float(np.searchsorted(np.sort(values), value, side="right") / values.size)


def _score_branch(model: Any, columns: list[str], values: dict[str, Any],
                  reference: Any, group_name: str) -> float:
    missing = [column for column in columns if column not in values]
    if missing:
        raise ValueError(f"Missing {group_name} features: {missing}")
    frame = pd.DataFrame([{column: values[column] for column in columns}],
                         columns=columns)
    raw = float(-model.score_samples(frame)[0])
    if not np.isfinite(raw):
        raise ValueError(f"{group_name} model returned a non-finite score.")
    return _percentile(reference, raw)


def score_prepared_record(bundle: dict[str, Any],
                          payload: dict[str, Any]) -> dict[str, Any]:
    """Score one dictionary matching the versioned prepared-feature contract."""
    required_groups = ("tabular", "temporal", "graph")
    for group in required_groups:
        if not isinstance(payload.get(group), dict):
            raise ValueError(f"'{group}' must be an object of prepared features.")

    components = {
        "tabular": _score_branch(
            bundle["tabular_model"], bundle["tabular_feature_cols"],
            payload["tabular"], bundle["tabular_reference"], "tabular"),
        "temporal": _score_branch(
            bundle["temporal_model"], bundle["temporal_feature_cols"],
            payload["temporal"], bundle["temporal_reference"], "temporal"),
        "graph": _score_branch(
            bundle["graph_model"], bundle["graph_feature_cols"],
            payload["graph"], bundle["graph_reference"], "graph"),
    }

    if bundle.get("text_active", False):
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise ValueError("'text' must be a string.")
        vector = bundle["text_vectorizer"].transform([text])
        vector = bundle["text_svd"].transform(vector)
        raw = float(-bundle["text_detector"].score_samples(vector)[0])
        components["text"] = _percentile(bundle["text_reference"], raw)

    if bundle.get("hypergraph_active", False):
        hypergraph = payload.get("hypergraph")
        if not isinstance(hypergraph, dict):
            raise ValueError("This model requires a hypergraph feature object.")
        components["hypergraph"] = _score_branch(
            bundle["hypergraph_model"], bundle["hypergraph_feature_cols"],
            hypergraph, bundle["hypergraph_reference"], "hypergraph")

    weights = bundle["weights"]
    available = {name: float(weight) for name, weight in weights.items()
                 if name in components and float(weight) > 0}
    denominator = sum(available.values())
    if denominator <= 0:
        raise ValueError("No active model branch has a positive fusion weight.")
    score = float(sum(components[name] * available[name]
                      for name in available) / denominator)

    if score >= float(bundle["high_threshold"]):
        band = "high_priority"
    elif score >= float(bundle["review_threshold"]):
        band = "review"
    else:
        band = "routine"

    return {
        "record_id": payload.get("record_id"),
        "ocid": payload.get("ocid"),
        "tgee_priority_score": round(score, 6),
        "review_band": band,
        "component_scores": {k: round(float(v), 6)
                             for k, v in components.items()},
        "score_semantics": (
            "Relative review-priority percentile; not a fraud probability "
            "or a finding of misconduct."
        ),
        "model_version": bundle.get("model_version", "unknown"),
        "context": payload.get("context", {}),
    }


def create_app(model_bundle: Optional[dict[str, Any]] = None,
               bundle_path: Optional[str | Path] = None) -> FastAPI:
    """Factory supports injected bundles for deterministic API tests."""
    path = Path(bundle_path or os.getenv("MODEL_BUNDLE_PATH", DEFAULT_BUNDLE))
    app = FastAPI(
        title="Kenya PPIP Procurement Review Prioritisation API",
        version="2.1.0",
        description=(
            "Prepared-feature scoring for human review. Scores are not "
            "fraud probabilities or legal findings."
        ),
    )
    app.state.model_bundle = model_bundle
    app.state.bundle_path = path

    def get_bundle() -> dict[str, Any]:
        if app.state.model_bundle is None:
            try:
                app.state.model_bundle = _load_bundle(app.state.bundle_path)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Unable to load model bundle ({type(exc).__name__}).",
                ) from exc
        return app.state.model_bundle

    @app.get("/health")
    def health() -> dict[str, Any]:
        bundle = get_bundle()
        return {
            "status": "ok",
            "model_version": bundle.get("model_version"),
            "data_mode": bundle.get("data_mode"),
            "score_semantics": bundle.get("score_semantics"),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        bundle = get_bundle()
        return {
            "project_name": bundle.get("project_name"),
            "model_version": bundle.get("model_version"),
            "analysis_period": bundle.get("analysis_period"),
            "active_branches": list(bundle.get("weights", {}).keys()),
            "weights": bundle.get("weights"),
            "review_threshold": bundle.get("review_threshold"),
            "high_threshold": bundle.get("high_threshold"),
            "hypergraph_active": bundle.get("hypergraph_active", False),
            "score_semantics": bundle.get("score_semantics"),
        }

    @app.post("/score")
    def score_endpoint(record: PreparedRecord) -> dict[str, Any]:
        try:
            return score_prepared_record(get_bundle(), record.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Scoring failed ({type(exc).__name__}).",
            ) from exc

    @app.post("/score/batch")
    def score_batch(request: BatchRequest) -> dict[str, Any]:
        try:
            bundle = get_bundle()
            results = [
                score_prepared_record(bundle, record.model_dump())
                for record in request.records
            ]
            return {"count": len(results), "results": results}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Batch scoring failed ({type(exc).__name__}).",
            ) from exc

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_service:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
