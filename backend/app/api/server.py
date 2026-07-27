import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from services.historical_snapshot_service import (
    load_us_stress_history_snapshot,
)
from services.production_inference_service import (
    load_production_risk_snapshot,
)


app = FastAPI(
    title="EconIntel API",
    description=(
        "API for EconIntel's explainable "
        "macroeconomic early-warning platform."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

# These origins allow the future local React dashboard
# to request data from the FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Snapshot-loading helpers
# ---------------------------------------------------------

def get_snapshot() -> dict[str, Any]:
    """
    Load the latest API-ready EconIntel risk snapshot.

    The API reads an already-generated JSON snapshot instead
    of retraining the model for every request.
    """

    try:
        snapshot = load_production_risk_snapshot()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "No EconIntel risk snapshot is available. "
                "Run the EconIntel pipeline first."
            ),
        ) from error

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The EconIntel risk snapshot contains "
                "invalid JSON."
            ),
        ) from error

    if not isinstance(snapshot, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                "The EconIntel risk snapshot has an "
                "invalid structure."
            ),
        )

    return snapshot


def get_stress_history() -> dict[str, Any]:
    """
    Load the chart-ready U.S. historical stress snapshot.
    """

    try:
        history = (
            load_us_stress_history_snapshot()
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "No U.S. stress-history snapshot is "
                "available. Run the EconIntel pipeline first."
            ),
        ) from error

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "The U.S. stress-history snapshot contains "
                "invalid JSON."
            ),
        ) from error

    if not isinstance(history, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                "The U.S. stress-history snapshot has an "
                "invalid structure."
            ),
        )

    return history


# ---------------------------------------------------------
# System endpoints
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, Any]:
    """
    Return basic information about the EconIntel API.
    """

    return {
        "application": "EconIntel API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "latest_risk": (
                "/api/v1/us/latest-risk"
            ),
            "assessment": (
                "/api/v1/us/assessment"
            ),
            "explanation": (
                "/api/v1/us/explanation"
            ),
            "model": (
                "/api/v1/us/model"
            ),
            "stress_history": (
                "/api/v1/us/stress-history"
            ),
            "crises": (
                "/api/v1/us/crises"
            ),
        },
    }


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, Any]:
    """
    Confirm that the API and latest prediction snapshot
    are available.
    """

    snapshot = get_snapshot()

    assessment = snapshot.get(
        "assessment",
        {},
    )

    return {
        "status": "healthy",
        "service": "EconIntel API",
        "checked_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "snapshot_available": True,
        "observation_date": assessment.get(
            "observation_date"
        ),
    }


# ---------------------------------------------------------
# U.S. latest-risk endpoints
# ---------------------------------------------------------

@app.get(
    "/api/v1/us/latest-risk",
    tags=["United States"],
)
def latest_us_risk() -> dict[str, Any]:
    """
    Return the complete latest U.S. risk snapshot.

    Includes:
    - current assessment;
    - economic drivers;
    - model information;
    - validation metrics;
    - limitations.
    """

    return get_snapshot()


@app.get(
    "/api/v1/us/assessment",
    tags=["United States"],
)
def latest_us_assessment() -> dict[str, Any]:
    """
    Return only the latest U.S. risk assessment and
    interpretation.
    """

    snapshot = get_snapshot()

    assessment = snapshot.get(
        "assessment"
    )

    if assessment is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The risk snapshot does not contain "
                "an assessment."
            ),
        )

    return {
        "status": snapshot.get(
            "status",
            "success",
        ),
        "generated_at_utc": snapshot.get(
            "generated_at_utc"
        ),
        "assessment": assessment,
        "interpretation": snapshot.get(
            "interpretation",
            {},
        ),
    }


@app.get(
    "/api/v1/us/explanation",
    tags=["United States"],
)
def latest_us_explanation() -> dict[str, Any]:
    """
    Return grouped and individual SHAP-based model drivers.
    """

    snapshot = get_snapshot()

    drivers = snapshot.get(
        "drivers"
    )

    if drivers is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The risk snapshot does not contain "
                "explanation drivers."
            ),
        )

    return {
        "status": snapshot.get(
            "status",
            "success",
        ),
        "observation_date": (
            snapshot
            .get("assessment", {})
            .get("observation_date")
        ),
        "drivers": drivers,
        "explanation_note": (
            "The contributions describe model behaviour "
            "and do not prove economic causation."
        ),
    }


@app.get(
    "/api/v1/us/model",
    tags=["United States"],
)
def us_model_information() -> dict[str, Any]:
    """
    Return the current U.S. model configuration,
    validation metrics, target, and limitations.
    """

    snapshot = get_snapshot()

    model_information = snapshot.get(
        "model"
    )

    if model_information is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "The risk snapshot does not contain "
                "model information."
            ),
        )

    return {
        "status": snapshot.get(
            "status",
            "success",
        ),
        "model": model_information,
        "limitations": snapshot.get(
            "limitations",
            [],
        ),
    }


# ---------------------------------------------------------
# U.S. historical endpoints
# ---------------------------------------------------------

@app.get(
    "/api/v1/us/stress-history",
    tags=["United States"],
)
def us_stress_history() -> dict[str, Any]:
    """
    Return chart-ready monthly U.S. stress and
    macroeconomic history.
    """

    return get_stress_history()


@app.get(
    "/api/v1/us/crises",
    tags=["United States"],
)
def us_historical_crises() -> dict[str, Any]:
    """
    Return historical crisis periods for dashboard
    chart overlays.
    """

    history = get_stress_history()

    return {
        "status": history.get(
            "status",
            "success",
        ),
        "country": (
            history
            .get("metadata", {})
            .get("country")
        ),
        "country_code": (
            history
            .get("metadata", {})
            .get("country_code")
        ),
        "crisis_periods": history.get(
            "crisis_periods",
            [],
        ),
        "note": (
            "These are historical labels, not future "
            "crisis predictions."
        ),
    }