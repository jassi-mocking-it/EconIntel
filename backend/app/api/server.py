import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


# Allows the future local React dashboard to access the API.
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


def get_snapshot() -> dict[str, Any]:
    """
    Load the latest API-ready EconIntel risk snapshot.

    The API reads an already-generated snapshot instead of
    training the model whenever a user opens the dashboard.
    """

    try:
        snapshot = load_production_risk_snapshot()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "No EconIntel risk snapshot is available. "
                "Run the data and inference pipeline first."
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


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, Any]:
    """
    Return basic information about the API.
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
        },
    }


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, Any]:
    """
    Confirm that the API and prediction snapshot are
    available.
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


@app.get(
    "/api/v1/us/latest-risk",
    tags=["United States"],
)
def latest_us_risk() -> dict[str, Any]:
    """
    Return the complete U.S. risk snapshot.

    This includes the assessment, economic drivers,
    validation metrics, interpretation, and limitations.
    """

    return get_snapshot()


@app.get(
    "/api/v1/us/assessment",
    tags=["United States"],
)
def latest_us_assessment() -> dict[str, Any]:
    """
    Return only the latest U.S. risk assessment.
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
    Return grouped and individual model drivers.
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
    Return model configuration, validation metrics,
    target definition, and limitations.
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