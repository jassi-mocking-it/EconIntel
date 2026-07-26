import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_DIRECTORY = Path(__file__).resolve().parents[1]

API_DATA_DIRECTORY = (
    APP_DIRECTORY
    / "data"
    / "api"
)

LATEST_RISK_SNAPSHOT_PATH = (
    API_DATA_DIRECTORY
    / "latest_risk_snapshot.json"
)


def make_json_safe(value: Any) -> Any:
    """
    Recursively convert pandas and NumPy values into
    standard Python values that JSON can safely store.
    """

    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(
        value,
        (
            np.floating,
            np.integer,
            np.bool_,
        ),
    ):
        return value.item()

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


def select_grouped_drivers(
    explanation,
    key,
    limit=4,
):
    """
    Extract the strongest grouped economic drivers from
    an explanation result.
    """

    drivers = explanation.get(
        key,
        [],
    )

    selected_drivers = []

    for driver in drivers[:limit]:
        selected_drivers.append(
            {
                "category": driver.get(
                    "group",
                    "Other Indicators",
                ),
                "contribution": float(
                    driver.get(
                        "shap_value",
                        0.0,
                    )
                ),
                "feature_count": int(
                    driver.get(
                        "feature_count",
                        0,
                    )
                ),
            }
        )

    return selected_drivers


def select_individual_drivers(
    explanation,
    key,
    limit=5,
):
    """
    Extract the strongest individual model features for
    technical and dashboard views.
    """

    drivers = explanation.get(
        key,
        [],
    )

    selected_drivers = []

    for driver in drivers[:limit]:
        raw_value = driver.get(
            "value"
        )

        selected_drivers.append(
            {
                "feature": driver.get(
                    "feature",
                    "Unknown Feature",
                ),
                "category": driver.get(
                    "group",
                    "Other Indicators",
                ),
                "value": make_json_safe(
                    raw_value
                ),
                "contribution": float(
                    driver.get(
                        "shap_value",
                        0.0,
                    )
                ),
            }
        )

    return selected_drivers


def validate_snapshot_inputs(
    latest_risk_assessment,
    latest_risk_explanation,
):
    """
    Confirm that the prediction and explanation represent
    the same observation.
    """

    assessment_date = str(
        latest_risk_assessment.get(
            "observation_date",
            "",
        )
    )

    explanation_date = str(
        latest_risk_explanation.get(
            "observation_date",
            "",
        )
    )

    if not assessment_date:
        raise ValueError(
            "The latest risk assessment does not contain "
            "an observation date."
        )

    if not explanation_date:
        raise ValueError(
            "The latest explanation does not contain "
            "an observation date."
        )

    if assessment_date != explanation_date:
        raise ValueError(
            "Prediction and explanation dates do not match: "
            f"{assessment_date} versus {explanation_date}."
        )


def create_production_risk_snapshot(
    latest_risk_assessment,
    latest_risk_explanation,
):
    """
    Combine the latest prediction and SHAP explanation into
    one stable, API-ready EconIntel response.

    The FastAPI backend will later serve this JSON without
    rerunning the full training pipeline.
    """

    print("\n" + "=" * 72)
    print("🌐 CREATING ECONINTEL PRODUCTION RISK SNAPSHOT")
    print("=" * 72)

    validate_snapshot_inputs(
        latest_risk_assessment,
        latest_risk_explanation,
    )

    probability = float(
        latest_risk_assessment[
            "risk_probability"
        ]
    )

    threshold = float(
        latest_risk_assessment[
            "warning_threshold"
        ]
    )

    snapshot = {
        "status": "success",

        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),

        "assessment": {
            "observation_date": (
                latest_risk_assessment[
                    "observation_date"
                ]
            ),
            "current_stress": float(
                latest_risk_assessment[
                    "current_stress"
                ]
            ),
            "risk_probability": probability,
            "risk_probability_percent": round(
                probability * 100,
                1,
            ),
            "warning_threshold": threshold,
            "warning_threshold_percent": round(
                threshold * 100,
                1,
            ),
            "warning_active": bool(
                latest_risk_assessment[
                    "warning_active"
                ]
            ),
            "warning_level": (
                latest_risk_assessment[
                    "warning_level"
                ]
            ),
            "forecast_horizon_months": int(
                latest_risk_assessment.get(
                    "forecast_horizon_months",
                    3,
                )
            ),
        },

        "drivers": {
            "categories_increasing_risk": (
                select_grouped_drivers(
                    latest_risk_explanation,
                    "grouped_increasing_risk",
                    limit=4,
                )
            ),
            "categories_reducing_risk": (
                select_grouped_drivers(
                    latest_risk_explanation,
                    "grouped_reducing_risk",
                    limit=4,
                )
            ),
            "features_increasing_risk": (
                select_individual_drivers(
                    latest_risk_explanation,
                    "increasing_risk",
                    limit=5,
                )
            ),
            "features_reducing_risk": (
                select_individual_drivers(
                    latest_risk_explanation,
                    "reducing_risk",
                    limit=5,
                )
            ),
        },

        "model": {
            "name": (
                latest_risk_assessment.get(
                    "model_name",
                    "Logistic Regression",
                )
            ),
            "penalty": "L2",
            "regularization_c": 1.0,
            "feature_count": 42,
            "target": (
                "Economic stress increases by at least "
                "five points within three months."
            ),
            "forecast_horizon_months": 3,
            "decision_threshold": threshold,
            "validation": {
                "method": (
                    "Five-fold chronological "
                    "walk-forward validation"
                ),
                "average_precision": 0.234,
                "average_recall": 0.843,
                "average_f1": 0.347,
                "average_pr_auc": 0.308,
                "average_roc_auc": 0.569,
            },
        },

        "interpretation": {
            "summary": (
                f"EconIntel estimates a "
                f"{probability * 100:.1f}% probability "
                "that economic stress will rise by at "
                "least five points within the next "
                "three months."
            ),
            "warning_message": (
                "The warning threshold has been reached."
                if probability >= threshold
                else
                "The warning threshold has not been reached."
            ),
        },

        "limitations": [
            (
                "This is an experimental early-warning "
                "model, not financial advice."
            ),
            (
                "The probability has not yet undergone "
                "formal probability calibration."
            ),
            (
                "SHAP values describe model contributions "
                "and do not prove economic causation."
            ),
            (
                "The current U.S. model produces a "
                "relatively high number of false alarms."
            ),
        ],
    }

    safe_snapshot = make_json_safe(
        snapshot
    )

    API_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LATEST_RISK_SNAPSHOT_PATH,
        "w",
        encoding="utf-8",
    ) as snapshot_file:
        json.dump(
            safe_snapshot,
            snapshot_file,
            indent=4,
        )

    print("✅ Production risk snapshot created.")

    print(
        f"Saved to: "
        f"{LATEST_RISK_SNAPSHOT_PATH}"
    )

    print(
        f"Observation date : "
        f"{snapshot['assessment']['observation_date']}"
    )

    print(
        f"Risk probability : "
        f"{snapshot['assessment']['risk_probability_percent']:.1f}%"
    )

    print(
        f"Warning level    : "
        f"{snapshot['assessment']['warning_level']}"
    )

    print(
        f"Warning active   : "
        f"{snapshot['assessment']['warning_active']}"
    )

    return safe_snapshot


def load_production_risk_snapshot():
    """
    Load the most recently generated API-ready snapshot.
    """

    if not LATEST_RISK_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            "No production risk snapshot exists at "
            f"{LATEST_RISK_SNAPSHOT_PATH}. "
            "Run the EconIntel pipeline first."
        )

    with open(
        LATEST_RISK_SNAPSHOT_PATH,
        "r",
        encoding="utf-8",
    ) as snapshot_file:
        return json.load(
            snapshot_file
        )