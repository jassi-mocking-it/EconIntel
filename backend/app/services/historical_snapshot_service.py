import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.production_inference_service import (
    make_json_safe,
)


APP_DIRECTORY = Path(__file__).resolve().parents[1]

API_DATA_DIRECTORY = (
    APP_DIRECTORY
    / "data"
    / "api"
)

US_STRESS_HISTORY_PATH = (
    API_DATA_DIRECTORY
    / "us_stress_history.json"
)


def numeric_or_none(value: Any):
    """
    Convert a numerical value into a normal Python float.

    Missing, infinite, or invalid values become None so the
    resulting output can be safely written as JSON.
    """

    if value is None or value is pd.NA:
        return None

    try:
        converted_value = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(converted_value):
        return None

    if np.isinf(converted_value):
        return None

    return converted_value


def text_or_none(value: Any):
    """
    Convert a text value into a clean string.
    """

    if value is None or value is pd.NA:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def crisis_flag(value: Any) -> bool:
    """
    Convert the historical crisis label into a Boolean.
    """

    if value is None or value is pd.NA:
        return False

    if pd.isna(value):
        return False

    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def build_crisis_periods(data: pd.DataFrame):
    """
    Combine consecutive crisis-labelled months into periods
    that can be displayed as shaded chart regions.
    """

    if "CRISIS" not in data.columns:
        return []

    crisis_periods = []

    active_period = None

    for _, row in data.iterrows():
        row_date = pd.to_datetime(
            row["date"]
        )

        is_crisis = crisis_flag(
            row.get("CRISIS", 0)
        )

        crisis_name = (
            text_or_none(
                row.get("CRISIS_NAME")
            )
            or "Historical Crisis"
        )

        if is_crisis:
            if active_period is None:
                active_period = {
                    "name": crisis_name,
                    "start_date": str(
                        row_date.date()
                    ),
                    "end_date": str(
                        row_date.date()
                    ),
                }

            elif active_period["name"] == crisis_name:
                active_period["end_date"] = str(
                    row_date.date()
                )

            else:
                crisis_periods.append(
                    active_period
                )

                active_period = {
                    "name": crisis_name,
                    "start_date": str(
                        row_date.date()
                    ),
                    "end_date": str(
                        row_date.date()
                    ),
                }

        elif active_period is not None:
            crisis_periods.append(
                active_period
            )

            active_period = None

    if active_period is not None:
        crisis_periods.append(
            active_period
        )

    return crisis_periods


def create_history_record(row: pd.Series):
    """
    Convert one monthly observation into a chart-ready
    API record.
    """

    return {
        "date": str(
            pd.to_datetime(
                row["date"]
            ).date()
        ),

        "economic_stress": numeric_or_none(
            row.get("ECON_STRESS")
        ),

        "stress_change_1m": numeric_or_none(
            row.get("STRESS_CHANGE_1M")
        ),

        "stress_change_3m": numeric_or_none(
            row.get("STRESS_CHANGE_3M")
        ),

        "stress_volatility_6m": numeric_or_none(
            row.get("STRESS_VOLATILITY_6M")
        ),

        "fed_funds_rate": numeric_or_none(
            row.get("FEDFUNDS")
        ),

        "unemployment_rate": numeric_or_none(
            row.get("UNRATE")
        ),

        "inflation_rate": numeric_or_none(
            row.get("INFLATION_RATE")
        ),

        "gdp_growth": numeric_or_none(
            row.get("GDP_GROWTH")
        ),

        "treasury_yield": numeric_or_none(
            row.get("TREASURY")
        ),

        "vix": numeric_or_none(
            row.get("VIX")
        ),

        "crisis": crisis_flag(
            row.get("CRISIS", 0)
        ),

        "crisis_name": text_or_none(
            row.get("CRISIS_NAME")
        ),
    }


def create_us_stress_history_snapshot(
    full_feature_df: pd.DataFrame,
):
    """
    Create the chart-ready historical U.S. stress snapshot.

    This file is read by FastAPI and later by the React
    dashboard. It includes monthly stress values, selected
    economic indicators, and historical crisis overlays.
    """

    print("\n" + "=" * 72)
    print("📈 CREATING U.S. STRESS-HISTORY SNAPSHOT")
    print("=" * 72)

    required_columns = [
        "date",
        "ECON_STRESS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in full_feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Cannot create stress history because these "
            "columns are missing: "
            + ", ".join(missing_columns)
        )

    data = full_feature_df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .dropna(
            subset=["ECON_STRESS"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            "No valid economic-stress observations exist."
        )

    history_records = [
        create_history_record(row)
        for _, row in data.iterrows()
    ]

    crisis_periods = build_crisis_periods(
        data
    )

    stress_series = pd.to_numeric(
        data["ECON_STRESS"],
        errors="coerce",
    )

    peak_index = stress_series.idxmax()

    latest_row = data.iloc[-1]

    snapshot = {
        "status": "success",

        "generated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),

        "metadata": {
            "country": "United States",
            "country_code": "USA",
            "frequency": "monthly",
            "start_date": str(
                data["date"].min().date()
            ),
            "end_date": str(
                data["date"].max().date()
            ),
            "observation_count": int(
                len(data)
            ),
        },

        "summary": {
            "latest_stress": numeric_or_none(
                latest_row["ECON_STRESS"]
            ),
            "average_stress": numeric_or_none(
                stress_series.mean()
            ),
            "minimum_stress": numeric_or_none(
                stress_series.min()
            ),
            "maximum_stress": numeric_or_none(
                stress_series.max()
            ),
            "peak_stress_date": str(
                data.loc[
                    peak_index,
                    "date",
                ].date()
            ),
        },

        "crisis_periods": crisis_periods,

        "series": history_records,

        "notes": [
            (
                "Economic stress is displayed on a "
                "zero-to-one-hundred index."
            ),
            (
                "Crisis overlays are historical labels and "
                "are not model predictions."
            ),
            (
                "Missing indicator values are represented "
                "as null."
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
        US_STRESS_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(
            safe_snapshot,
            history_file,
            indent=4,
        )

    print("✅ U.S. stress-history snapshot created.")

    print(
        f"Saved to: "
        f"{US_STRESS_HISTORY_PATH}"
    )

    print(
        f"Historical observations: "
        f"{len(history_records)}"
    )

    print(
        f"Date range: "
        f"{snapshot['metadata']['start_date']} "
        f"to {snapshot['metadata']['end_date']}"
    )

    print(
        f"Crisis periods: "
        f"{len(crisis_periods)}"
    )

    print(
        f"Peak stress: "
        f"{snapshot['summary']['maximum_stress']:.2f}"
    )

    print(
        f"Peak date: "
        f"{snapshot['summary']['peak_stress_date']}"
    )

    return safe_snapshot


def load_us_stress_history_snapshot():
    """
    Load the API-ready U.S. stress-history snapshot.
    """

    if not US_STRESS_HISTORY_PATH.exists():
        raise FileNotFoundError(
            "No U.S. stress-history snapshot exists at "
            f"{US_STRESS_HISTORY_PATH}. "
            "Run the EconIntel pipeline first."
        )

    with open(
        US_STRESS_HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as history_file:
        return json.load(
            history_file
        )