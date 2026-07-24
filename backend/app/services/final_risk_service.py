import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "TARGET_RISK_RISING_3M"

EXCLUDED_COLUMNS = [
    "date",

    # Future targets
    "TARGET_STRESS_3M",
    "TARGET_STRESS_CHANGE_3M",
    "TARGET_RISK_RISING_3M",
    "TARGET_CRISIS_3M",

    # Historical labels
    "CRISIS",
    "CRISIS_NAME",

    # Internal index value
    "RAW_ECON_STRESS",
]


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "final_risk_model.pkl"
)

METADATA_PATH = (
    MODEL_DIRECTORY
    / "final_risk_model_metadata.json"
)


def build_final_logistic_model():
    """
    Build EconIntel's selected early-warning classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def get_warning_level(
    probability,
    warning_threshold,
):
    """
    Convert the model probability into a readable
    dashboard warning level.

    These bands are presentation categories.
    The actual model alert begins at warning_threshold.
    """

    if probability < 0.20:
        return "Low"

    if probability < warning_threshold:
        return "Guarded"

    if probability < 0.60:
        return "Elevated"

    if probability < 0.80:
        return "High"

    return "Critical"


def prepare_training_data(df):
    """
    Extract chronologically ordered labelled observations.
    """

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .dropna(subset=[TARGET_COLUMN])
        .sort_values("date")
        .reset_index(drop=True)
    )

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype(int)
    )

    X = data.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    # Keep only numerical model inputs.
    X = X.select_dtypes(
        include=["number"]
    )

    y = data[TARGET_COLUMN]

    return data, X, y


def train_and_save_final_risk_model(
    training_df,
    full_feature_df,
    warning_threshold,
):
    """
    Train the selected Logistic Regression model on all
    labelled history, save it, and predict the latest row.
    """

    print("\n" + "=" * 72)
    print("💾 TRAINING FINAL ECONINTEL EARLY-WARNING MODEL")
    print("=" * 72)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    labelled_data, X_train, y_train = (
        prepare_training_data(
            training_df
        )
    )

    model = build_final_logistic_model()

    model.fit(
        X_train,
        y_train,
    )

    model_bundle = {
        "model": model,
        "feature_names": list(
            X_train.columns
        ),
        "warning_threshold": float(
            warning_threshold
        ),
        "target_column": TARGET_COLUMN,
        "forecast_horizon_months": 3,
        "model_name": "Logistic Regression",
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    metadata = {
        "model_name": (
            "Logistic Regression"
        ),
        "target": TARGET_COLUMN,
        "forecast_horizon_months": 3,
        "warning_threshold": float(
            warning_threshold
        ),
        "training_rows": int(
            len(X_train)
        ),
        "positive_training_cases": int(
            y_train.sum()
        ),
        "negative_training_cases": int(
            len(y_train) - y_train.sum()
        ),
        "training_start": str(
            labelled_data["date"]
            .min()
            .date()
        ),
        "training_end": str(
            labelled_data["date"]
            .max()
            .date()
        ),
        "feature_count": int(
            len(X_train.columns)
        ),
        "feature_names": list(
            X_train.columns
        ),
        "created_at_utc": (
            datetime.utcnow()
            .isoformat()
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    latest_prediction = (
        predict_latest_risk(
            full_feature_df,
            model_bundle=model_bundle,
        )
    )

    print("✅ Final risk model trained and saved.")

    print(
        f"Model path: {MODEL_PATH}"
    )

    print(
        f"Metadata path: {METADATA_PATH}"
    )

    print(
        f"Training observations: "
        f"{len(X_train)}"
    )

    print(
        f"Features used: "
        f"{len(X_train.columns)}"
    )

    print("\nLatest EconIntel risk assessment")
    print("--------------------------------")

    print(
        f"Observation date : "
        f"{latest_prediction['observation_date']}"
    )

    print(
        f"Current stress   : "
        f"{latest_prediction['current_stress']:.2f}"
    )

    print(
        f"Risk probability : "
        f"{latest_prediction['risk_probability']:.1%}"
    )

    print(
        f"Alert threshold  : "
        f"{latest_prediction['warning_threshold']:.2f}"
    )

    print(
        f"Warning active   : "
        f"{latest_prediction['warning_active']}"
    )

    print(
        f"Warning level    : "
        f"{latest_prediction['warning_level']}"
    )

    print(
        "Forecast horizon : 3 months"
    )

    return (
        model_bundle,
        latest_prediction,
    )


def predict_latest_risk(
    full_feature_df,
    model_bundle=None,
):
    """
    Generate a three-month rising-stress prediction for
    the latest available macroeconomic observation.
    """

    if model_bundle is None:
        model_bundle = joblib.load(
            MODEL_PATH
        )

    model = model_bundle["model"]

    feature_names = model_bundle[
        "feature_names"
    ]

    warning_threshold = float(
        model_bundle[
            "warning_threshold"
        ]
    )

    data = full_feature_df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    latest_row = data.iloc[-1]

    latest_features = pd.DataFrame(
        [
            {
                feature: latest_row.get(
                    feature,
                    pd.NA,
                )
                for feature in feature_names
            }
        ]
    )

    for feature in feature_names:
        latest_features[feature] = (
            pd.to_numeric(
                latest_features[feature],
                errors="coerce",
            )
        )

    probability = float(
        model.predict_proba(
            latest_features
        )[0, 1]
    )

    warning_active = bool(
        probability >= warning_threshold
    )

    warning_level = get_warning_level(
        probability=probability,
        warning_threshold=warning_threshold,
    )

    current_stress = float(
        latest_row.get(
            "ECON_STRESS",
            0.0,
        )
    )

    return {
        "observation_date": str(
            latest_row["date"].date()
        ),
        "current_stress": (
            current_stress
        ),
        "risk_probability": (
            probability
        ),
        "warning_threshold": (
            warning_threshold
        ),
        "warning_active": (
            warning_active
        ),
        "warning_level": (
            warning_level
        ),
        "forecast_horizon_months": 3,
        "model_name": (
            model_bundle["model_name"]
        ),
    }