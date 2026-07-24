import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from config.model_features import FEATURE_GROUPS


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "final_risk_model.pkl"
)

EXPLANATION_PATH = (
    MODEL_DIRECTORY
    / "latest_risk_explanation.json"
)


def find_feature_group(feature_name):
    """
    Map an individual model feature to a readable
    economic category.
    """

    for group_name, group_features in FEATURE_GROUPS.items():
        if feature_name in group_features:
            return group_name

    return "Other Indicators"


def dataframe_records_to_json(df):
    """
    Convert selected DataFrame rows into JSON-safe records.
    """

    safe_df = df.copy()

    safe_df = safe_df.replace(
        {
            np.nan: None,
            np.inf: None,
            -np.inf: None,
        }
    )

    return safe_df.to_dict(
        orient="records"
    )


def explain_latest_risk(
    full_feature_df,
    model_bundle=None,
    top_n=8,
):
    """
    Explain EconIntel's latest three-month rising-risk
    probability.

    Positive SHAP values increase the model's predicted risk.
    Negative SHAP values reduce the model's predicted risk.

    The function produces:
    1. Individual feature contributions.
    2. Grouped economic-category contributions.
    3. A JSON file for the future dashboard and API.
    """

    print("\n" + "=" * 72)
    print("🔎 ECONINTEL LATEST-RISK EXPLAINABILITY")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load the saved model bundle if one was not supplied
    # ---------------------------------------------------------

    if model_bundle is None:
        model_bundle = joblib.load(
            MODEL_PATH
        )

    pipeline = model_bundle["model"]

    feature_names = model_bundle[
        "feature_names"
    ]

    warning_threshold = float(
        model_bundle[
            "warning_threshold"
        ]
    )

    # ---------------------------------------------------------
    # Extract fitted pipeline components
    # ---------------------------------------------------------

    imputer = pipeline.named_steps[
        "imputer"
    ]

    scaler = pipeline.named_steps[
        "scaler"
    ]

    logistic_model = pipeline.named_steps[
        "model"
    ]

    # ---------------------------------------------------------
    # Prepare full feature history
    # ---------------------------------------------------------

    data = full_feature_df.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    feature_data = data.reindex(
        columns=feature_names
    ).copy()

    for feature in feature_names:
        feature_data[feature] = pd.to_numeric(
            feature_data[feature],
            errors="coerce",
        )

    if len(feature_data) < 2:
        raise ValueError(
            "Explainability requires at least two "
            "historical observations."
        )

    # ---------------------------------------------------------
    # Separate historical background from latest observation
    # ---------------------------------------------------------

    background_raw = (
        feature_data.iloc[:-1]
        .copy()
    )

    latest_raw = (
        feature_data.iloc[[-1]]
        .copy()
    )

    # ---------------------------------------------------------
    # Apply the same transformations used during training
    # ---------------------------------------------------------

    background_imputed = imputer.transform(
        background_raw
    )

    latest_imputed = imputer.transform(
        latest_raw
    )

    background_scaled = scaler.transform(
        background_imputed
    )

    latest_scaled = scaler.transform(
        latest_imputed
    )

    # Keep the most recent 200 observations as background.
    # SHAP may internally subsample this further.
    if len(background_scaled) > 200:
        background_scaled = (
            background_scaled[-200:]
        )

    # ---------------------------------------------------------
    # Create SHAP explanation
    # ---------------------------------------------------------

    explainer = shap.LinearExplainer(
        logistic_model,
        background_scaled,
        feature_names=feature_names,
    )

    explanation = explainer(
        latest_scaled
    )

    shap_values = np.asarray(
        explanation.values
    ).reshape(-1)

    raw_feature_values = (
        latest_raw.iloc[0]
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Build individual feature contribution table
    # ---------------------------------------------------------

    contributions = pd.DataFrame(
        {
            "feature": feature_names,
            "value": raw_feature_values,
            "shap_value": shap_values,
        }
    )

    contributions["absolute_shap"] = (
        contributions["shap_value"]
        .abs()
    )

    contributions["group"] = (
        contributions["feature"]
        .apply(find_feature_group)
    )

    contributions = (
        contributions
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_contributions = (
        contributions
        .head(top_n)
        .copy()
    )

    increasing_risk = (
        contributions[
            contributions["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    reducing_risk = (
        contributions[
            contributions["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .head(top_n)
        .copy()
    )

    # ---------------------------------------------------------
    # Aggregate features into economic categories
    # ---------------------------------------------------------

    grouped_contributions = (
        contributions
        .groupby(
            "group",
            as_index=False,
        )
        .agg(
            shap_value=(
                "shap_value",
                "sum",
            ),
            absolute_shap=(
                "shap_value",
                lambda values: (
                    values.abs().sum()
                ),
            ),
            feature_count=(
                "feature",
                "count",
            ),
        )
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    grouped_increasing_risk = (
        grouped_contributions[
            grouped_contributions[
                "shap_value"
            ] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    grouped_reducing_risk = (
        grouped_contributions[
            grouped_contributions[
                "shap_value"
            ] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Recalculate latest model probability
    # ---------------------------------------------------------

    probability = float(
        pipeline.predict_proba(
            latest_raw
        )[0, 1]
    )

    warning_active = bool(
        probability >= warning_threshold
    )

    latest_date = str(
        data.iloc[-1]["date"].date()
    )

    # ---------------------------------------------------------
    # Build API-ready result
    # ---------------------------------------------------------

    result = {
        "observation_date": latest_date,
        "model_name": model_bundle.get(
            "model_name",
            "Logistic Regression",
        ),
        "forecast_horizon_months": int(
            model_bundle.get(
                "forecast_horizon_months",
                3,
            )
        ),
        "risk_probability": probability,
        "warning_threshold": warning_threshold,
        "warning_active": warning_active,

        "top_contributions": (
            dataframe_records_to_json(
                top_contributions[
                    [
                        "feature",
                        "group",
                        "value",
                        "shap_value",
                        "absolute_shap",
                    ]
                ]
            )
        ),

        "increasing_risk": (
            dataframe_records_to_json(
                increasing_risk[
                    [
                        "feature",
                        "group",
                        "value",
                        "shap_value",
                        "absolute_shap",
                    ]
                ]
            )
        ),

        "reducing_risk": (
            dataframe_records_to_json(
                reducing_risk[
                    [
                        "feature",
                        "group",
                        "value",
                        "shap_value",
                        "absolute_shap",
                    ]
                ]
            )
        ),

        "grouped_contributions": (
            dataframe_records_to_json(
                grouped_contributions[
                    [
                        "group",
                        "shap_value",
                        "absolute_shap",
                        "feature_count",
                    ]
                ]
            )
        ),

        "grouped_increasing_risk": (
            dataframe_records_to_json(
                grouped_increasing_risk[
                    [
                        "group",
                        "shap_value",
                        "absolute_shap",
                        "feature_count",
                    ]
                ]
            )
        ),

        "grouped_reducing_risk": (
            dataframe_records_to_json(
                grouped_reducing_risk[
                    [
                        "group",
                        "shap_value",
                        "absolute_shap",
                        "feature_count",
                    ]
                ]
            )
        ),
    }

    # ---------------------------------------------------------
    # Save explanation for API/dashboard use
    # ---------------------------------------------------------

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EXPLANATION_PATH,
        "w",
        encoding="utf-8",
    ) as explanation_file:
        json.dump(
            result,
            explanation_file,
            indent=4,
            default=float,
        )

    # ---------------------------------------------------------
    # Terminal output
    # ---------------------------------------------------------

    print(
        f"Observation date : "
        f"{latest_date}"
    )

    print(
        f"Risk probability : "
        f"{probability:.1%}"
    )

    print(
        f"Alert threshold  : "
        f"{warning_threshold:.2f}"
    )

    print(
        f"Warning active   : "
        f"{warning_active}"
    )

    print("\nMain individual factors increasing risk")
    print("---------------------------------------")

    if increasing_risk.empty:
        print(
            "No individual features increased risk."
        )
    else:
        for _, row in increasing_risk.iterrows():
            print(
                f"+ {row['feature']:<28} "
                f"[{row['group']}] "
                f"SHAP={row['shap_value']:.4f}"
            )

    print("\nMain individual factors reducing risk")
    print("-------------------------------------")

    if reducing_risk.empty:
        print(
            "No individual features reduced risk."
        )
    else:
        for _, row in reducing_risk.iterrows():
            print(
                f"- {row['feature']:<28} "
                f"[{row['group']}] "
                f"SHAP={row['shap_value']:.4f}"
            )

    print("\nEconomic categories increasing risk")
    print("-----------------------------------")

    if grouped_increasing_risk.empty:
        print(
            "No economic categories increased risk."
        )
    else:
        for _, row in (
            grouped_increasing_risk
            .iterrows()
        ):
            print(
                f"+ {row['group']:<24} "
                f"SHAP={row['shap_value']:.4f} "
                f"Features={int(row['feature_count'])}"
            )

    print("\nEconomic categories reducing risk")
    print("---------------------------------")

    if grouped_reducing_risk.empty:
        print(
            "No economic categories reduced risk."
        )
    else:
        for _, row in (
            grouped_reducing_risk
            .iterrows()
        ):
            print(
                f"- {row['group']:<24} "
                f"SHAP={row['shap_value']:.4f} "
                f"Features={int(row['feature_count'])}"
            )

    print(
        f"\nExplanation saved to: "
        f"{EXPLANATION_PATH}"
    )

    return result