import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit


TARGET_COLUMN = "TARGET_STRESS_3M"

EXCLUDED_COLUMNS = [
    "date",
    "TARGET_STRESS_3M",
    "TARGET_CRISIS_3M",
    "CRISIS",
    "CRISIS_NAME",
    "RAW_ECON_STRESS",
]


def calculate_metrics(actual, predicted):
    """
    Calculate forecasting metrics for one validation fold.
    """

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(
            actual,
            predicted,
        ) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def build_random_forest():
    """
    Create the Random Forest used during walk-forward validation.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def run_walk_forward_validation(
    df,
    n_splits=5,
    include_current_stress=True,
):
    """
    Evaluate EconIntel across several chronological test windows.

    Each fold trains on all previous observations and tests
    only on a later unseen period.
    """

    model_name = (
        "Random Forest + Current Stress"
        if include_current_stress
        else "Random Forest — Macro Only"
    )

    print("\n" + "=" * 65)
    print(f"⏳ WALK-FORWARD VALIDATION: {model_name}")
    print("=" * 65)

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = (
        data
        .dropna(subset=[TARGET_COLUMN])
        .sort_values("date")
        .reset_index(drop=True)
    )

    excluded = EXCLUDED_COLUMNS.copy()

    if not include_current_stress:
        excluded.append("ECON_STRESS")

    X = data.drop(
        columns=excluded,
        errors="ignore",
    )

    y = data[TARGET_COLUMN]

    dates = data["date"]

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
    )

    fold_results = []

    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        test_dates = dates.iloc[test_indices]

        model = build_random_forest()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        model_metrics = calculate_metrics(
            y_test,
            predictions,
        )

        # Persistence forecast:
        # future stress equals stress observed today.
        baseline_predictions = data.iloc[test_indices][
            "ECON_STRESS"
        ].to_numpy()

        baseline_metrics = calculate_metrics(
            y_test,
            baseline_predictions,
        )

        fold_result = {
            "Fold": fold_number,
            "Test Start": test_dates.min().date(),
            "Test End": test_dates.max().date(),
            "Model MAE": model_metrics["MAE"],
            "Model RMSE": model_metrics["RMSE"],
            "Model R2": model_metrics["R2"],
            "Baseline RMSE": baseline_metrics["RMSE"],
            "RMSE Improvement": (
                baseline_metrics["RMSE"]
                - model_metrics["RMSE"]
            ),
        }

        fold_results.append(fold_result)

        print(
            f"\nFold {fold_number}: "
            f"{test_dates.min().date()} "
            f"to {test_dates.max().date()}"
        )

        print(
            f"Model RMSE   : "
            f"{model_metrics['RMSE']:.3f}"
        )

        print(
            f"Baseline RMSE: "
            f"{baseline_metrics['RMSE']:.3f}"
        )

    results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 65)
    print("📊 WALK-FORWARD RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    average_model_rmse = results_df[
        "Model RMSE"
    ].mean()

    average_baseline_rmse = results_df[
        "Baseline RMSE"
    ].mean()

    average_improvement = results_df[
        "RMSE Improvement"
    ].mean()

    folds_beating_baseline = int(
        (
            results_df["RMSE Improvement"] > 0
        ).sum()
    )

    print("\nOverall Walk-Forward Summary")
    print("----------------------------")

    print(
        f"Average model RMSE   : "
        f"{average_model_rmse:.3f}"
    )

    print(
        f"Average baseline RMSE: "
        f"{average_baseline_rmse:.3f}"
    )

    print(
        f"Average improvement  : "
        f"{average_improvement:.3f}"
    )

    print(
        f"Baseline beaten in   : "
        f"{folds_beating_baseline}/{n_splits} folds"
    )

    if average_improvement > 0:
        print(
            "✅ Model beats persistence on average."
        )
    else:
        print(
            "⚠️ Model does not beat persistence on average."
        )

    return results_df