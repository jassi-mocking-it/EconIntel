import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "TARGET_RISK_RISING_3M"

EXCLUDED_COLUMNS = [
    "date",
    "TARGET_STRESS_3M",
    "TARGET_STRESS_CHANGE_3M",
    "TARGET_RISK_RISING_3M",
    "TARGET_CRISIS_3M",
    "CRISIS",
    "CRISIS_NAME",
    "RAW_ECON_STRESS",
]


def build_logistic_regression():
    """
    Build EconIntel's current best rising-risk classifier.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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


def tune_probability_threshold(
    df,
    n_splits=5,
):
    """
    Find a probability threshold for EconIntel using
    out-of-fold walk-forward predictions.

    Earlier periods are used to predict later periods,
    preventing random time-series shuffling.
    """

    print("\n" + "=" * 70)
    print("🎚️ ECONINTEL PROBABILITY-THRESHOLD TUNING")
    print("=" * 70)

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

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

    y = data[TARGET_COLUMN]

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
    )

    all_probabilities = []
    all_actual_values = []
    all_dates = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        if y_train.nunique() < 2:
            print(
                f"Fold {fold_number} skipped because "
                "its training data contains only one class."
            )
            continue

        model = build_logistic_regression()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

        all_actual_values.extend(
            y_test.tolist()
        )

        all_dates.extend(
            data["date"]
            .iloc[test_indices]
            .tolist()
        )

    prediction_df = pd.DataFrame(
        {
            "date": all_dates,
            "actual": all_actual_values,
            "probability": all_probabilities,
        }
    )

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    threshold_results = []

    for threshold in thresholds:
        predictions = (
            prediction_df["probability"]
            >= threshold
        ).astype(int)

        precision = precision_score(
            prediction_df["actual"],
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            prediction_df["actual"],
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            prediction_df["actual"],
            predictions,
            zero_division=0,
        )

        matrix = confusion_matrix(
            prediction_df["actual"],
            predictions,
            labels=[0, 1],
        )

        true_negatives = matrix[0, 0]
        false_positives = matrix[0, 1]
        false_negatives = matrix[1, 0]
        true_positives = matrix[1, 1]

        threshold_results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "True Positives": true_positives,
                "False Positives": false_positives,
                "False Negatives": false_negatives,
                "True Negatives": true_negatives,
            }
        )

    results_df = pd.DataFrame(
        threshold_results
    )

    results_df = results_df.sort_values(
        "F1",
        ascending=False,
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]

    pr_auc = average_precision_score(
        prediction_df["actual"],
        prediction_df["probability"],
    )

    print("\nThreshold Comparison")
    print("--------------------")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nBest Threshold")
    print("--------------")

    print(
        f"Threshold : "
        f"{best_row['Threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_row['Precision']:.3f}"
    )

    print(
        f"Recall    : "
        f"{best_row['Recall']:.3f}"
    )

    print(
        f"F1        : "
        f"{best_row['F1']:.3f}"
    )

    print(
        f"PR-AUC    : "
        f"{pr_auc:.3f}"
    )

    return (
        results_df,
        float(best_row["Threshold"]),
        prediction_df,
    )