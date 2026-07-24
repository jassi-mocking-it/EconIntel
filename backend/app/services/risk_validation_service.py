import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def build_random_forest_classifier():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def calculate_metrics(
    actual,
    predicted,
    probabilities,
):
    metrics = {
        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "F1": f1_score(
            actual,
            predicted,
            zero_division=0,
        ),
        "PR_AUC": average_precision_score(
            actual,
            probabilities,
        ),
    }

    if len(np.unique(actual)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(
            actual,
            probabilities,
        )
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


def run_classifier_walk_forward(
    df,
    n_splits=5,
):
    """
    Test EconIntel rising-risk classifiers across
    multiple chronological economic periods.
    """

    print("\n" + "=" * 70)
    print("🚨 ECONINTEL CLASSIFICATION WALK-FORWARD VALIDATION")
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
    dates = data["date"]

    models = {
        "Logistic Regression": build_logistic_regression,
        "Random Forest Classifier": (
            build_random_forest_classifier
        ),
    }

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
    )

    all_results = []

    for model_name, model_builder in models.items():
        print("\n" + "-" * 70)
        print(model_name)
        print("-" * 70)

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

            test_dates = dates.iloc[test_indices]

            # A fold must contain both training classes.
            if y_train.nunique() < 2:
                print(
                    f"Fold {fold_number} skipped: "
                    "training data contains one class."
                )
                continue

            model = model_builder()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(X_test)[:, 1]
            )

            predictions = (
                probabilities >= 0.50
            ).astype(int)

            metrics = calculate_metrics(
                y_test,
                predictions,
                probabilities,
            )

            result = {
                "Model": model_name,
                "Fold": fold_number,
                "Test Start": (
                    test_dates.min().date()
                ),
                "Test End": (
                    test_dates.max().date()
                ),
                "Positive Cases": int(
                    y_test.sum()
                ),
                **metrics,
            }

            all_results.append(result)

            print(
                f"\nFold {fold_number}: "
                f"{test_dates.min().date()} "
                f"to {test_dates.max().date()}"
            )

            print(
                f"Precision: "
                f"{metrics['Precision']:.3f}"
            )

            print(
                f"Recall   : "
                f"{metrics['Recall']:.3f}"
            )

            print(
                f"F1       : "
                f"{metrics['F1']:.3f}"
            )

            print(
                f"PR-AUC   : "
                f"{metrics['PR_AUC']:.3f}"
            )

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION WALK-FORWARD RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    summary = (
        results_df
        .groupby("Model")
        .agg(
            Average_Precision=(
                "Precision",
                "mean",
            ),
            Average_Recall=(
                "Recall",
                "mean",
            ),
            Average_F1=(
                "F1",
                "mean",
            ),
            Average_PR_AUC=(
                "PR_AUC",
                "mean",
            ),
            Average_ROC_AUC=(
                "ROC_AUC",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Average_F1",
            ascending=False,
        )
    )

    print("\n" + "=" * 70)
    print("📈 AVERAGE CLASSIFICATION PERFORMANCE")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    return results_df, summary