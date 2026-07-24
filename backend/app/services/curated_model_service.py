import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.model_features import EARLY_WARNING_FEATURES


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


def build_logistic_model():
    """
    Build the Logistic Regression pipeline used for both
    the full and curated feature comparisons.
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


def calculate_metrics(
    actual,
    probabilities,
    threshold,
):
    """
    Calculate early-warning metrics at a selected threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1],
    )

    true_negatives = int(matrix[0, 0])
    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])
    true_positives = int(matrix[1, 1])

    metrics = {
        "Precision": precision_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            actual,
            predictions,
            zero_division=0,
        ),
        "PR_AUC": average_precision_score(
            actual,
            probabilities,
        ),
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "True Negatives": true_negatives,
    }

    if len(np.unique(actual)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(
            actual,
            probabilities,
        )
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


def prepare_data(df):
    """
    Prepare chronologically ordered labelled data.
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

    y = data[TARGET_COLUMN]

    full_features = data.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    full_features = (
        full_features
        .select_dtypes(include=["number"])
    )

    missing_curated_features = [
        feature
        for feature in EARLY_WARNING_FEATURES
        if feature not in data.columns
    ]

    if missing_curated_features:
        raise ValueError(
            "The following curated features are missing: "
            + ", ".join(missing_curated_features)
        )

    curated_features = data[
        EARLY_WARNING_FEATURES
    ].copy()

    return (
        data,
        full_features,
        curated_features,
        y,
    )


def run_curated_feature_comparison(
    df,
    threshold=0.35,
    n_splits=5,
):
    """
    Compare the existing full feature set with the smaller,
    more interpretable curated feature set.
    """

    print("\n" + "=" * 76)
    print("🧪 ECONINTEL FULL VS CURATED FEATURE COMPARISON")
    print("=" * 76)

    (
        data,
        full_X,
        curated_X,
        y,
    ) = prepare_data(df)

    dates = data["date"]

    print(
        f"Full feature count    : "
        f"{len(full_X.columns)}"
    )

    print(
        f"Curated feature count : "
        f"{len(curated_X.columns)}"
    )

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    results = []

    feature_sets = {
        "Full 42-Feature Model": full_X,
        "Curated Feature Model": curated_X,
    }

    for feature_set_name, X in feature_sets.items():
        print("\n" + "-" * 76)
        print(feature_set_name)
        print("-" * 76)

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

            if y_train.nunique() < 2:
                print(
                    f"Fold {fold_number} skipped because "
                    "training contains only one class."
                )
                continue

            model = build_logistic_model()

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(X_test)[:, 1]
            )

            metrics = calculate_metrics(
                actual=y_test,
                probabilities=probabilities,
                threshold=threshold,
            )

            results.append(
                {
                    "Feature Set": feature_set_name,
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
            )

            print(
                f"Fold {fold_number}: "
                f"F1={metrics['F1']:.3f}  "
                f"Recall={metrics['Recall']:.3f}  "
                f"Precision={metrics['Precision']:.3f}  "
                f"PR-AUC={metrics['PR_AUC']:.3f}  "
                f"False positives="
                f"{metrics['False Positives']}"
            )

    results_df = pd.DataFrame(
        results
    )

    summary = (
        results_df
        .groupby("Feature Set")
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
            Total_True_Positives=(
                "True Positives",
                "sum",
            ),
            Total_False_Positives=(
                "False Positives",
                "sum",
            ),
            Total_False_Negatives=(
                "False Negatives",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "Average_PR_AUC",
                "Average_F1",
            ],
            ascending=False,
        )
    )

    print("\n" + "=" * 76)
    print("📊 FULL VS CURATED FEATURE SUMMARY")
    print("=" * 76)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    winner = summary.iloc[0]

    print("\nCurrent comparison winner")
    print("-------------------------")

    print(
        f"Feature set : "
        f"{winner['Feature Set']}"
    )

    print(
        f"PR-AUC     : "
        f"{winner['Average_PR_AUC']:.3f}"
    )

    print(
        f"F1         : "
        f"{winner['Average_F1']:.3f}"
    )

    print(
        f"Recall     : "
        f"{winner['Average_Recall']:.3f}"
    )

    print(
        f"False alarms across folds: "
        f"{int(winner['Total_False_Positives'])}"
    )

    return results_df, summary