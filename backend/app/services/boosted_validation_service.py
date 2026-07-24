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

from xgboost import XGBClassifier


TARGET_COLUMN = "TARGET_RISK_RISING_3M"
WARNING_THRESHOLD = 0.35

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

    # Internal stress-index representation
    "RAW_ECON_STRESS",
]


def calculate_metrics(
    actual,
    probabilities,
    threshold=WARNING_THRESHOLD,
):
    """
    Convert probabilities into warnings and calculate
    EconIntel classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

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
    }

    if len(np.unique(actual)) == 2:
        metrics["ROC_AUC"] = roc_auc_score(
            actual,
            probabilities,
        )
    else:
        metrics["ROC_AUC"] = np.nan

    return metrics


def build_logistic_regression():
    """
    Current best EconIntel baseline classifier.
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


def build_random_forest():
    """
    Random Forest comparison model.
    """

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


def build_xgboost(scale_pos_weight):
    """
    Build an imbalance-aware XGBoost classifier.

    scale_pos_weight gives additional importance to
    the less common rising-risk class.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",

                    n_estimators=350,
                    learning_rate=0.03,
                    max_depth=3,
                    min_child_weight=3,

                    subsample=0.80,
                    colsample_bytree=0.80,

                    reg_alpha=0.10,
                    reg_lambda=1.50,

                    scale_pos_weight=scale_pos_weight,

                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def run_boosted_model_validation(
    df,
    n_splits=5,
    threshold=WARNING_THRESHOLD,
):
    """
    Compare Logistic Regression, Random Forest and XGBoost
    over identical chronological walk-forward folds.
    """

    print("\n" + "=" * 72)
    print("⚡ ECONINTEL BOOSTED-MODEL WALK-FORWARD BENCHMARK")
    print("=" * 72)

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

    y = data[TARGET_COLUMN]
    dates = data["date"]

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    all_results = []

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

        positive_count = int(y_train.sum())
        negative_count = int(
            len(y_train) - positive_count
        )

        if positive_count == 0:
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = (
                negative_count / positive_count
            )

        models = {
            "Logistic Regression": (
                build_logistic_regression()
            ),
            "Random Forest": (
                build_random_forest()
            ),
            "XGBoost": (
                build_xgboost(
                    scale_pos_weight
                )
            ),
        }

        print(
            f"\nFold {fold_number}: "
            f"{test_dates.min().date()} "
            f"to {test_dates.max().date()}"
        )

        print(
            f"Training positive-class weight: "
            f"{scale_pos_weight:.3f}"
        )

        for model_name, model in models.items():
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

            all_results.append(
                {
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
            )

            print(
                f"{model_name:<22} "
                f"F1={metrics['F1']:.3f}  "
                f"Recall={metrics['Recall']:.3f}  "
                f"PR-AUC={metrics['PR_AUC']:.3f}"
            )

    results_df = pd.DataFrame(
        all_results
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
            [
                "Average_PR_AUC",
                "Average_F1",
            ],
            ascending=False,
        )
    )

    print("\n" + "=" * 72)
    print("📊 BOOSTED MODEL SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    best_model = summary.iloc[0]

    print("\nCurrent benchmark winner")
    print("------------------------")

    print(
        f"Model  : {best_model['Model']}"
    )

    print(
        f"PR-AUC : "
        f"{best_model['Average_PR_AUC']:.3f}"
    )

    print(
        f"F1     : "
        f"{best_model['Average_F1']:.3f}"
    )

    print(
        f"Recall : "
        f"{best_model['Average_Recall']:.3f}"
    )

    return results_df, summary