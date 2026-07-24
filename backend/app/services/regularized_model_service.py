import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
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


def build_regularized_model(
    penalty,
    c_value,
    l1_ratio=None,
):
    """
    Build a scaled Logistic Regression pipeline.

    C controls regularization strength:
    smaller C means stronger regularization.
    """

    model_arguments = {
        "penalty": penalty,
        "C": c_value,
        "class_weight": "balanced",
        "solver": "saga",
        "max_iter": 10000,
        "tol": 1e-4,
        "random_state": 42,
    }

    if penalty == "elasticnet":
        model_arguments["l1_ratio"] = l1_ratio

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
                    **model_arguments
                ),
            ),
        ]
    )


def calculate_metrics(
    actual,
    probabilities,
    threshold,
):
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

    result = {
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
        result["ROC_AUC"] = roc_auc_score(
            actual,
            probabilities,
        )
    else:
        result["ROC_AUC"] = np.nan

    return result


def count_nonzero_coefficients(model):
    """
    Count coefficients retained by the fitted model.
    """

    logistic_model = model.named_steps[
        "model"
    ]

    coefficients = (
        logistic_model.coef_
        .reshape(-1)
    )

    return int(
        np.count_nonzero(
            np.abs(coefficients) > 1e-8
        )
    )


def prepare_model_data(df):
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

    X = X.select_dtypes(
        include=["number"]
    )

    y = data[TARGET_COLUMN]

    return data, X, y


def run_regularized_model_comparison(
    df,
    threshold=0.35,
    n_splits=5,
):
    """
    Compare several regularization approaches using
    identical walk-forward folds.
    """

    print("\n" + "=" * 82)
    print("🧪 ECONINTEL REGULARIZED LOGISTIC COMPARISON")
    print("=" * 82)

    data, X, y = prepare_model_data(
        df
    )

    dates = data["date"]

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print(
        f"Warning threshold: {threshold:.2f}"
    )

    configurations = [
        {
            "Model": "L2 C=1.00",
            "Penalty": "l2",
            "C": 1.00,
            "L1 Ratio": None,
        },
        {
            "Model": "L2 C=0.30",
            "Penalty": "l2",
            "C": 0.30,
            "L1 Ratio": None,
        },
        {
            "Model": "L2 C=0.10",
            "Penalty": "l2",
            "C": 0.10,
            "L1 Ratio": None,
        },
        {
            "Model": "L1 C=1.00",
            "Penalty": "l1",
            "C": 1.00,
            "L1 Ratio": None,
        },
        {
            "Model": "L1 C=0.30",
            "Penalty": "l1",
            "C": 0.30,
            "L1 Ratio": None,
        },
        {
            "Model": "L1 C=0.10",
            "Penalty": "l1",
            "C": 0.10,
            "L1 Ratio": None,
        },
        {
            "Model": "Elastic Net C=1.00",
            "Penalty": "elasticnet",
            "C": 1.00,
            "L1 Ratio": 0.50,
        },
        {
            "Model": "Elastic Net C=0.30",
            "Penalty": "elasticnet",
            "C": 0.30,
            "L1 Ratio": 0.50,
        },
        {
            "Model": "Elastic Net C=0.10",
            "Penalty": "elasticnet",
            "C": 0.10,
            "L1 Ratio": 0.50,
        },
    ]

    splitter = TimeSeriesSplit(
        n_splits=n_splits
    )

    results = []

    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    for configuration in configurations:
        model_name = configuration[
            "Model"
        ]

        print("\n" + "-" * 82)
        print(model_name)
        print("-" * 82)

        for fold_number, (
            train_indices,
            test_indices,
        ) in enumerate(
            splitter.split(X),
            start=1,
        ):
            X_train = X.iloc[
                train_indices
            ]

            X_test = X.iloc[
                test_indices
            ]

            y_train = y.iloc[
                train_indices
            ]

            y_test = y.iloc[
                test_indices
            ]

            test_dates = dates.iloc[
                test_indices
            ]

            if y_train.nunique() < 2:
                print(
                    f"Fold {fold_number} skipped: "
                    "training has one class."
                )
                continue

            model = build_regularized_model(
                penalty=configuration[
                    "Penalty"
                ],
                c_value=configuration[
                    "C"
                ],
                l1_ratio=configuration[
                    "L1 Ratio"
                ],
            )

            model.fit(
                X_train,
                y_train,
            )

            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            metrics = calculate_metrics(
                actual=y_test,
                probabilities=probabilities,
                threshold=threshold,
            )

            retained_features = (
                count_nonzero_coefficients(
                    model
                )
            )

            results.append(
                {
                    "Model": model_name,
                    "Fold": fold_number,
                    "Test Start": (
                        test_dates.min().date()
                    ),
                    "Test End": (
                        test_dates.max().date()
                    ),
                    "Retained Features": (
                        retained_features
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
                f"FP={metrics['False Positives']}  "
                f"Features={retained_features}"
            )

    results_df = pd.DataFrame(
        results
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
            Average_Retained_Features=(
                "Retained Features",
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
    )

    summary["False_Alarms_Per_Year"] = (
        summary["Total_False_Positives"]
        / (
            (
                dates.max()
                - dates.iloc[
                    len(data) // (
                        n_splits + 1
                    )
                ]
            ).days
            / 365.25
        )
    )

    summary = summary.sort_values(
        [
            "Average_PR_AUC",
            "Average_F1",
            "Total_False_Positives",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    print("\n" + "=" * 82)
    print("📊 REGULARIZED MODEL SUMMARY")
    print("=" * 82)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    winner = summary.iloc[0]

    print("\nCurrent regularization winner")
    print("-----------------------------")

    print(
        f"Model               : "
        f"{winner['Model']}"
    )

    print(
        f"PR-AUC              : "
        f"{winner['Average_PR_AUC']:.3f}"
    )

    print(
        f"F1                  : "
        f"{winner['Average_F1']:.3f}"
    )

    print(
        f"Recall              : "
        f"{winner['Average_Recall']:.3f}"
    )

    print(
        f"False positives     : "
        f"{int(winner['Total_False_Positives'])}"
    )

    print(
        f"Average features    : "
        f"{winner['Average_Retained_Features']:.1f}"
    )

    return results_df, summary