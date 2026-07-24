from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "TARGET_RISK_RISING_3M"

EXCLUDED_COLUMNS = [
    "date",

    # Regression and classification targets
    "TARGET_STRESS_3M",
    "TARGET_STRESS_CHANGE_3M",
    "TARGET_RISK_RISING_3M",
    "TARGET_CRISIS_3M",

    # Historical labels
    "CRISIS",
    "CRISIS_NAME",

    # Internal duplicate representation
    "RAW_ECON_STRESS",
]


def calculate_classification_metrics(
    actual,
    predicted,
    probabilities=None,
):
    """
    Calculate EconIntel early-warning classification metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
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
    }

    if probabilities is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(
                actual,
                probabilities,
            )

            metrics["PR_AUC"] = average_precision_score(
                actual,
                probabilities,
            )

        except ValueError:
            metrics["ROC_AUC"] = float("nan")
            metrics["PR_AUC"] = float("nan")
    else:
        metrics["ROC_AUC"] = float("nan")
        metrics["PR_AUC"] = float("nan")

    return metrics


def build_dummy_classifier():
    """
    Baseline that always predicts the most frequent class.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent",
                ),
            ),
        ]
    )


def build_logistic_regression():
    """
    Scaled Logistic Regression classifier.
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


def build_random_forest_classifier():
    """
    Random Forest early-warning classifier.
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


def evaluate_classifier(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate one classifier.
    """

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    metrics = calculate_classification_metrics(
        y_test,
        predictions,
        probabilities,
    )

    print(f"✅ {model_name} trained!")

    return {
        "name": model_name,
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "feature_names": list(X_train.columns),
    }


def print_classifier_details(result, y_test):
    """
    Print the confusion matrix and classification report.
    """

    print(f"\n{result['name']}")
    print("-" * len(result["name"]))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            result["predictions"],
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            result["predictions"],
            target_names=[
                "Stable/Falling",
                "Rising Risk",
            ],
            zero_division=0,
        )
    )


def print_feature_importance(result, top_n=15):
    """
    Display Random Forest feature importance.
    """

    classifier = result["model"].named_steps["model"]

    importance_df = pd.DataFrame(
        {
            "Feature": result["feature_names"],
            "Importance": classifier.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    print(
        f"\nTop {top_n} Early-Warning Features"
    )
    print("-" * 50)

    print(
        importance_df
        .head(top_n)
        .to_string(index=False)
    )


def save_best_risk_model(best_result):
    """
    Save the best EconIntel rising-risk classifier.
    """

    models_directory = (
        Path(__file__).resolve().parent.parent
        / "models"
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_directory
        / "best_risk_classifier.pkl"
    )

    artifact = {
        "model_name": best_result["name"],
        "model": best_result["model"],
        "features": best_result["feature_names"],
        "metrics": best_result["metrics"],
        "target": TARGET_COLUMN,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    print(
        f"\n💾 Best risk classifier saved to:\n"
        f"{model_path}"
    )


def run_risk_classification_benchmark(df):
    """
    Benchmark EconIntel's rising-risk classifiers.
    """

    print("\n" + "=" * 65)
    print("🚨 ECONINTEL RISING-RISK CLASSIFICATION")
    print("=" * 65)

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

    # Chronological 80/20 split.
    split_index = int(len(data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = data["date"].iloc[split_index:]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print(
        f"Testing period: "
        f"{test_dates.min().date()} "
        f"to {test_dates.max().date()}"
    )

    print(
        f"Training rising-risk cases: "
        f"{int(y_train.sum())}"
    )

    print(
        f"Testing rising-risk cases : "
        f"{int(y_test.sum())}"
    )

    models = [
        (
            "Majority-Class Baseline",
            build_dummy_classifier(),
        ),
        (
            "Logistic Regression",
            build_logistic_regression(),
        ),
        (
            "Random Forest Classifier",
            build_random_forest_classifier(),
        ),
    ]

    results = []

    for model_name, model in models:
        result = evaluate_classifier(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(result)

    comparison = pd.DataFrame(
        [
            {
                "Model": result["name"],
                **result["metrics"],
            }
            for result in results
        ]
    ).sort_values(
        "F1",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("📊 RISING-RISK MODEL COMPARISON")
    print("=" * 65)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    # Ignore the dummy baseline when choosing the saved model.
    real_models = [
        result
        for result in results
        if result["name"] != "Majority-Class Baseline"
    ]

    best_result = max(
        real_models,
        key=lambda result: result["metrics"]["F1"],
    )

    print(
        f"\n🏆 Best Risk Classifier: "
        f"{best_result['name']}"
    )

    print(
        f"Best F1 Score: "
        f"{best_result['metrics']['F1']:.3f}"
    )

    print_classifier_details(
        best_result,
        y_test,
    )

    if best_result["name"] == "Random Forest Classifier":
        print_feature_importance(best_result)

    save_best_risk_model(best_result)

    return comparison
