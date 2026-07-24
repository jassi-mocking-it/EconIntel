from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "TARGET_STRESS_3M"

EXCLUDED_COLUMNS = [
    "date",
    "TARGET_STRESS_3M",
    "TARGET_CRISIS_3M",
    "CRISIS",
    "CRISIS_NAME",

    # Internal stress-calculation column.
    # ECON_STRESS is the readable 0–100 version.
    "RAW_ECON_STRESS",
]


def calculate_metrics(actual, predicted):
    """
    Calculate EconIntel regression metrics.
    """

    mae = mean_absolute_error(actual, predicted)

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def create_time_split(df):
    """
    Create a chronological 80/20 train-test split.
    """

    data = df.copy()

    data = data.dropna(
        subset=[TARGET_COLUMN]
    )

    data = data.sort_values("date").reset_index(drop=True)

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows : {len(test_data)}")

    print(
        "Training period:",
        train_data["date"].min(),
        "to",
        train_data["date"].max(),
    )

    print(
        "Testing period :",
        test_data["date"].min(),
        "to",
        test_data["date"].max(),
    )

    return train_data, test_data


def prepare_features(
    train_data,
    test_data,
    include_current_stress=True,
):
    """
    Prepare feature matrices for a model.

    When include_current_stress=False, ECON_STRESS
    is removed so we can test whether macroeconomic
    variables alone contain forecasting information.
    """

    excluded = EXCLUDED_COLUMNS.copy()

    if not include_current_stress:
        excluded.append("ECON_STRESS")

    X_train = train_data.drop(
        columns=excluded,
        errors="ignore",
    )

    X_test = test_data.drop(
        columns=excluded,
        errors="ignore",
    )

    y_train = train_data[TARGET_COLUMN]
    y_test = test_data[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_random_forest():
    """
    Build the Random Forest pipeline.
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


def build_linear_regression():
    """
    Build a scaled Linear Regression baseline.
    """

    regression_pipeline = Pipeline(
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
                LinearRegression(),
            ),
        ]
    )

    # Scale the target during training and convert predictions
    # back to the original 0-100 stress-index scale.
    return TransformedTargetRegressor(
        regressor=regression_pipeline,
        transformer=StandardScaler(),
    )


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train one model and return its performance.
    """

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print(f"✅ {model_name} trained!")

    return {
        "name": model_name,
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "feature_names": list(X_train.columns),
    }


def print_feature_importance(result, top_n=15):
    """
    Print Random Forest feature importance.
    """

    pipeline = result["model"]

    random_forest = pipeline.named_steps["model"]

    importance_df = pd.DataFrame(
        {
            "Feature": result["feature_names"],
            "Importance": random_forest.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    print(
        f"\nTop {top_n} Features — {result['name']}"
    )

    print("-" * 55)

    print(
        importance_df
        .head(top_n)
        .to_string(index=False)
    )


def save_best_model(best_result):
    """
    Save the best-performing ML model.
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
        / "best_stress_model.pkl"
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

    print(f"\n💾 Best model saved to:\n{model_path}")


def run_model_benchmark(df):
    """
    Run EconIntel's first forecasting benchmark.

    Models:
    1. Persistence baseline
    2. Random Forest including current ECON_STRESS
    3. Random Forest using macroeconomic variables only
    4. Linear Regression
    """

    print("\n" + "=" * 60)
    print("🧠 ECONINTEL MODEL BENCHMARK")
    print("=" * 60)

    train_data, test_data = create_time_split(df)

    y_test = test_data[TARGET_COLUMN]

    # -------------------------------------------------
    # 1. Persistence baseline
    # -------------------------------------------------

    baseline_predictions = test_data[
        "ECON_STRESS"
    ].to_numpy()

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    results = [
        {
            "name": "Persistence Baseline",
            "model": None,
            "predictions": baseline_predictions,
            "metrics": baseline_metrics,
            "feature_names": ["ECON_STRESS"],
        }
    ]

    # -------------------------------------------------
    # 2. Random Forest with current stress
    # -------------------------------------------------

    (
        X_train_with_stress,
        X_test_with_stress,
        y_train,
        y_test,
    ) = prepare_features(
        train_data,
        test_data,
        include_current_stress=True,
    )

    rf_with_stress = evaluate_model(
        model_name="Random Forest + Current Stress",
        model=build_random_forest(),
        X_train=X_train_with_stress,
        X_test=X_test_with_stress,
        y_train=y_train,
        y_test=y_test,
    )

    results.append(rf_with_stress)

    # -------------------------------------------------
    # 3. Random Forest using macro features only
    # -------------------------------------------------

    (
        X_train_macro,
        X_test_macro,
        y_train_macro,
        y_test_macro,
    ) = prepare_features(
        train_data,
        test_data,
        include_current_stress=False,
    )

    rf_macro_only = evaluate_model(
        model_name="Random Forest — Macro Only",
        model=build_random_forest(),
        X_train=X_train_macro,
        X_test=X_test_macro,
        y_train=y_train_macro,
        y_test=y_test_macro,
    )

    results.append(rf_macro_only)

    # -------------------------------------------------
    # 4. Linear Regression
    # -------------------------------------------------

    linear_regression = evaluate_model(
        model_name="Linear Regression",
        model=build_linear_regression(),
        X_train=X_train_macro,
        X_test=X_test_macro,
        y_train=y_train_macro,
        y_test=y_test_macro,
    )

    results.append(linear_regression)

    # -------------------------------------------------
    # Comparison table
    # -------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": result["name"],
                "MAE": result["metrics"]["MAE"],
                "RMSE": result["metrics"]["RMSE"],
                "R²": result["metrics"]["R2"],
            }
            for result in results
        ]
    ).sort_values(
        "RMSE",
        ascending=True,
    )

    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    # Lowest RMSE wins.
    ml_results = [
        result
        for result in results
        if result["model"] is not None
    ]

    best_result = min(
        ml_results,
        key=lambda result: result["metrics"]["RMSE"],
    )

    print(
        f"\n🏆 Best ML Model: {best_result['name']}"
    )

    print(
        f"Best RMSE: "
        f"{best_result['metrics']['RMSE']:.3f}"
    )

    # Print feature importance for both forests.
    print_feature_importance(rf_with_stress)
    print_feature_importance(rf_macro_only)

    save_best_model(best_result)

    return comparison