from config.indicators import INDICATORS

from services.data_service import save_to_csv
from services.feature_service import create_features
from services.fred_service import get_series
from services.label_service import create_crisis_labels
from services.merge_service import build_master_dataset
from services.historical_snapshot_service import (
    create_us_stress_history_snapshot,
)
from services.production_inference_service import (
    create_production_risk_snapshot,
)
from services.feature_audit_service import (
    run_feature_audit,
)
from services.explainability_service import (
    explain_latest_risk,
)
from services.final_risk_service import (
    train_and_save_final_risk_model,
)
from services.boosted_validation_service import (
    run_boosted_model_validation,
)
from services.threshold_service import (
    tune_probability_threshold,
)
from services.risk_validation_service import (
    run_classifier_walk_forward,
)
from services.risk_model_service import (
    run_risk_classification_benchmark,
)
from services.preprocessing_service import (
    clean_dataset,
    inspect_missing_values,
    load_dataset,
    save_clean_dataset,
    save_feature_dataset,
)
from services.stress_service import (
    calculate_stress_index,
    save_stress_dataset,
)
from services.target_service import create_prediction_targets
from services.training_service import create_training_dataset
from services.curated_model_service import (
    run_curated_feature_comparison,
)
from services.regularized_model_service import (
    run_regularized_model_comparison,
)
# We will re-enable these after inspecting the new targets.
# from services.model_service import run_model_benchmark
# from services.validation_service import run_walk_forward_validation


def main():
    """
    Run the complete EconIntel data and target-generation pipeline.
    """

    print("\n🚀 EconIntel Data Pipeline Started\n")

    # =========================================================
    # 1. Download FRED indicators
    # =========================================================

    print("=" * 60)
    print("1. DOWNLOADING ECONOMIC INDICATORS")
    print("=" * 60)

    for series_id, column_name in INDICATORS.items():
        print(f"\nDownloading {column_name}...")

        observations = get_series(series_id)

        save_to_csv(
            observations,
            series_id,
        )

    # =========================================================
    # 2. Merge raw indicator datasets
    # =========================================================

    print("\n" + "=" * 60)
    print("2. MERGING DATASETS")
    print("=" * 60)

    merged_df = build_master_dataset()

    print("\nMerged dataset:")
    print(merged_df.head())

    print("\nMerged shape:")
    print(merged_df.shape)

    # =========================================================
    # 3. Load merged dataset
    # =========================================================

    print("\n" + "=" * 60)
    print("3. LOADING MERGED DATASET")
    print("=" * 60)

    dataset = load_dataset()

    print("\nFirst 5 merged rows:")
    print(dataset.head())

    inspect_missing_values(dataset)

    # =========================================================
    # 4. Clean and convert to monthly frequency
    # =========================================================

    print("\n" + "=" * 60)
    print("4. CLEANING AND MONTHLY ALIGNMENT")
    print("=" * 60)

    dataset = clean_dataset(dataset)

    print("\nFirst 5 cleaned monthly rows:")
    print(dataset.head())

    print("\nRemaining missing values:")
    print(dataset.isna().sum())

    save_clean_dataset(dataset)

    # =========================================================
    # 5. Feature engineering
    # =========================================================

    print("\n" + "=" * 60)
    print("5. FEATURE ENGINEERING")
    print("=" * 60)

    dataset = create_features(dataset)

    save_feature_dataset(dataset)

    print("\nEngineered dataset:")
    print(dataset.head())

    print("\nFeature columns:")
    print(dataset.columns)

    print("\nMissing values after feature engineering:")
    print(dataset.isna().sum())

    print(f"\nTotal columns: {len(dataset.columns)}")

    # =========================================================
    # 6. Causal Economic Stress Index
    # =========================================================

    print("\n" + "=" * 60)
    print("6. ECONOMIC STRESS INDEX")
    print("=" * 60)

    dataset = calculate_stress_index(dataset)

    save_stress_dataset(dataset)

    print("\nFirst valid stress scores:")

    print(
        dataset[
            [
                "date",
                "ECON_STRESS",
            ]
        ]
        .dropna()
        .head(10)
    )

    # =========================================================
    # 7. Historical crisis labels
    # =========================================================

    print("\n" + "=" * 60)
    print("7. HISTORICAL CRISIS LABELS")
    print("=" * 60)

    dataset = create_crisis_labels(dataset)

    print("\nSample crisis-labelled rows:")

    print(
        dataset[
            [
                "date",
                "ECON_STRESS",
                "CRISIS",
                "CRISIS_NAME",
            ]
        ]
        .dropna(subset=["ECON_STRESS"])
        .tail(15)
    )

    # =========================================================
    # 8. Three-month early-warning targets
    # =========================================================

    print("\n" + "=" * 60)
    print("8. THREE-MONTH EARLY-WARNING TARGETS")
    print("=" * 60)

    dataset = create_prediction_targets(dataset)

    print("\nSample prediction targets:")

    print(
        dataset[
            [
                "date",
                "ECON_STRESS",
                "TARGET_STRESS_3M",
                "TARGET_STRESS_CHANGE_3M",
                "TARGET_RISK_RISING_3M",
                "CRISIS",
                "TARGET_CRISIS_3M",
            ]
        ]
        .dropna(subset=["ECON_STRESS"])
        .tail(15)
    )

    # =========================================================
    # 9. Prepare ML dataset
    # =========================================================

    print("\n" + "=" * 60)
    print("9. PREPARING ML DATASET")
    print("=" * 60)

    training_df = create_training_dataset(dataset)

    print("\nTraining dataset sample:")

    print(
        training_df[
            [
                "date",
                "ECON_STRESS",
                "TARGET_STRESS_3M",
                "TARGET_STRESS_CHANGE_3M",
                "TARGET_RISK_RISING_3M",
            ]
        ].head(10)
    )

    # =========================================================
    # 10. Rising-risk classification
    # =========================================================

    risk_comparison = (
        run_risk_classification_benchmark(
            training_df
        )
    )

    # =========================================================
    # 11. Classification walk-forward validation
    # =========================================================

    risk_fold_results, risk_validation_summary = (
        run_classifier_walk_forward(
            training_df,
            n_splits=5,
        )
    )

    # =========================================================
    # 12. Probability-threshold tuning
    # =========================================================

    (
        threshold_results,
        best_threshold,
        threshold_predictions,
    ) = tune_probability_threshold(
        training_df,
        n_splits=5,
    )
    
    print(
        f"\nSelected EconIntel warning threshold: "
        f"{best_threshold:.2f}"
    )
    
    boosted_fold_results, boosted_summary = (run_boosted_model_validation
    (
        training_df,
        n_splits=5,
        threshold=best_threshold,
        )
    )
    # =========================================================
    # 13. XGBoost and boosted-model benchmark
    # =========================================================

    final_model_bundle, latest_risk_assessment = (
        train_and_save_final_risk_model(
            training_df=training_df,
            full_feature_df=dataset,
            warning_threshold=best_threshold,
        )
    )
    # =========================================================
    # 15. Explain latest risk prediction
    # =========================================================

    print("\n" + "=" * 60)
    print("15. LATEST-RISK EXPLAINABILITY")
    print("=" * 60)

    latest_risk_explanation = explain_latest_risk(
        full_feature_df=dataset,
        model_bundle=final_model_bundle,
        top_n=8,
    )

    feature_audit_results = run_feature_audit(
        training_df,
        correlation_threshold=0.90,
    )
    (
        curated_fold_results,
        curated_feature_summary,
    ) = run_curated_feature_comparison(
        training_df,
        threshold=best_threshold,
        n_splits=5,
    )
    (
        regularized_fold_results,
        regularized_model_summary,
    ) = run_regularized_model_comparison(
        training_df,
        threshold=best_threshold,
        n_splits=5,
    )

    # =========================================================
    # 16. Create production API snapshot
    # =========================================================

    print("\n" + "=" * 60)
    print("16. PRODUCTION RISK SNAPSHOT")
    print("=" * 60)

    production_risk_snapshot = (
        create_production_risk_snapshot(
            latest_risk_assessment=(
                latest_risk_assessment
            ),
            latest_risk_explanation=(
                latest_risk_explanation
            ),
        )
    )
    print("\n" + "=" * 60)
    print("✅ ECONINTEL EARLY-WARNING PIPELINE COMPLETED")
    print("=" * 60)
        # =========================================================
    # 17. Create historical stress snapshot
    # =========================================================

    print("\n" + "=" * 60)
    print("17. HISTORICAL STRESS SNAPSHOT")
    print("=" * 60)

    us_stress_history_snapshot = (
        create_us_stress_history_snapshot(
            full_feature_df=dataset,
        )
    )

if __name__ == "__main__":

    main()