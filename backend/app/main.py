from services.feature_service import create_features
from services.fred_service import get_series
from services.data_service import save_to_csv
from services.merge_service import build_master_dataset
from services.preprocessing_service import save_feature_dataset
from services.label_service import create_crisis_labels
from services.target_service import create_prediction_targets
from services.stress_service import (
    calculate_stress_index,
    save_stress_dataset,
)
from services.preprocessing_service import (
    load_dataset,
    inspect_missing_values,
)

from config.indicators import INDICATORS

print(" EconIntel Data Pipeline Started\n")

# -----------------------------
# Download all indicators
# -----------------------------
for series_id, column_name in INDICATORS.items():
    print(f"Downloading {column_name}...")

    data = get_series(series_id)

    save_to_csv(data, series_id)

# -----------------------------
# Merge datasets
# -----------------------------
master = build_master_dataset()

print("\nMerged Dataset:")
print(master.head())

print("\nShape:", master.shape)

# -----------------------------
# Load processed dataset
# -----------------------------
dataset = load_dataset()

print("\nFirst 5 rows:")
print(dataset.head())

# -----------------------------
# Missing values
# -----------------------------
inspect_missing_values(dataset)
from services.preprocessing_service import (
    clean_dataset,
    save_clean_dataset,
)

clean_df = clean_dataset(dataset)

print("\nFirst 5 cleaned rows:")
print(clean_df.head())

print("\nRemaining Missing Values:")
print(clean_df.isna().sum())

save_clean_dataset(clean_df)
# -----------------------------------
# Feature Engineering
# -----------------------------------

feature_df = create_features(clean_df)
save_feature_dataset(feature_df)

print("\n📊 Engineered Dataset:")
print(feature_df.head())

print("\nColumns:")
print(feature_df.columns)

print("\nMissing After Feature Engineering:")
print(feature_df.isna().sum())
print("\nTotal Features:", len(feature_df.columns))

# --------------------------------
# Economic Stress Index
# --------------------------------

stress_df = calculate_stress_index(feature_df)

print("\nFirst 5 Stress Scores:")
print(
    stress_df[
        [
            "date",
            "ECON_STRESS",
        ]
    ].head()
)
# ----------------------------
# Crisis Labels
# ----------------------------
stress_df = create_crisis_labels(stress_df)
print("\nFirst Crisis Labels:")

print(
    stress_df[
        [
            "date",
            "ECON_STRESS",
            "CRISIS",
            "CRISIS_NAME"
        ]
    ].tail(30)
)
# ----------------------------
# Prediction Targets
# ----------------------------

dataset = create_prediction_targets(stress_df)

print("\n🎯 First Prediction Targets:")

print(
    dataset[
        [
            "date",
            "ECON_STRESS",
            "TARGET_STRESS_3M",
            "CRISIS",
            "TARGET_CRISIS_3M"
        ]
    ].tail(10)
)