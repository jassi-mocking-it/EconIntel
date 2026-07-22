from services.feature_service import create_features
from services.fred_service import get_series
from services.data_service import save_to_csv
from services.merge_service import build_master_dataset
from services.preprocessing_service import save_feature_dataset
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