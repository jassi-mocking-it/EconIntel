from config.indicators import INDICATORS
from services.fred_service import get_series
from services.data_service import save_to_csv

print("🚀 EconIntel Data Pipeline Started\n")

for series_id, name in INDICATORS.items():

    print(f"Downloading {name}...")

    data = get_series(series_id)

    print(f"Loaded {len(data)} observations")

    save_to_csv(data, series_id)

    print()
from services.merge_service import build_master_dataset

master = build_master_dataset()

print(master.head(20))

print()
print(master.shape)

print("\nMissing values:")
print(master.isna().sum())