import os
import pandas as pd

# Project root
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def save_to_csv(data, filename):
    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.DataFrame(data)

    output_path = os.path.join(DATA_DIR, f"{filename}.csv")

    df.to_csv(output_path, index=False)

    print(f"✅ Saved {len(df)} rows to")
    print(output_path)