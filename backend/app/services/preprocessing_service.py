import os
import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_dataset():

    path = os.path.join(PROCESSED_DIR, "merged_macro_data.csv")

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    print("✅ Dataset loaded successfully!")

    return df


def inspect_missing_values(df):

    print("\n📊 Missing Values:\n")
    print(df.isna().sum())


def clean_dataset(df):

    print("\n🧹 Cleaning dataset...")

    # Sort by date
    df = df.sort_values("date")

    # Forward fill
    df = df.ffill()

    # Backward fill remaining NaNs
    df = df.bfill()

    print("✅ Dataset cleaned!")

    return df


def save_clean_dataset(df):

    output_path = os.path.join(
        PROCESSED_DIR,
        "clean_macro_data.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\n Clean dataset saved to:\n{output_path}")

def save_feature_dataset(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "feature_macro_data.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n✅ Feature dataset saved to:")
    print(output_path)