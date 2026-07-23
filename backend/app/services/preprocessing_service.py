import os

import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_dataset():
    """
    Load the merged macroeconomic dataset.
    """

    path = os.path.join(PROCESSED_DIR, "merged_macro_data.csv")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    print("✅ Dataset loaded successfully!")

    return df


def inspect_missing_values(df):
    """
    Display missing-value counts for every column.
    """

    print("\n📊 Missing Values:\n")
    print(df.isna().sum())


def clean_dataset(df):
    """
    Convert mixed-frequency data into a monthly dataset.

    Monthly indicators use their latest monthly observation.
    GDP is carried forward between quarterly observations.
    Daily Treasury and VIX observations are converted to monthly averages.
    Data before a series begins is not backward-filled.
    """

    print("\n🧹 Cleaning and standardizing dataset to monthly frequency...")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")

    # Create a continuous monthly index.
    monthly_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="MS",
    )

    monthly = pd.DataFrame(index=monthly_index)
    monthly.index.name = "date"

    # Monthly indicators.
    for column in ["FEDFUNDS", "UNRATE", "CPI"]:
        monthly[column] = data[column].resample("MS").last()

    # Quarterly GDP: carry the latest observation forward.
    monthly["GDP"] = data["GDP"].resample("MS").last()
    monthly["GDP"] = monthly["GDP"].ffill()

    # Daily financial indicators: monthly averages.
    for column in ["TREASURY", "VIX"]:
        monthly[column] = data[column].resample("MS").mean()

    # Fill internal gaps only after each series actually begins.
    for column in monthly.columns:
        first_valid_date = monthly[column].first_valid_index()

        if first_valid_date is not None:
            monthly.loc[first_valid_date:, column] = (
                monthly.loc[first_valid_date:, column].ffill()
            )

    # Keep only the common period where every required series exists.
    monthly = monthly.dropna().reset_index()

    print("✅ Monthly dataset created!")
    print(f"Rows: {len(monthly)}")
    print(
        f"Date range: {monthly['date'].min().date()} "
        f"to {monthly['date'].max().date()}"
    )

    return monthly


def save_clean_dataset(df):
    """
    Save the monthly cleaned dataset.
    """

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "clean_macro_data.csv",
    )

    df.to_csv(output_path, index=False)

    print(f"\n✅ Clean dataset saved to:\n{output_path}")


def save_feature_dataset(df):
    """
    Save the engineered feature dataset.
    """

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "feature_macro_data.csv",
    )

    df.to_csv(output_path, index=False)

    print(f"\n✅ Feature dataset saved to:\n{output_path}")