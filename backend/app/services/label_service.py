import pandas as pd

from config.crisis_periods import CRISIS_PERIODS


def create_crisis_labels(df):
    """
    Add crisis labels to the dataset.
    """

    print("\n🏷️ Creating crisis labels...")

    df = df.copy()

    df["CRISIS"] = 0
    df["CRISIS_NAME"] = "None"

    df["date"] = pd.to_datetime(df["date"])

    for start, end, name in CRISIS_PERIODS:

        mask = (
            (df["date"] >= pd.to_datetime(start))
            &
            (df["date"] <= pd.to_datetime(end))
        )

        df.loc[mask, "CRISIS"] = 1
        df.loc[mask, "CRISIS_NAME"] = name

    print("✅ Crisis labels created!")

    return df