import pandas as pd


def create_training_dataset(df):
    """
    Create ML-ready dataset by predicting
    future economic stress.
    """

    print("\n🧠 Preparing ML dataset...")

    data = df.copy()

    # Target = Stress Index 3 months ahead
    data["TARGET_STRESS_3M"] = data["ECON_STRESS"].shift(-3)

    # Remove rows where future target doesn't exist
    data = data.dropna(subset=["TARGET_STRESS_3M"])

    print("✅ Training dataset created!")

    return data