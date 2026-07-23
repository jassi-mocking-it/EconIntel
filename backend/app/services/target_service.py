import pandas as pd


def create_prediction_targets(df):
    """
    Create future prediction targets.
    """

    print("\n🎯 Creating prediction targets...")

    df = df.copy()

    # Predict stress 3 months ahead
    df["TARGET_STRESS_3M"] = df["ECON_STRESS"].shift(-3)

    # Binary crisis prediction
    df["TARGET_CRISIS_3M"] = df["CRISIS"].shift(-3)

    print("✅ Prediction targets created!")

    return df