import pandas as pd


def create_training_dataset(df):
    """
    Prepare an ML-ready EconIntel dataset.

    Target columns must already have been created by
    target_service.py.
    """

    print("\n🧠 Preparing ML dataset...")

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    required_targets = [
        "TARGET_STRESS_3M",
        "TARGET_STRESS_CHANGE_3M",
        "TARGET_RISK_RISING_3M",
    ]

    missing_columns = [
        column
        for column in required_targets
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing target columns: "
            + ", ".join(missing_columns)
        )

    # Remove observations whose future outcome is unknown.
    data = data.dropna(
        subset=[
            "TARGET_STRESS_CHANGE_3M",
            "TARGET_RISK_RISING_3M",
        ]
    )

    data = data.sort_values("date").reset_index(drop=True)

    print("✅ Training dataset created!")
    print(f"Training observations: {len(data)}")

    return data