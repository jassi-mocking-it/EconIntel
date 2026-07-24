import pandas as pd


FORECAST_HORIZON_MONTHS = 3
RISING_STRESS_THRESHOLD = 5.0


def create_prediction_targets(df):
    """
    Create EconIntel's three-month forecasting targets.

    Targets:
    1. Stress level three months ahead.
    2. Change in stress over three months.
    3. Whether stress rises by at least five points.
    4. Whether a crisis occurs three months ahead.
    """

    print("\n🎯 Creating EconIntel prediction targets...")

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Future stress level
    # ---------------------------------------------------------

    data["TARGET_STRESS_3M"] = (
        data["ECON_STRESS"]
        .shift(-FORECAST_HORIZON_MONTHS)
    )

    # ---------------------------------------------------------
    # Future stress change
    # ---------------------------------------------------------

    data["TARGET_STRESS_CHANGE_3M"] = (
        data["TARGET_STRESS_3M"]
        - data["ECON_STRESS"]
    )

    # ---------------------------------------------------------
    # Rising-risk classification target
    # ---------------------------------------------------------

    # Start every row as unknown.
    # The final three months have no known future outcome.
    data["TARGET_RISK_RISING_3M"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64",
    )

    valid_target_mask = (
        data["TARGET_STRESS_CHANGE_3M"]
        .notna()
    )

    data.loc[
        valid_target_mask,
        "TARGET_RISK_RISING_3M",
    ] = (
        data.loc[
            valid_target_mask,
            "TARGET_STRESS_CHANGE_3M",
        ]
        >= RISING_STRESS_THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # Future crisis target
    # ---------------------------------------------------------

    data["TARGET_CRISIS_3M"] = (
        data["CRISIS"]
        .shift(-FORECAST_HORIZON_MONTHS)
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    valid_changes = (
        data["TARGET_STRESS_CHANGE_3M"]
        .dropna()
    )

    rising_count = int(
        (
            data["TARGET_RISK_RISING_3M"] == 1
        ).sum()
    )

    stable_or_falling_count = int(
        (
            data["TARGET_RISK_RISING_3M"] == 0
        ).sum()
    )

    print("✅ Prediction targets created!")

    print(
        f"Valid three-month changes: "
        f"{len(valid_changes)}"
    )

    print(
        f"Average three-month change: "
        f"{valid_changes.mean():.3f}"
    )

    print(
        f"Rising-risk observations: "
        f"{rising_count}"
    )

    print(
        f"Stable/falling observations: "
        f"{stable_or_falling_count}"
    )

    return data