import pandas as pd


def create_features(df):
    """
    Create EconIntel's macroeconomic forecasting features.

    The input dataset must contain one row per month.
    """

    print("\n📊 Creating engineered features...")

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    # Ensure macroeconomic columns are numeric.
    numeric_columns = [
        "FEDFUNDS",
        "UNRATE",
        "CPI",
        "GDP",
        "TREASURY",
        "VIX",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # =================================================
    # 1. Monthly changes
    # =================================================

    data["FED_CHANGE"] = data["FEDFUNDS"].diff()

    data["TREASURY_CHANGE"] = data["TREASURY"].diff()

    data["VIX_CHANGE"] = data["VIX"].diff()

    data["UNRATE_CHANGE"] = data["UNRATE"].diff()

    # =================================================
    # 2. Year-over-year macroeconomic growth
    # =================================================

    # CPI percentage change compared with 12 months earlier.
    data["INFLATION_RATE"] = (
        data["CPI"].pct_change(
            periods=12,
            fill_method=None,
        )
        * 100
    )

    # GDP percentage change compared with 12 months earlier.
    data["GDP_GROWTH"] = (
        data["GDP"].pct_change(
            periods=12,
            fill_method=None,
        )
        * 100
    )

    # =================================================
    # 3. Rolling averages
    # =================================================

    data["FED_3M_AVG"] = (
        data["FEDFUNDS"]
        .rolling(window=3)
        .mean()
    )

    data["FED_6M_AVG"] = (
        data["FEDFUNDS"]
        .rolling(window=6)
        .mean()
    )

    data["CPI_3M_AVG"] = (
        data["CPI"]
        .rolling(window=3)
        .mean()
    )

    data["GDP_3M_AVG"] = (
        data["GDP"]
        .rolling(window=3)
        .mean()
    )

    data["UNRATE_3M_AVG"] = (
        data["UNRATE"]
        .rolling(window=3)
        .mean()
    )

    data["VIX_3M_AVG"] = (
        data["VIX"]
        .rolling(window=3)
        .mean()
    )

    # =================================================
    # 4. Lag features
    # =================================================

    data["FED_LAG1"] = data["FEDFUNDS"].shift(1)

    data["GDP_LAG1"] = data["GDP"].shift(1)

    data["CPI_LAG1"] = data["CPI"].shift(1)

    data["UNRATE_LAG1"] = data["UNRATE"].shift(1)

    data["TREASURY_LAG1"] = data["TREASURY"].shift(1)

    data["VIX_LAG1"] = data["VIX"].shift(1)

    # =================================================
    # 5. Six-month volatility
    # =================================================

    data["VIX_VOLATILITY"] = (
        data["VIX"]
        .rolling(window=6)
        .std()
    )

    data["FED_VOLATILITY"] = (
        data["FEDFUNDS"]
        .rolling(window=6)
        .std()
    )

    # =================================================
    # 6. Momentum
    # =================================================

    data["FED_MOMENTUM"] = (
        data["FEDFUNDS"]
        - data["FED_3M_AVG"]
    )

    data["GDP_MOMENTUM"] = (
        data["GDP"]
        - data["GDP_3M_AVG"]
    )

    data["UNRATE_MOMENTUM"] = (
        data["UNRATE"]
        - data["UNRATE_3M_AVG"]
    )

    data["VIX_MOMENTUM"] = (
        data["VIX"]
        - data["VIX_3M_AVG"]
    )

    print("✅ Features created!")

    return data