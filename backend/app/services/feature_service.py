import pandas as pd


def create_features(df):
    """
    Create engineered macroeconomic features.
    """

    print("\n Creating engineered features...")

    # -----------------------------
    # Interest Rate Features
    # -----------------------------

    df["FED_CHANGE"] = df["FEDFUNDS"].diff()

    # -----------------------------
    # Inflation Features
    # -----------------------------

    df["INFLATION_RATE"] = df["CPI"].pct_change() * 100

    # -----------------------------
    # GDP Growth
    # -----------------------------

    df["GDP_GROWTH"] = df["GDP"].pct_change() * 100

    # -----------------------------
    # Treasury Change
    # -----------------------------

    df["TREASURY_CHANGE"] = df["TREASURY"].diff()

    # -----------------------------
    # VIX Change
    # -----------------------------

    df["VIX_CHANGE"] = df["VIX"].diff()

    # -----------------------------
    # Unemployment Change
    # -----------------------------

    df["UNRATE_CHANGE"] = df["UNRATE"].diff()

    # -----------------------------
# Rolling Averages
# -----------------------------

    df["FED_3M_AVG"] = df["FEDFUNDS"].rolling(window=3).mean()
    df["FED_6M_AVG"] = df["FEDFUNDS"].rolling(window=6).mean()

    df["CPI_3M_AVG"] = df["CPI"].rolling(window=3).mean()

    df["VIX_3M_AVG"] = df["VIX"].rolling(window=3).mean()

    df["UNRATE_3M_AVG"] = df["UNRATE"].rolling(window=3).mean()
    # -----------------------------
# Lag Features
# -----------------------------

    df["FED_LAG1"] = df["FEDFUNDS"].shift(1)

    df["GDP_LAG1"] = df["GDP"].shift(1)

    df["UNRATE_LAG1"] = df["UNRATE"].shift(1)

    df["VIX_LAG1"] = df["VIX"].shift(1)


# ------------------------------------
# Rolling Averages
# ------------------------------------

    df["FED_3M_AVG"] = df["FEDFUNDS"].rolling(3).mean()
    df["FED_6M_AVG"] = df["FEDFUNDS"].rolling(6).mean()

    df["GDP_3M_AVG"] = df["GDP"].rolling(3).mean()

    df["CPI_3M_AVG"] = df["CPI"].rolling(3).mean()

    df["UNRATE_3M_AVG"] = df["UNRATE"].rolling(3).mean()

    df["VIX_3M_AVG"] = df["VIX"].rolling(3).mean()

# ------------------------------------
# Lag Features
# ------------------------------------

    df["FED_LAG1"] = df["FEDFUNDS"].shift(1)
    df["GDP_LAG1"] = df["GDP"].shift(1)
    df["CPI_LAG1"] = df["CPI"].shift(1)
    df["UNRATE_LAG1"] = df["UNRATE"].shift(1)
    df["TREASURY_LAG1"] = df["TREASURY"].shift(1)
    df["VIX_LAG1"] = df["VIX"].shift(1)

    # ------------------------------------
# Rolling Standard Deviation
# ------------------------------------

    df["VIX_VOLATILITY"] = df["VIX"].rolling(6).std()

    df["FED_VOLATILITY"] = df["FEDFUNDS"].rolling(6).std()
    # ------------------------------------
# Momentum Features
# ------------------------------------

    df["FED_MOMENTUM"] = df["FEDFUNDS"] - df["FED_3M_AVG"]

    df["GDP_MOMENTUM"] = df["GDP"] - df["GDP_3M_AVG"]

    df["UNRATE_MOMENTUM"] = df["UNRATE"] - df["UNRATE_3M_AVG"]

    df["VIX_MOMENTUM"] = df["VIX"] - df["VIX_3M_AVG"]
    return df
    print("✅ Features created!")