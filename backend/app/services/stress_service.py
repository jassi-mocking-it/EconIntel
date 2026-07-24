from pathlib import Path

import numpy as np
import pandas as pd


def expanding_z_score(series, minimum_periods=24):
    """
    Calculate a causal expanding z-score.

    Each row is standardized using only the observations
    available up to that date.
    """

    expanding_mean = series.expanding(
        min_periods=minimum_periods
    ).mean()

    expanding_std = series.expanding(
        min_periods=minimum_periods
    ).std()

    z_score = (
        series - expanding_mean
    ) / expanding_std.replace(0, np.nan)

    return z_score


def calculate_stress_index(df):
    """
    Calculate EconIntel's causal Economic Stress Index.

    The score uses only current and historical information,
    preventing future-data leakage.
    """

    print("\n📈 Calculating causal Economic Stress Index...")

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    # ================================================
    # Causal standardization
    # ================================================

    data["Z_FEDFUNDS"] = expanding_z_score(
        data["FEDFUNDS"]
    )

    data["Z_UNRATE"] = expanding_z_score(
        data["UNRATE"]
    )

    data["Z_INFLATION"] = expanding_z_score(
        data["INFLATION_RATE"]
    )

    data["Z_GDP_GROWTH"] = expanding_z_score(
        data["GDP_GROWTH"]
    )

    data["Z_VIX"] = expanding_z_score(
        data["VIX"]
    )

    # ================================================
    # Transparent weighted stress formula
    # ================================================

    data["RAW_ECON_STRESS"] = (
        0.15 * data["Z_FEDFUNDS"]
        + 0.25 * data["Z_UNRATE"]
        + 0.25 * data["Z_INFLATION"]
        - 0.20 * data["Z_GDP_GROWTH"]
        + 0.15 * data["Z_VIX"]
    )

    # Convert the unbounded score into a stable 0–100 range.
    # The logistic transformation avoids using future min/max.
    data["ECON_STRESS"] = (
        100
        / (
            1
            + np.exp(
                -data["RAW_ECON_STRESS"]
            )
        )
    )
        # ================================================
    # Stress trend and early-warning features
    # ================================================

    # Change during the latest month.
    data["STRESS_CHANGE_1M"] = (
        data["ECON_STRESS"].diff(1)
    )

    # Change during the latest three months.
    data["STRESS_CHANGE_3M"] = (
        data["ECON_STRESS"].diff(3)
    )

    # Recent average stress level.
    data["STRESS_3M_AVG"] = (
        data["ECON_STRESS"]
        .rolling(window=3)
        .mean()
    )

    # Whether present stress is above or below
    # its recent three-month average.
    data["STRESS_GAP_3M"] = (
        data["ECON_STRESS"]
        - data["STRESS_3M_AVG"]
    )

    # Measures instability in the stress index.
    data["STRESS_VOLATILITY_6M"] = (
        data["ECON_STRESS"]
        .rolling(window=6)
        .std()
    )

    # Change in monthly stress momentum.
    data["STRESS_ACCELERATION"] = (
        data["STRESS_CHANGE_1M"].diff(1)
    )

    print("✅ Causal Economic Stress Index created!")

    valid_scores = data["ECON_STRESS"].dropna()

    if not valid_scores.empty:
        print(
            f"Valid stress observations: {len(valid_scores)}"
        )
        print(
            f"Stress range: "
            f"{valid_scores.min():.2f}–"
            f"{valid_scores.max():.2f}"
        )

    return data


def save_stress_dataset(df):
    """
    Save the stress-index dataset.
    """

    models_root = (
        Path(__file__).resolve().parents[3]
    )

    output_directory = (
        models_root
        / "data"
        / "processed"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "stress_macro_data.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\n✅ Stress dataset saved to:\n{output_path}"
    )