import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_stress_index(df):
    """
    Create a simple Economic Stress Index (ESI)
    using normalized macroeconomic indicators.
    """

    print("\n📈 Calculating Economic Stress Index...")

    dataset = df.copy()

    # -----------------------------
    # Features we'll use
    # -----------------------------
    features = [
        "FEDFUNDS",
        "UNRATE",
        "INFLATION_RATE",
        "GDP_GROWTH",
        "VIX",
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(dataset[features])

    scaled_df = pd.DataFrame(
        scaled,
        columns=features,
        index=dataset.index
    )

    # -----------------------------
    # Weighted Stress Formula
    # -----------------------------
    dataset["ECON_STRESS"] = (
        0.20 * scaled_df["FEDFUNDS"]
        + 0.25 * scaled_df["UNRATE"]
        + 0.25 * scaled_df["INFLATION_RATE"]
        - 0.20 * scaled_df["GDP_GROWTH"]
        + 0.10 * scaled_df["VIX"]
    )

    # -----------------------------
    # Convert to 0-100
    # -----------------------------
    minimum = dataset["ECON_STRESS"].min()
    maximum = dataset["ECON_STRESS"].max()

    dataset["ECON_STRESS"] = (
        (dataset["ECON_STRESS"] - minimum)
        / (maximum - minimum)
    ) * 100

    print("✅ Economic Stress Index created!")

    return dataset
import os


def save_stress_dataset(df):

    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../"
        )
    )

    output_dir = os.path.join(BASE_DIR, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "stress_macro_data.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"\n✅ Stress dataset saved to:\n{output_path}")