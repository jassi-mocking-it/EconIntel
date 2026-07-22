import os
import pandas as pd

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

RAW_DATA = os.path.join(BASE_DIR, "data", "raw")


def load_indicator(filename, column_name):

    path = os.path.join(RAW_DATA, f"{filename}.csv")

    df = pd.read_csv(path)

    df = df[["date", "value"]]

    df.rename(columns={"value": column_name}, inplace=True)

    return df


def build_master_dataset():

    fed = load_indicator("FEDFUNDS", "FEDFUNDS")
    unemployment = load_indicator("UNRATE", "UNRATE")
    cpi = load_indicator("CPIAUCSL", "CPI")
    gdp = load_indicator("GDP", "GDP")
    treasury = load_indicator("DGS10", "TREASURY")
    vix = load_indicator("VIXCLS", "VIX")

    master = fed.merge(unemployment, on="date", how="outer")
    master = master.merge(cpi, on="date", how="outer")
    master = master.merge(gdp, on="date", how="outer")
    master = master.merge(treasury, on="date", how="outer")
    master = master.merge(vix, on="date", how="outer")

    master.sort_values("date", inplace=True)

    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "merged_macro_data.csv")

    master.to_csv(output_path, index=False)

    print(f"\n✅ Saved merged dataset to:\n{output_path}")

    return master