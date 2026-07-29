from pathlib import Path
from typing import Any

import pandas as pd
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.global_settings import (
    GLOBAL_END_YEAR,
    GLOBAL_START_YEAR,
    WORLD_BANK_COUNTRIES,
    WORLD_BANK_INDICATORS,
)


WORLD_BANK_API_BASE = (
    "https://api.worldbank.org/v2"
)


REPOSITORY_DIRECTORY = (
    Path(__file__).resolve().parents[3]
)

GLOBAL_DATA_DIRECTORY = (
    REPOSITORY_DIRECTORY
    / "data"
    / "global"
)

RAW_DATA_DIRECTORY = (
    GLOBAL_DATA_DIRECTORY
    / "raw"
)

PROCESSED_DATA_DIRECTORY = (
    GLOBAL_DATA_DIRECTORY
    / "processed"
)

REPORT_DIRECTORY = (
    GLOBAL_DATA_DIRECTORY
    / "reports"
)


RAW_LONG_DATA_PATH = (
    RAW_DATA_DIRECTORY
    / "world_bank_global_long.csv"
)

COUNTRY_PANEL_PATH = (
    PROCESSED_DATA_DIRECTORY
    / "world_bank_country_year.csv"
)

MISSING_DATA_REPORT_PATH = (
    REPORT_DIRECTORY
    / "world_bank_missing_data.csv"
)


def create_http_session() -> requests.Session:
    """
    Create a requests session with retries for temporary
    API and network failures.
    """

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "EconIntel/1.0 "
                "global-sovereign-risk-project"
            ),
            "Accept": "application/json",
        }
    )

    return session


def validate_world_bank_response(
    payload: Any,
    indicator_code: str,
) -> list[dict]:
    """
    Validate the World Bank API response and return its
    observation records.
    """

    if not isinstance(payload, list):
        raise ValueError(
            "Unexpected World Bank response type for "
            f"{indicator_code}."
        )

    if len(payload) < 2:
        raise ValueError(
            "World Bank response did not contain data for "
            f"{indicator_code}."
        )

    metadata = payload[0]
    records = payload[1]

    if (
        isinstance(metadata, dict)
        and "message" in metadata
    ):
        raise ValueError(
            "World Bank API returned an error for "
            f"{indicator_code}: "
            f"{metadata['message']}"
        )

    if records is None:
        return []

    if not isinstance(records, list):
        raise ValueError(
            "World Bank observation data has an invalid "
            f"structure for {indicator_code}."
        )

    return records


def fetch_world_bank_indicator(
    session: requests.Session,
    indicator_code: str,
    output_name: str,
    start_year: int = GLOBAL_START_YEAR,
    end_year: int = GLOBAL_END_YEAR,
) -> pd.DataFrame:
    """
    Download one World Bank indicator for all configured
    countries and return it in long format.
    """

    country_codes = ";".join(
        WORLD_BANK_COUNTRIES.keys()
    )

    url = (
        f"{WORLD_BANK_API_BASE}"
        f"/country/{country_codes}"
        f"/indicator/{indicator_code}"
    )

    parameters = {
        "format": "json",
        "date": (
            f"{start_year}:{end_year}"
        ),
        "per_page": 20000,
    }

    print(
        f"Downloading {indicator_code:<22} "
        f"as {output_name}..."
    )

    response = session.get(
        url,
        params=parameters,
        timeout=45,
    )

    response.raise_for_status()

    payload = response.json()

    records = validate_world_bank_response(
        payload=payload,
        indicator_code=indicator_code,
    )

    rows = []

    for record in records:
        country_code = record.get(
            "countryiso3code"
        )

        if (
            country_code
            not in WORLD_BANK_COUNTRIES
        ):
            continue

        year_value = record.get(
            "date"
        )

        try:
            year = int(year_value)
        except (TypeError, ValueError):
            continue

        raw_value = record.get(
            "value"
        )

        numeric_value = pd.to_numeric(
            raw_value,
            errors="coerce",
        )

        rows.append(
            {
                "country_code": country_code,
                "country": (
                    WORLD_BANK_COUNTRIES[
                        country_code
                    ]
                ),
                "year": year,
                "indicator_code": indicator_code,
                "indicator": output_name,
                "value": numeric_value,
            }
        )

    indicator_df = pd.DataFrame(
        rows
    )

    if indicator_df.empty:
        print(
            f"⚠️ No observations returned for "
            f"{indicator_code}."
        )
    else:
        valid_values = int(
            indicator_df["value"]
            .notna()
            .sum()
        )

        print(
            f"✅ {len(indicator_df)} rows, "
            f"{valid_values} valid values"
        )

    return indicator_df


def download_world_bank_data(
    start_year: int = GLOBAL_START_YEAR,
    end_year: int = GLOBAL_END_YEAR,
) -> pd.DataFrame:
    """
    Download every configured World Bank indicator.
    """

    print("\n" + "=" * 76)
    print("🌍 DOWNLOADING ECONINTEL WORLD BANK DATA")
    print("=" * 76)

    session = create_http_session()

    indicator_frames = []

    try:
        for (
            indicator_code,
            output_name,
        ) in WORLD_BANK_INDICATORS.items():
            indicator_df = (
                fetch_world_bank_indicator(
                    session=session,
                    indicator_code=(
                        indicator_code
                    ),
                    output_name=output_name,
                    start_year=start_year,
                    end_year=end_year,
                )
            )

            if not indicator_df.empty:
                indicator_frames.append(
                    indicator_df
                )
    finally:
        session.close()

    if not indicator_frames:
        raise RuntimeError(
            "No World Bank indicators were downloaded."
        )

    long_df = pd.concat(
        indicator_frames,
        ignore_index=True,
    )

    long_df = (
        long_df
        .sort_values(
            [
                "country_code",
                "year",
                "indicator",
            ]
        )
        .reset_index(drop=True)
    )

    RAW_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_df.to_csv(
        RAW_LONG_DATA_PATH,
        index=False,
    )

    print(
        f"\n✅ Raw World Bank data saved to:\n"
        f"{RAW_LONG_DATA_PATH}"
    )

    return long_df


def create_country_year_panel(
    long_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the long World Bank observations into one row
    per country and year.
    """

    print("\n" + "=" * 76)
    print("🧩 CREATING WORLD BANK COUNTRY-YEAR PANEL")
    print("=" * 76)

    required_columns = {
        "country_code",
        "country",
        "year",
        "indicator",
        "value",
    }

    missing_columns = (
        required_columns
        - set(long_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "World Bank long data is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    panel_df = (
        long_df
        .pivot_table(
            index=[
                "country_code",
                "country",
                "year",
            ],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    panel_df.columns.name = None

    all_country_years = pd.MultiIndex.from_product(
        [
            list(
                WORLD_BANK_COUNTRIES.keys()
            ),
            range(
                GLOBAL_START_YEAR,
                GLOBAL_END_YEAR + 1,
            ),
        ],
        names=[
            "country_code",
            "year",
        ],
    ).to_frame(
        index=False
    )

    country_lookup = pd.DataFrame(
        {
            "country_code": list(
                WORLD_BANK_COUNTRIES.keys()
            ),
            "country": list(
                WORLD_BANK_COUNTRIES.values()
            ),
        }
    )

    complete_panel = (
        all_country_years
        .merge(
            country_lookup,
            on="country_code",
            how="left",
        )
        .merge(
            panel_df.drop(
                columns=["country"],
                errors="ignore",
            ),
            on=[
                "country_code",
                "year",
            ],
            how="left",
        )
    )

    ordered_indicator_columns = list(
        WORLD_BANK_INDICATORS.values()
    )

    for column in ordered_indicator_columns:
        if column not in complete_panel.columns:
            complete_panel[column] = pd.NA

    complete_panel = complete_panel[
        [
            "country_code",
            "country",
            "year",
            *ordered_indicator_columns,
        ]
    ]

    complete_panel = (
        complete_panel
        .sort_values(
            [
                "country_code",
                "year",
            ]
        )
        .reset_index(drop=True)
    )

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_panel.to_csv(
        COUNTRY_PANEL_PATH,
        index=False,
    )

    print(
        f"Countries: "
        f"{complete_panel['country_code'].nunique()}"
    )

    print(
        f"Years: "
        f"{complete_panel['year'].min()}–"
        f"{complete_panel['year'].max()}"
    )

    print(
        f"Panel rows: "
        f"{len(complete_panel)}"
    )

    print(
        f"Indicators: "
        f"{len(ordered_indicator_columns)}"
    )

    print(
        f"\n✅ Country-year panel saved to:\n"
        f"{COUNTRY_PANEL_PATH}"
    )

    return complete_panel


def create_missing_data_report(
    panel_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing-data coverage by indicator and
    country.
    """

    print("\n" + "=" * 76)
    print("🔍 AUDITING GLOBAL MISSING DATA")
    print("=" * 76)

    indicator_columns = list(
        WORLD_BANK_INDICATORS.values()
    )

    report_rows = []

    for country_code, group in panel_df.groupby(
        "country_code"
    ):
        country_name = (
            group["country"].iloc[0]
        )

        for indicator in indicator_columns:
            missing_count = int(
                group[indicator]
                .isna()
                .sum()
            )

            total_count = int(
                len(group)
            )

            valid_count = (
                total_count
                - missing_count
            )

            coverage_percentage = (
                valid_count
                / total_count
                * 100
                if total_count
                else 0.0
            )

            report_rows.append(
                {
                    "country_code": (
                        country_code
                    ),
                    "country": country_name,
                    "indicator": indicator,
                    "valid_values": valid_count,
                    "missing_values": (
                        missing_count
                    ),
                    "coverage_percentage": (
                        coverage_percentage
                    ),
                }
            )

    report_df = pd.DataFrame(
        report_rows
    )

    report_df = (
        report_df
        .sort_values(
            [
                "coverage_percentage",
                "country_code",
                "indicator",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_df.to_csv(
        MISSING_DATA_REPORT_PATH,
        index=False,
    )

    print("\nLowest-coverage country indicators")
    print("----------------------------------")

    print(
        report_df.head(20).to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.1f}"
            ),
        )
    )

    print(
        f"\n✅ Missing-data report saved to:\n"
        f"{MISSING_DATA_REPORT_PATH}"
    )

    return report_df


def build_world_bank_global_dataset():
    """
    Run the complete first-stage World Bank pipeline.
    """

    long_df = download_world_bank_data()

    panel_df = create_country_year_panel(
        long_df
    )

    missing_report = (
        create_missing_data_report(
            panel_df
        )
    )

    print("\n" + "=" * 76)
    print("✅ ECONINTEL GLOBAL DATA FOUNDATION COMPLETED")
    print("=" * 76)

    return (
        long_df,
        panel_df,
        missing_report,
    )