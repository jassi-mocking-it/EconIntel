from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET_RISK_RISING_3M"

EXCLUDED_COLUMNS = [
    "date",

    # Future targets
    "TARGET_STRESS_3M",
    "TARGET_STRESS_CHANGE_3M",
    "TARGET_RISK_RISING_3M",
    "TARGET_CRISIS_3M",

    # Historical labels
    "CRISIS",
    "CRISIS_NAME",

    # Internal stress-index representation
    "RAW_ECON_STRESS",
]


REPORT_DIRECTORY = (
    Path(__file__).resolve().parent.parent
    / "reports"
)

FEATURE_AUDIT_PATH = (
    REPORT_DIRECTORY
    / "feature_audit.csv"
)

CORRELATION_PATH = (
    REPORT_DIRECTORY
    / "high_feature_correlations.csv"
)


def prepare_feature_data(df):
    """
    Prepare numerical model features for auditing.
    """

    data = df.copy()

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"]
        )

        data = (
            data
            .sort_values("date")
            .reset_index(drop=True)
        )

    X = data.drop(
        columns=EXCLUDED_COLUMNS,
        errors="ignore",
    )

    X = X.select_dtypes(
        include=["number"]
    )

    return X


def create_feature_summary(X):
    """
    Calculate basic quality information for every feature.
    """

    summary_rows = []

    total_rows = len(X)

    for feature in X.columns:
        series = X[feature]

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        variance = float(
            series.var(skipna=True)
        )

        summary_rows.append(
            {
                "Feature": feature,
                "Missing Count": missing_count,
                "Missing Percentage": (
                    missing_percentage
                ),
                "Unique Values": unique_count,
                "Variance": variance,
                "Mean": float(
                    series.mean(skipna=True)
                ),
                "Standard Deviation": float(
                    series.std(skipna=True)
                ),
                "Minimum": float(
                    series.min(skipna=True)
                ),
                "Maximum": float(
                    series.max(skipna=True)
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        [
            "Missing Percentage",
            "Feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    return summary_df


def find_high_correlations(
    X,
    threshold=0.90,
):
    """
    Find feature pairs with a large absolute correlation.

    Correlation does not automatically mean a feature must
    be removed. It identifies pairs that require review.
    """

    correlation_matrix = X.corr(
        method="pearson"
    )

    correlation_pairs = []

    columns = list(
        correlation_matrix.columns
    )

    for first_index in range(
        len(columns)
    ):
        for second_index in range(
            first_index + 1,
            len(columns),
        ):
            first_feature = columns[
                first_index
            ]

            second_feature = columns[
                second_index
            ]

            correlation = correlation_matrix.loc[
                first_feature,
                second_feature,
            ]

            if pd.isna(correlation):
                continue

            absolute_correlation = abs(
                correlation
            )

            if absolute_correlation >= threshold:
                correlation_pairs.append(
                    {
                        "Feature 1": (
                            first_feature
                        ),
                        "Feature 2": (
                            second_feature
                        ),
                        "Correlation": float(
                            correlation
                        ),
                        "Absolute Correlation": float(
                            absolute_correlation
                        ),
                    }
                )

    correlation_df = pd.DataFrame(
        correlation_pairs
    )

    if not correlation_df.empty:
        correlation_df = (
            correlation_df
            .sort_values(
                "Absolute Correlation",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return correlation_df


def find_constant_features(X):
    """
    Find features containing zero or one unique value.
    """

    constant_features = []

    for feature in X.columns:
        unique_count = X[
            feature
        ].nunique(
            dropna=True
        )

        if unique_count <= 1:
            constant_features.append(
                feature
            )

    return constant_features


def run_feature_audit(
    df,
    correlation_threshold=0.90,
):
    """
    Audit EconIntel model features without changing them.
    """

    print("\n" + "=" * 72)
    print("🧹 ECONINTEL FEATURE AUDIT")
    print("=" * 72)

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = prepare_feature_data(
        df
    )

    feature_summary = (
        create_feature_summary(
            X
        )
    )

    high_correlations = (
        find_high_correlations(
            X,
            threshold=correlation_threshold,
        )
    )

    constant_features = (
        find_constant_features(
            X
        )
    )

    feature_summary.to_csv(
        FEATURE_AUDIT_PATH,
        index=False,
    )

    high_correlations.to_csv(
        CORRELATION_PATH,
        index=False,
    )

    print(
        f"Model features audited: "
        f"{len(X.columns)}"
    )

    print(
        f"Observations inspected: "
        f"{len(X)}"
    )

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Highly correlated pairs "
        f"(|correlation| >= "
        f"{correlation_threshold:.2f}): "
        f"{len(high_correlations)}"
    )

    if constant_features:
        print("\nConstant features")
        print("-----------------")

        for feature in constant_features:
            print(
                f"- {feature}"
            )

    print("\nFeatures with the most missing values")
    print("-------------------------------------")

    missing_preview = (
        feature_summary[
            [
                "Feature",
                "Missing Count",
                "Missing Percentage",
            ]
        ]
        .head(10)
    )

    print(
        missing_preview.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print("\nStrongest feature correlations")
    print("------------------------------")

    if high_correlations.empty:
        print(
            "No feature pairs exceeded "
            "the selected threshold."
        )
    else:
        print(
            high_correlations
            .head(20)
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

    print(
        f"\nFeature summary saved to: "
        f"{FEATURE_AUDIT_PATH}"
    )

    print(
        f"Correlation report saved to: "
        f"{CORRELATION_PATH}"
    )

    return {
        "feature_summary": (
            feature_summary
        ),
        "high_correlations": (
            high_correlations
        ),
        "constant_features": (
            constant_features
        ),
        "feature_count": int(
            len(X.columns)
        ),
    }