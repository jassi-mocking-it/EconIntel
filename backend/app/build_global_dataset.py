from services.world_bank_service import (
    build_world_bank_global_dataset,
)


def main():
    """
    Build EconIntel's first global country-year dataset.

    This runner is intentionally separate from the U.S.
    monthly early-warning pipeline.
    """

    print("\n" + "=" * 76)
    print("🌐 ECONINTEL GLOBAL SOVEREIGN-RISK DATA PIPELINE")
    print("=" * 76)

    (
        world_bank_long_df,
        country_year_panel,
        missing_data_report,
    ) = build_world_bank_global_dataset()

    print("\nGlobal pipeline summary")
    print("-----------------------")

    print(
        f"Raw observations: "
        f"{len(world_bank_long_df)}"
    )

    print(
        f"Country-year rows: "
        f"{len(country_year_panel)}"
    )

    print(
        f"Missing-data audit rows: "
        f"{len(missing_data_report)}"
    )


if __name__ == "__main__":
    main()