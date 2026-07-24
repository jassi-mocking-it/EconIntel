EARLY_WARNING_FEATURES = [
    # Current economic stress state
    "ECON_STRESS",

    # Stress direction and instability
    "STRESS_CHANGE_1M",
    "STRESS_CHANGE_3M",
    "STRESS_GAP_3M",
    "STRESS_VOLATILITY_6M",
    "STRESS_ACCELERATION",

    # Inflation and economic growth
    "INFLATION_RATE",
    "GDP_GROWTH",
    "GDP_MOMENTUM",

    # Labour market
    "UNRATE",
    "UNRATE_CHANGE",
    "UNRATE_MOMENTUM",

    # Monetary policy
    "FEDFUNDS",
    "FED_CHANGE",
    "FED_MOMENTUM",

    # Treasury conditions
    "TREASURY",
    "TREASURY_CHANGE",

    # Financial-market stress
    "VIX",
    "VIX_CHANGE",
    "VIX_VOLATILITY",
    "VIX_MOMENTUM",
]
FEATURE_GROUPS = {
    "Labour Market": [
        "UNRATE",
        "UNRATE_CHANGE",
        "UNRATE_3M_AVG",
        "UNRATE_LAG1",
        "UNRATE_MOMENTUM",
        "Z_UNRATE",
    ],

    "Inflation": [
        "CPI",
        "CPI_3M_AVG",
        "CPI_LAG1",
        "INFLATION_RATE",
        "Z_INFLATION",
    ],

    "Economic Growth": [
        "GDP",
        "GDP_3M_AVG",
        "GDP_LAG1",
        "GDP_GROWTH",
        "GDP_MOMENTUM",
        "Z_GDP_GROWTH",
    ],

    "Monetary Policy": [
        "FEDFUNDS",
        "FED_CHANGE",
        "FED_3M_AVG",
        "FED_6M_AVG",
        "FED_LAG1",
        "FED_VOLATILITY",
        "FED_MOMENTUM",
        "Z_FEDFUNDS",
    ],

    "Treasury Conditions": [
        "TREASURY",
        "TREASURY_CHANGE",
        "TREASURY_LAG1",
    ],

    "Market Volatility": [
        "VIX",
        "VIX_CHANGE",
        "VIX_3M_AVG",
        "VIX_LAG1",
        "VIX_VOLATILITY",
        "VIX_MOMENTUM",
        "Z_VIX",
    ],

    "Stress Dynamics": [
        "ECON_STRESS",
        "STRESS_CHANGE_1M",
        "STRESS_CHANGE_3M",
        "STRESS_3M_AVG",
        "STRESS_GAP_3M",
        "STRESS_VOLATILITY_6M",
        "STRESS_ACCELERATION",
    ],
}