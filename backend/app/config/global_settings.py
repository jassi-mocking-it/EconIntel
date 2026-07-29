"""
Configuration for EconIntel's global sovereign-risk module.
"""


GLOBAL_START_YEAR = 2000
GLOBAL_END_YEAR = 2025


WORLD_BANK_COUNTRIES = {
    "IND": "India",
    "PAK": "Pakistan",
    "LKA": "Sri Lanka",
    "ARG": "Argentina",
    "EGY": "Egypt",
    "USA": "United States",
}


WORLD_BANK_INDICATORS = {
    # Economic growth
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",

    # Country development level
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",

    # Consumer-price inflation
    "FP.CPI.TOTL.ZG": "inflation",

    # Labour-market conditions
    "SL.UEM.TOTL.ZS": "unemployment",

    # Current-account position as a percentage of GDP
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",

    # Import-cover provided by foreign-exchange reserves
    "FI.RES.TOTL.MO": "reserves_months_imports",

    # Local-currency units per U.S. dollar
    "PA.NUS.FCRF": "official_exchange_rate",

    # Total trade as a percentage of GDP
    "NE.TRD.GNFS.ZS": "trade_gdp",

    # Foreign direct investment inflows as a percentage of GDP
    "BX.KLT.DINV.WD.GD.ZS": "fdi_inflows_gdp",

    # External debt stocks as a percentage of GNI
    "DT.DOD.DECT.GN.ZS": "external_debt_gni",
}