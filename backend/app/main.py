from config.settings import FRED_API_KEY

print("EconIntel Backend Started")

if FRED_API_KEY:
    print("FRED API Key Loaded Successfully")
else:
    print("FRED API Key Not Found")