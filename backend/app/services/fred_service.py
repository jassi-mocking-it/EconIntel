import requests
from config.settings import FRED_API_KEY

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_series(series_id):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    return clean_observations(response.json())


def clean_observations(raw_data):
    cleaned = []

    for observation in raw_data["observations"]:

        if observation["value"] == ".":
            continue

        cleaned.append({
            "date": observation["date"],
            "value": float(observation["value"])
        })

    return cleaned