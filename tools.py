"""Weather tools that the LLM can call. Both query the OpenWeather REST API."""

import os
from collections import Counter, defaultdict
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"
TIMEOUT = 10  # seconds


def _clean_city(city: str) -> str:
    """Basic input cleanup so only a short place name reaches the API."""
    city = (city or "").strip()
    if not city or len(city) > 80:
        raise ValueError("Please provide a valid city name.")
    return city


def _get(endpoint: str, city: str) -> dict:
    """Call an OpenWeather endpoint and return the JSON response."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is missing from .env")
    resp = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "imperial"},
        timeout=TIMEOUT,
    )
    if resp.status_code == 404:
        raise ValueError(f"City '{city}' was not found.")
    resp.raise_for_status()
    return resp.json()


@tool
def get_current_weather(city: str) -> dict:
    """Get the current weather for a city. Input: a city name, e.g. 'Amarillo' or 'Amarillo,US'."""
    try:
        city = _clean_city(city)
        data = _get("weather", city)
        return {
            "city": f"{data['name']}, {data['sys'].get('country', '')}",
            "temperature_f": round(data["main"]["temp"]),
            "feels_like_f": round(data["main"]["feels_like"]),
            "humidity_pct": data["main"]["humidity"],
            "wind_speed_mph": round(data["wind"]["speed"]),
            "condition": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_weather_forecast(city: str) -> dict:
    """Get a 5-day weather forecast for a city. Input: a city name, e.g. 'Amarillo' or 'Amarillo,US'."""
    try:
        city = _clean_city(city)
        data = _get("forecast", city)

        # OpenWeather returns 3-hour blocks; group them by day.
        days = defaultdict(lambda: {"temps": [], "conditions": []})
        for item in data["list"]:
            day = datetime.fromtimestamp(item["dt"]).strftime("%a %b %d")
            days[day]["temps"].append(item["main"]["temp"])
            days[day]["conditions"].append(item["weather"][0]["description"])

        forecast = []
        for day, info in list(days.items())[:5]:
            forecast.append({
                "date": day,
                "high_f": round(max(info["temps"])),
                "low_f": round(min(info["temps"])),
                "condition": Counter(info["conditions"]).most_common(1)[0][0].title(),
            })

        return {
            "city": f"{data['city']['name']}, {data['city'].get('country', '')}",
            "forecast": forecast,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Quick test: python tools.py
    print(get_current_weather.invoke({"city": "Amarillo"}))
    print(get_weather_forecast.invoke({"city": "Amarillo"}))
