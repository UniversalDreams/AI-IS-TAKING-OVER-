import requests
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv

# --- Configuration Constants (Modify if needed) ---
TICKER = "NVDA"
MULTIPLIER = 1      # 1-minute bars
TIMESPAN = "minute"
# Adjusted date range to ensure total bar count is <= 50,000
# (Approx. 8 months of trading days should fit)
START_DATE = "2024-04-01"
END_DATE = "2025-01-01"
OUTPUT_DIR = "data"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, f"{TICKER}_historical.csv")
MAX_LIMIT = 50000


def fetch_polygon_data_and_save():
    """Fetches Polygon.io aggregate data and saves it to a local CSV."""

    # 1. Load envs variables from.env file
    load_dotenv()

    # 2. Retrieve API Key
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

    if not POLYGON_API_KEY:
        print("FATAL ERROR: POLYGON_API_KEY not found. Ensure it is set in your.env file.")
        return

    print(f"Starting fetch for {TICKER} ({MULTIPLIER}-{TIMESPAN} bars) from {START_DATE} to {END_DATE}...")

    # Polygon Aggregates API Endpoint
    url = f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/{MULTIPLIER}/{TIMESPAN}/{START_DATE}/{END_DATE}"

    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": POLYGON_API_KEY,
        "limit": MAX_LIMIT
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if not data.get('results'):
            print("No data returned from Polygon.io. Check ticker and date range.")
            return

        # Convert results list to DataFrame
        df = pd.DataFrame(data['results'])

        # Rename columns: 'c' is close price, 't' is Unix timestamp
        df.rename(columns={'c': 'close', 'v': 'volume', 't': 'timestamp'}, inplace=True)

        # Convert Unix timestamp (ms) to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Save the processed DataFrame to a local file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(OUTPUT_FILE_PATH)
        print(f"\n--- SUCCESS ---")
        print(f"Successfully fetched {len(df)} records for {TICKER}.")
        print(f"Data saved to: {OUTPUT_FILE_PATH}")

    except requests.exceptions.RequestException as e:
        print(f"\nFATAL API ERROR: Could not connect to Polygon.io or bad response.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    fetch_polygon_data_and_save()