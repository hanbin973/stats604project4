import os
import glob
from datetime import datetime, timedelta
import time
import json
import requests

import numpy as np
import pandas as pd
import pytz
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODELS_DIR = "/app/models"
OUTPUT_PATH = "/app/output/merged_all_years.csv"

TBASE = 18.0  # same as in fitting.py
TZ_NAME = "America/New_York"

# --- Configuration: PJM Zones ---
ZONE_LOCATIONS = {
    "COMED": {"lat": 41.88, "lon": -87.63, "timezone": "America/Chicago"},
    "PSEG": {"lat": 40.73, "lon": -74.17, "timezone": "America/New_York"},
    "DOM": {"lat": 37.54, "lon": -77.43, "timezone": "America/New_York"},
    "BGE": {"lat": 39.29, "lon": -76.61, "timezone": "America/New_York"},
    "AEP": {"lat": 39.96, "lon": -83.00, "timezone": "America/New_York"},
    "AECO": {"lat": 39.36, "lon": -74.42, "timezone": "America/New_York"},
    "AEPAPT": {"lat": 37.27, "lon": -79.94, "timezone": "America/New_York"},
    "AEPIMP": {"lat": 41.08, "lon": -85.14, "timezone": "America/Indiana/Indianapolis"},
    "AEPKPT": {"lat": 38.48, "lon": -82.64, "timezone": "America/New_York"},
    "AEPOPT": {"lat": 40.80, "lon": -81.38, "timezone": "America/New_York"},
    "AP": {"lat": 40.30, "lon": -79.54, "timezone": "America/New_York"},
    "BC": {"lat": 39.29, "lon": -76.61, "timezone": "America/New_York"},
    "CE": {"lat": 41.50, "lon": -81.69, "timezone": "America/New_York"},
    "DAY": {"lat": 39.76, "lon": -84.19, "timezone": "America/New_York"},
    "DEOK": {"lat": 39.10, "lon": -84.51, "timezone": "America/New_York"},
    "DPLCO": {"lat": 39.75, "lon": -75.55, "timezone": "America/New_York"},
    "DUQ": {"lat": 40.44, "lon": -79.99, "timezone": "America/New_York"},
    "EASTON": {"lat": 38.77, "lon": -76.08, "timezone": "America/New_York"},
    "EKPC": {"lat": 37.99, "lon": -84.18, "timezone": "America/New_York"},
    "JC": {"lat": 40.80, "lon": -74.48, "timezone": "America/New_York"},
    "ME": {"lat": 40.34, "lon": -75.93, "timezone": "America/New_York"},
    "OE": {"lat": 41.08, "lon": -81.52, "timezone": "America/New_York"},
    "OVEC": {"lat": 39.06, "lon": -83.01, "timezone": "America/New_York"},
    "PAPWR": {"lat": 40.61, "lon": -75.49, "timezone": "America/New_York"},
    "PE": {"lat": 39.95, "lon": -75.16, "timezone": "America/New_York"},
    "PEPCO": {"lat": 38.91, "lon": -77.04, "timezone": "America/New_York"},
    "PLCO": {"lat": 40.61, "lon": -75.49, "timezone": "America/New_York"},
    "PN": {"lat": 42.13, "lon": -80.09, "timezone": "America/New_York"},
    "PS": {"lat": 40.73, "lon": -74.17, "timezone": "America/New_York"},
    "RECO": {"lat": 41.09, "lon": -74.05, "timezone": "America/New_York"},
    "SMECO": {"lat": 38.52, "lon": -76.80, "timezone": "America/New_York"},
    "UGI": {"lat": 41.25, "lon": -75.88, "timezone": "America/New_York"},
    "VMEU": {"lat": 39.49, "lon": -75.03, "timezone": "America/New_York"}
}

def download_models_from_github():
    """
    Downloads .npy model files from the specific GitHub repository folder.
    Repo: seyonghw/604project4
    Path: models/
    """
    print("--- Downloading models from GitHub ---")
    api_url = "https://api.github.com/repos/seyonghw/604project4/contents/models"
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error accessing GitHub API: {response.status_code}")
            return

        files = response.json()
        if not isinstance(files, list):
             print("Unexpected response format from GitHub.")
             return

        count = 0
        for file_info in files:
            if file_info['name'].endswith('.npy'):
                download_url = file_info['download_url']
                local_path = os.path.join(MODELS_DIR, file_info['name'])
                
                print(f"  Downloading {file_info['name']}...")
                r = requests.get(download_url)
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                count += 1
        
        print(f"Successfully downloaded {count} model files to {MODELS_DIR}")

    except Exception as e:
        print(f"Exception during model download: {e}")

def get_today_str():
    """Return today's date string in America/New_York, format YYYY-MM-DD."""
    tz = pytz.timezone(TZ_NAME)
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d"), now

def fetch_live_weather_for_zone(zone, reference_date=None):
    """
    Fetches live weather data directly from the API.
    
    Args:
        zone (str): The PJM zone identifier.
        reference_date (date or datetime, optional): A reference date to calculate historical fetch window.
                                                     past_days = 2 + (current_date - reference_date).
                                                     If None, defaults to past_days = 2.
    
    Returns:
        DataFrame: A DataFrame tailored for the model's input (for the current day).
    """
    if zone not in ZONE_LOCATIONS:
        return None

    info = ZONE_LOCATIONS[zone]
    forecast_api_url = "https://api.open-meteo.com/v1/forecast"
    
    # Determine current date in zone's timezone
    tz = pytz.timezone(info["timezone"])
    current_date_obj = datetime.now(tz).date()

    # Calculate past_days dynamically
    past_days = 2
    if reference_date:
        # Handle input being datetime vs date
        if isinstance(reference_date, datetime):
            ref_date = reference_date.date()
        else:
            ref_date = reference_date
        
        # Calculate difference in days
        days_diff = (current_date_obj - ref_date).days
        
        # Formula: 2 + (current - reference)
        # Ensure we don't send negative days to API if reference is in future
        past_days = max(0, days_diff)

    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "hourly": "temperature_2m",
        "past_days": past_days,
        "forecast_days": 11,
        "timezone": info["timezone"]
    }

    try:
        response = requests.get(forecast_api_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            print(f"!!! FLAG: Weather information absent in API (no hourly data) for zone {zone} !!!")
            return None

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Rename to standard training names
        df = df.rename(columns={"temperature_2m": "temp_at_time_t"})
        
        # Create the lag/lead features required by the model
        #df["temp_at_time_t_minus_24h"] = df["temp_at_time_t"].shift(24)
        #df["temp_at_time_t_minus_48h"] = df["temp_at_time_t"].shift(48)
        #df["temp_at_time_t_plus_24h_FORECAST"] = df["temp_at_time_t"].shift(-24)

        # Filter for today (local zone time)
        #today_str = datetime.now(tz).strftime("%Y-%m-%d")

        #today_mask = df.index.strftime("%Y-%m-%d") == today_str
        
        #if not any(today_mask):
        #     print(f"!!! FLAG: Weather information absent for today's date ({today_str}) for zone {zone} !!!")
        #     return None
        
        # Return only the 24 hours for today
        # return df.loc[today_mask].copy()
        return df

    except Exception as e:
        print(f"!!! FLAG: Exception occurred fetching weather for zone {zone}: {e}")
        return None

def load_zones_from_models():
    if not os.path.exists(MODELS_DIR):
         return []
         
    pattern = os.path.join(MODELS_DIR, "*_params.npy")
    param_paths = sorted(glob.glob(pattern))
    zones = []

    for path in param_paths:
        fname = os.path.basename(path)
        if fname.endswith("_params.npy"):
            zone = fname[:-len("_params.npy")]
            zones.append(zone)

    return zones

def prepare_training_data():
    """
    Load merged_all_years.csv and reproduce the feature engineering from fitting.py.
    Returns:
        df: preprocessed DataFrame with columns:
            ['datetime_beginning_ept', 'load_area', 'mw', 'CDH', 'HDH', dow_* ...]
        exog_cols: list of exogenous column names used in the model.
    """
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found. Make sure load_data.py has created it."
        )

    # Load merged data
    df = pd.read_csv(OUTPUT_PATH, parse_dates=[0])

    # Ensure we have a proper datetime column (same logic as fitting.py)
    if "datetime_beginning_ept" in df.columns:
        dt = pd.to_datetime(df["datetime_beginning_ept"])
    else:
        dt = pd.to_datetime(df.iloc[:, 0])
        df.insert(0, "datetime_beginning_ept", dt)

    needed_cols = ["datetime_beginning_ept", "load_area", "mw", "temperature_2m"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged_all_years.csv: {missing}")

    df = df[needed_cols].copy()
    df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    # Day of week dummies
    df["dow"] = df["datetime_beginning_ept"].dt.dayofweek  # 0=Mon,...,6=Sun
    df = pd.get_dummies(df, columns=["dow"], prefix="dow", dtype=float, drop_first=True)

    # Drop AE and RTO regions (as in fitting.py)
    df = df[~df["load_area"].isin(["AE", "RTO"])]

    # Temperature-based features
    temp = df["temperature_2m"]
    df["CDH"] = (temp - TBASE).clip(lower=0)
    df["HDH"] = (TBASE - temp).clip(lower=0)

    # Restrict to data from 2025 onward (same as in your current fitting.py)
    df = df[df["datetime_beginning_ept"].dt.year >= 2025].copy()
    if df.empty:
        raise ValueError("No data from 2025 onward found in merged_all_years.csv.")

    # Exogenous columns
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]

    # Ensure all exog columns exist (some dow_* may be missing)
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df, exog_cols

def build_exog_from_weather(weather_df, today_local):
    """
    Given a per-zone live weather DataFrame for 'today' (24 hours),
    build the 96-step exogenous matrix needed for the SARIMAX forecast.
    """
    if "time" in weather_df.columns:
        weather_df["time"] = pd.to_datetime(weather_df["time"])
        weather_df = weather_df.sort_values("time").reset_index(drop=True)
    temps = weather_df["temp_at_time_t"].to_numpy().T.flatten()
    df_temp = pd.DataFrame({"temp": temps})
    df_temp["datetime"] = weather_df.index
    df_temp["CDH"] = (df_temp["temp"] - TBASE).clip(lower=0).values
    df_temp["HDH"] = (TBASE - df_temp["temp"]).clip(lower=0).values
    df_temp["dow"] = df_temp["datetime"].dt.dayofweek
    df_temp = pd.get_dummies(df_temp, columns=["dow"], prefix="dow", dtype=float, drop_first=True)
    exog_cols = ["CDH", "HDH", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]
    for col in exog_cols:
        if col not in df_temp.columns:
            df_temp[col] = 0.0
    return df_temp[exog_cols]


def main():
    # --- STEP 1: Download Models ---
    download_models_from_github()

    today_str, now_local = get_today_str()
    zones = load_zones_from_models()
    if not zones:
        print("No models found after download attempt. Exiting.")
        return

    try:
        df_train, exog_cols = prepare_training_data()
    except Exception as e:
        print(f"Error preparing training data: {e}")
        return

    all_zone_daily_loads = []
    all_zone_peak_hours = []
    all_zone_top_two_days = []

    # Define the reference date here (November 11, 2025)
    reference_date = datetime(2025, 11, 17).date()

    for zone in zones:
        param_path = os.path.join(MODELS_DIR, f"{zone}_params.npy")
        
        # --- DIRECTLY FETCH LIVE WEATHER HERE ---
        # Pass a reference_date if you have one, otherwise it defaults to None
        # Example: reference_date = datetime(2025, 10, 1).date()
        weather_df = fetch_live_weather_for_zone(zone, reference_date=reference_date)
        
        # Rate limiting
        time.sleep(0.2) 
        
        if weather_df is None or weather_df.empty:
            all_zone_daily_loads.append([-1]*24)
            all_zone_peak_hours.append(-1)
            continue

        params = np.load(param_path)
        
        # Sort by time and drop NaNs for correct training data
        df_zone = df_train[df_train["load_area"] == zone].sort_values("datetime_beginning_ept").copy()
        df_zone = df_zone.dropna(subset=["mw"] + exog_cols)
        
        if df_zone.empty:
            # DEBUG FLAG for Empty Data
            # print(f"!!! FLAG: Historical Data Missing for {zone}. Check merging in load_data.py")
            all_zone_daily_loads.append([-1]*24)
            all_zone_peak_hours.append(-1)
            continue
        
        df_zone = df_zone.reset_index(drop=True)
        y_train = df_zone["mw"]
        exog_train = df_zone[exog_cols]

        try:
            model = SARIMAX(
                y_train,
                order=(1, 0, 0),
                seasonal_order=(1, 1, 1, 24),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.filter(params)
            exog_future = build_exog_from_weather(weather_df, now_local)
            
            exog_future = exog_future[exog_cols]
            steps = len(exog_future)

            forecast_res = results.get_forecast(steps=steps, exog=exog_future)
            mean_forecast = forecast_res.predicted_mean

            last_24 = mean_forecast.iloc[-(24*10):-(24*9)]
            #print("last_24", last_24)
            daily_loads_int = np.rint(last_24).astype(int).tolist()
            all_zone_daily_loads.append(daily_loads_int)
            peak_hour = int(last_24.argmax())
            all_zone_peak_hours.append(peak_hour)

            top_two = np.zeros(10)
            daily_mean = mean_forecast.values.reshape(-1, 24).mean(axis=-1)[-10:]
            print(daily_mean)
            top_two[np.argsort(daily_mean)[::-1][:2]] = 1
            print(top_two)
            all_zone_top_two_days.append(top_two[0])

        except Exception:
            all_zone_daily_loads.append([-1]*24)
            all_zone_peak_hours.append(-1)
            all_zone_top_two_days.append(-1)

    fields = []
    fields.append(f'"{today_str}"')
    for loads in all_zone_daily_loads:
        fields.extend(str(int(x)) for x in loads)
    for ph in all_zone_peak_hours:
        fields.append(str(int(ph)))
    for pd in all_zone_top_two_days:
        fields.append(str(int(pd)))

    print(",".join(fields))

if __name__ == "__main__":
    main()
