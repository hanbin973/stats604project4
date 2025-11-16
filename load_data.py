import pandas as pd
import glob
import os
import requests  # For making API calls
import json
from datetime import datetime
import time # To prevent API rate limiting

# --- Configuration ---
# A map of PJM zones to their representative locations and timezones.
ZONE_LOCATIONS = {
    # Original Zones
    "COMED": {"lat": 41.88, "lon": -87.63, "timezone": "America/Chicago"}, # Commonwealth Edison (Chicago, IL)
    "PSEG": {"lat": 40.73, "lon": -74.17, "timezone": "America/New_York"}, # Public Service Electric and Gas (Newark, NJ)
    "DOM": {"lat": 37.54, "lon": -77.43, "timezone": "America/New_York"}, # Dominion Virginia Power (Richmond, VA)
    "BGE": {"lat": 39.29, "lon": -76.61, "timezone": "America/New_York"}, # Baltimore Gas & Electric (Baltimore, MD)
    "PEPCO": {"lat": 38.91, "lon": -77.04, "timezone": "America/New_York"}, # Potomac Electric Power Company (Washington, D.C.)
    "AEP": {"lat": 39.96, "lon": -83.00, "timezone": "America/New_York"}, # American Electric Power (Columbus, OH - General)
    "AECO": {"lat": 39.36, "lon": -74.42, "timezone": "America/New_York"},   # Atlantic City Electric (Atlantic City, NJ)
    "AEPAPT": {"lat": 37.27, "lon": -79.94, "timezone": "America/New_York"}, # AEP Appalachian Power (Roanoke, VA)
    "AEPIMP": {"lat": 41.08, "lon": -85.14, "timezone": "America/Indiana/Indianapolis"}, # AEP Indiana Michigan Power (Fort Wayne, IN)
    "AEPKPT": {"lat": 38.48, "lon": -82.64, "timezone": "America/New_York"}, # AEP Kentucky Power (Ashland, KY)
    "AEPOPT": {"lat": 40.80, "lon": -81.38, "timezone": "America/New_York"}, # AEP Ohio Power (Canton, OH)
    "AP": {"lat": 40.30, "lon": -79.54, "timezone": "America/New_York"},     # Allegheny Power (Greensburg, PA)
    "BC": {"lat": 39.29, "lon": -76.61, "timezone": "America/New_York"},     # Alias for BGE (Baltimore, MD)
    "CE": {"lat": 41.50, "lon": -81.69, "timezone": "America/New_York"},     # Cleveland Electric Illuminating (Cleveland, OH)
    "DAY": {"lat": 39.76, "lon": -84.19, "timezone": "America/New_York"},    # Dayton Power and Light (Dayton, OH)
    "DEOK": {"lat": 39.10, "lon": -84.51, "timezone": "America/New_York"},   # Duke Energy Ohio/Kentucky (Cincinnati, OH)
    "DPLCO": {"lat": 39.75, "lon": -75.55, "timezone": "America/New_York"},  # Delmarva Power & Light (Wilmington, DE)
    "DUQ": {"lat": 40.44, "lon": -79.99, "timezone": "America/New_York"},    # Duquesne Light (Pittsburgh, PA)
    "EASTON": {"lat": 38.77, "lon": -76.08, "timezone": "America/New_York"}, # Easton Utilities (Easton, MD)
    "EKPC": {"lat": 37.99, "lon": -84.18, "timezone": "America/New_York"},   # East Kentucky Power Cooperative (Winchester, KY)
    "JC": {"lat": 40.80, "lon": -74.48, "timezone": "America/New_York"},     # Jersey Central Power & Light (Morristown, NJ)
    "ME": {"lat": 40.34, "lon": -75.93, "timezone": "America/New_York"},     # Metropolitan Edison (Reading, PA)
    "OE": {"lat": 41.08, "lon": -81.52, "timezone": "America/New_York"},     # Ohio Edison (Akron, OH)
    "OVEC": {"lat": 39.06, "lon": -83.01, "timezone": "America/New_York"},   # Ohio Valley Electric Corporation (Piketon, OH)
    "PAPWR": {"lat": 40.61, "lon": -75.49, "timezone": "America/New_York"},  # PPL (Pennsylvania Power & Light) (Allentown, PA)
    "PE": {"lat": 39.95, "lon": -75.16, "timezone": "America/New_York"}      # Alias for PECO (Philadelphia, PA)
}
# --- End of Configuration ---


def find_zone_column(df):
    """Helper function to find the zone column, case-insensitive."""
    possible_names = ['zone', 'ZONE', 'Area', 'AREA', 'region', 'REGION']
    for col in df.columns:
        if col in possible_names:
            return col
    return None

# --- 1. Load and Consolidate All CSV Data ---
print("--- 1. Loading and Consolidating All CSV Data ---")
DATA_DIR = '/app/data'
search_path = os.path.join(DATA_DIR, '**', '*.csv')
csv_files = glob.glob(search_path, recursive=True)

all_data_df = None
all_dates = []
zone_col = None

if not csv_files:
    print(f"Error: No CSV files found in {DATA_DIR}.")
else:
    print(f"Found {len(csv_files)} CSV files. Loading and concatenating...")
    df_list = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Find zone column (only need to do this once, assumes same format)
            if zone_col is None:
                zone_col = find_zone_column(df)

            # Get date column (second column)
            date_col = df.columns[1]
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            
            df[date_col] = parsed_dates
            all_dates.append(parsed_dates)
            df_list.append(df)
            
        except Exception as e:
            print(f"Could not load {os.path.basename(file_path)}: {e}")

    # Combine all data into one big DataFrame
    all_data_df = pd.concat(df_list, ignore_index=True)
    all_dates_combined = pd.concat(all_dates).dropna()
    
    min_date = all_dates_combined.min()
    max_date = all_dates_combined.max()
    
    # Get all unique zones found in the data
    if zone_col:
        unique_zones = all_data_df[zone_col].unique()
        print(f"\nSuccessfully loaded {len(all_data_df)} total rows.")
        print(f"Found {len(unique_zones)} unique zones: {unique_zones}")
    else:
        print("\nError: Could not find a 'zone' column in the CSV files. Stopping.")
        unique_zones = []

# --- 2. Fetching Weather Data for All Zones ---
if all_data_df is None or len(unique_zones) == 0:
    print("\nNo data loaded or no zones found, skipping weather fetch.")
else:
    print("\n--- 2. Fetching Historical Weather Data for All Zones ---")
    start_str = min_date.strftime('%Y-%m-%d')
    # Add one day to the end_date to make sure we can get the t+24h for the last row
    end_str = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Global date range: {start_str} to {end_str}")

    all_weather_dfs = [] # To store weather dataframes for each zone

    for zone in unique_zones:
        if zone not in ZONE_LOCATIONS:
            print(f"Warning: No location data for zone '{zone}'. Skipping.")
            continue
        
        location = ZONE_LOCATIONS[zone]
        print(f"Fetching weather for: {zone} (Lat: {location['lat']}, Lon: {location['lon']})...")
        
        API_URL = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": location['lat'],
            "longitude": location['lon'],
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m",
            "timezone": location['timezone']
        }
        
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            weather_data = response.json()
            
            hourly_data = weather_data.get('hourly', {})
            if hourly_data:
                weather_df = pd.DataFrame(hourly_data)
                weather_df['time'] = pd.to_datetime(weather_df['time'])
                weather_df['zone'] = zone  # Add the zone column
                all_weather_dfs.append(weather_df)
                print(f"  ...Success for {zone}.")
            else:
                print(f"  ...No 'hourly' data returned for {zone}.")

            time.sleep(1) # Be nice to the API
        except Exception as e:
            print(f"  ...Error fetching weather for {zone}: {e}")

    # Combine all zonal weather data into one big DataFrame
    if not all_weather_dfs:
        print("Error: No weather data could be fetched for any zone.")
    else:
        all_weather_df = pd.concat(all_weather_dfs, ignore_index=True)
        all_weather_df = all_weather_df.set_index('time')
        print(f"\nSuccessfully combined weather data for {len(all_weather_dfs)} zones.")

        # --- 3. Process and Merge Historical Data (For Model Training) ---
        print("\n--- 3. Processing and Merging Historical Data (For Model Training) ---")

        # --- This is the training data logic ---
        # 1. Rename the main temperature column
        all_weather_df = all_weather_df.rename(columns={'temperature_2m': 'temp_at_time_t'})
        
        # 2. Create 'yesterday's temp' (t-24h)
        # We MUST groupby 'zone' so data doesn't bleed across zones
        all_weather_df['temp_at_time_t_minus_24h'] = all_weather_df.groupby('zone')['temp_at_time_t'].shift(24)
        
        # 3. Create 'tomorrow's temp' (t+24h) - THIS IS HISTORICAL GROUND TRUTH
        all_weather_df['temp_at_time_t_plus_24h_OBSERVED'] = all_weather_df.groupby('zone')['temp_at_time_t'].shift(-24)
        
        print("Created historical weather DataFrame with relative columns (Grouped by Zone).")

        # Prepare main data for merge
        date_col = all_data_df.columns[1] # Second column
        all_data_df = all_data_df.rename(columns={date_col: 'time', zone_col: 'zone'})
        
        # Merge on both time and zone
        # We reset index on weather_df to use 'time' and 'zone' as merge keys
        merged_df = pd.merge(
            all_data_df, 
            all_weather_df.reset_index(), 
            on=['time', 'zone'], 
            how='left'
        )
        
        print("Successfully merged all PJM data with all weather data.")
        
        print("\nMerged Historical Training DataFrame (Head):")
        print(merged_df.head())
        
        print("\nMerged Historical Training DataFrame (Tail):")
        print(merged_df.tail())

    # --- 4. Fetching "Live" Data for New Predictions ---
    print("\n--- 4. Fetching 'Live' Data for New Predictions (All Zones) ---")
    
    live_weather_dfs = []
    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

    for zone in unique_zones:
        if zone not in ZONE_LOCATIONS:
            print(f"Warning: No location data for zone '{zone}'. Skipping live data.")
            continue
        
        location = ZONE_LOCATIONS[zone]
        print(f"Fetching live forecast for: {zone}...")
        
        forecast_params = {
            "latitude": location['lat'],
            "longitude": location['lon'],
            "hourly": "temperature_2m",
            "past_days": 1,       # Get yesterday's observed data
            "forecast_days": 2,   # Get today and tomorrow's data
            "timezone": location['timezone']
        }
        
        try:
            response = requests.get(FORECAST_API_URL, params=forecast_params)
            response.raise_for_status()
            live_data = response.json()
            
            hourly_live = live_data.get('hourly', {})
            if hourly_live and 'time' in hourly_live and len(hourly_live['time']) >= 72:
                live_df = pd.DataFrame(hourly_live)
                live_df['time'] = pd.to_datetime(live_df['time'])
                live_df['zone'] = zone
                live_weather_dfs.append(live_df)
                print(f"  ...Success for {zone}.")
            else:
                 print(f"  ...No 'hourly' data returned for {zone}.")
            
            time.sleep(1) # Be nice to the API
        except Exception as e:
            print(f"  ...Error fetching live forecast for {zone}: {e}")

    if live_weather_dfs:
        # Combine all live weather data
        live_weather_df = pd.concat(live_weather_dfs, ignore_index=True)
        live_weather_df = live_weather_df.set_index('time')

        # Create the 3 columns, grouped by zone
        live_weather_df = live_weather_df.rename(columns={'temperature_2m': 'temp_at_time_t'})
        live_weather_df['temp_at_time_t_minus_24h'] = live_weather_df.groupby('zone')['temp_at_time_t'].shift(24)
        live_weather_df['temp_at_time_t_plus_24h_FORECAST'] = live_weather_df.groupby('zone')['temp_at_time_t'].shift(-24)
        
        # Filter for just today's 24 hours
        today_str = datetime.now().strftime('%Y-%m-%d')
        live_prediction_df = live_weather_df.loc[today_str].copy()
        
        print(f"\nSuccessfully created 'Live Prediction' DataFrame for: {today_str}")
        print("Showing data for all zones (first 10 rows):")
        print(live_prediction_df.head(10))
        print("\nLast 10 rows:")
        print(live_prediction_df.tail(10))

print("\n--- Python script execution finished. ---")
