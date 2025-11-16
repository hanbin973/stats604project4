import pandas as pd
import glob
import os
import requests  # For making API calls
import json
from datetime import datetime

# --- 1. Loading CSV data from OSF file ---
print("--- 1. Loading CSV data from OSF file ---")
DATA_DIR = '/app/data'
print(f"Searching for CSV files in {DATA_DIR} and its subdirectories...")

search_path = os.path.join(DATA_DIR, '**', '*.csv')
csv_files = glob.glob(search_path, recursive=True)

if not csv_files:
    print(f"Error: No CSV files found in {DATA_DIR}.")
    dataframes = {}
else:
    print(f"Found {len(csv_files)} CSV files. Loading each one...")
    dataframes = {}
    all_dates = []  # To store all dates from all files

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            # --- This is the new parsing logic ---
            # Get the first column name
            date_col = df.columns[0]
            # Parse the date column. 
            # `pd.to_datetime` is smart enough to handle "1/1/2016 5:00:00 AM"
            parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
            
            # Store the parsed dates and the DataFrame
            all_dates.append(parsed_dates)
            dataframes[file_name] = df
            
            print(f"  - Successfully loaded and parsed dates from {file_name}")
        except Exception as e:
            print(f"Could not load {file_name}: {e}")

    print(f"\nSuccessfully loaded {len(dataframes)} DataFrames.")

# --- 2. Fetching Weather Data for CSV Date Range ---
if not dataframes:
    print("\nNo dataframes loaded, skipping weather fetch.")
else:
    print("\n--- 2. Fetching Weather Data for CSV Date Range ---")
    
    # Combine all dates from all files to find the min and max
    all_dates_combined = pd.concat(all_dates).dropna()
    min_date = all_dates_combined.min()
    max_date = all_dates_combined.max()
    
    # Format dates for the API (YYYY-MM-DD)
    start_str = min_date.strftime('%Y-%m-%d')
    # Add one day to the end_date to make sure we can get the t+24h for the last row
    end_str = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Date range found in CSVs (with +1 day buffer): {start_str} to {end_str}")
    print("Fetching corresponding weather data from Open-Meteo...")

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 39.95,  # Philadelphia
        "longitude": -75.16, # Philadelphia
        "start_date": start_str,
        "end_date": end_str,
        "hourly": "temperature_2m"
    }

    # Define merged_df in this scope so Section 4 can access it
    merged_df = None 

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
        weather_data = response.json()
        print("Successfully fetched and parsed historical weather data.")

        # --- 3. Process and Merge Historical Data (For Model Training) ---
        print("\n--- 3. Processing and Merging Historical Data (For Model Training) ---")

        # Load weather data into a DataFrame
        hourly_data = weather_data.get('hourly', {})
        if not hourly_data:
            print("Error: No 'hourly' data found in historical API response.")
        else:
            # Create a clean, time-indexed DataFrame for the weather
            weather_df = pd.DataFrame(hourly_data)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.set_index('time')

            # --- This is the training data logic ---
            # 1. Rename the main temperature column
            weather_df = weather_df.rename(columns={'temperature_2m': 'temp_at_time_t'})
            
            # 2. Create 'yesterday's temp' (temp at t - 24 hours)
            weather_df['temp_at_time_t_minus_24h'] = weather_df['temp_at_time_t'].shift(24)
            
            # 3. Create 'tomorrow's temp' (temp at t + 24 hours) - THIS IS HISTORICAL GROUND TRUTH
            weather_df['temp_at_time_t_plus_24h_OBSERVED'] = weather_df['temp_at_time_t'].shift(-24)
            
            print("Created historical weather DataFrame with relative columns (Head):")
            print(weather_df.head(5))

            # Now, let's merge this weather data back into your original CSVs
            # We'll just show an example using the first DataFrame
            
            first_df_key = list(dataframes.keys())[0]
            print(f"\nExample merge with first file: '{first_df_key}'")
            
            original_df = dataframes[first_df_key].copy() # Use .copy() to avoid SettingWithCopyWarning
            date_col = original_df.columns[0]
            
            # Parse dates and set as index (this time on the original df)
            original_df[date_col] = pd.to_datetime(original_df[date_col])
            original_df = original_df.set_index(date_col)
            
            # Merge! This joins the temperature to the matching timestamp.
            # 'how=left' keeps all your original data.
            merged_df = original_df.merge(weather_df, left_index=True, right_index=True, how='left')
            
            print("Merged Historical Training DataFrame (Head):")
            print(merged_df.head())
            
            print("\nMerged Historical Training DataFrame (Tail):")
            print(merged_df.tail())


    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical weather data: {e}")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from historical weather API.")
    except Exception as e:
        print(f"An error occurred during historical weather processing: {e}")

    # --- 4. Fetching "Live" Data for New Predictions ---
    print("\n--- 4. Fetching 'Live' Data for New Predictions (Yesterday, Today, Tomorrow Forecast) ---")
    
    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": 39.95,
        "longitude": -75.16,
        "hourly": "temperature_2m",
        "past_days": 1,       # Get yesterday's observed data
        "forecast_days": 2,   # Get today and tomorrow's data
        "timezone": "America/New_York"
    }

    try:
        response = requests.get(FORECAST_API_URL, params=forecast_params)
        response.raise_for_status()
        live_data = response.json()
        
        hourly_live = live_data.get('hourly', {})
        
        if not hourly_live or 'time' not in hourly_live or len(hourly_live['time']) < 72:
            print("Error: Could not retrieve full 72 hours of data for live prediction.")
        else:
            live_weather_df = pd.DataFrame(hourly_live)
            live_weather_df['time'] = pd.to_datetime(live_weather_df['time'])
            live_weather_df = live_weather_df.set_index('time')
            
            # This DataFrame contains data for yesterday, today, and tomorrow.
            # We only care about the rows for "today" to make predictions.
            
            # Create the 3 columns you want
            # temp_at_time_t
            # temp_at_time_t_minus_24h (shift 24)
            # temp_at_time_t_plus_24h_FORECAST (shift -24)
            
            live_weather_df = live_weather_df.rename(columns={'temperature_2m': 'temp_at_time_t'})
            live_weather_df['temp_at_time_t_minus_24h'] = live_weather_df['temp_at_time_t'].shift(24)
            live_weather_df['temp_at_time_t_plus_24h_FORECAST'] = live_weather_df['temp_at_time_t'].shift(-24)
            
            # The data for "today" (Nov 16, 2025) will have all 3 columns populated.
            # Let's filter to just the 24 hours for today.
            today_str = datetime.now().strftime('%Y-%m-%d')
            live_prediction_df = live_weather_df.loc[today_str].copy()
            
            print(f"\nSuccessfully created 'Live Prediction' DataFrame for: {today_str}")
            print(live_prediction_df.to_string())

    except requests.exceptions.RequestException as e:
        print(f"\nError fetching live forecast data: {e}")
    except Exception as e:
        print(f"\nAn error occurred during live data processing: {e}")


print("\n--- Python script execution finished. ---")
