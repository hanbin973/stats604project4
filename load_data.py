import pandas as pd
import glob
import os
import requests  # For making API calls
import json

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
    end_str = max_date.strftime('%Y-%m-%d')
    
    print(f"Date range found in CSVs: {start_str} to {end_str}")
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

        # --- 3. Process and Merge Weather Data ---
        print("\n--- 3. Processing and Merging Weather Data ---")

        # Load weather data into a DataFrame
        hourly_data = weather_data.get('hourly', {})
        if not hourly_data:
            print("Error: No 'hourly' data found in historical API response.")
        else:
            # Create a clean, time-indexed DataFrame for the weather
            weather_df = pd.DataFrame(hourly_data)
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df = weather_df.set_index('time')
            print("Created historical weather DataFrame. Head:")
            print(weather_df.head())

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
            
            print("Merged DataFrame (head):")
            print(merged_df.head())

    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical weather data: {e}")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from historical weather API.")
    except Exception as e:
        print(f"An error occurred during historical weather processing: {e}")

    # --- 4. Fetching Yesterday, Today, & Tomorrow's Temperatures ---
    print("\n--- 4. Fetching Yesterday, Today, & Tomorrow's Temperatures ---")
    print("Fetching 3-day temperature data from Open-Meteo Forecast API...")
    
    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast" # Corrected typo
    forecast_params = {
        "latitude": 39.95,  # Philadelphia
        "longitude": -75.16, # Philadelphia
        "daily": "temperature_2m_mean",     # Get daily mean temp
        "past_days": 1,                     # Get yesterday's observed data
        "forecast_days": 2,                 # Get today and tomorrow's data
        "timezone": "America/New_York"      # Use a specific timezone for consistency
    }

    try:
        response = requests.get(FORECAST_API_URL, params=forecast_params)
        response.raise_for_status()
        three_day_data = response.json()

        # --- Process 3-Day Forecast ---
        daily_data = three_day_data.get('daily', {})
        
        if not daily_data or 'time' not in daily_data or len(daily_data['time']) < 3:
            print("\nError: Could not retrieve the 3-day data from forecast response.")
        else:
            # Create a DataFrame from the daily forecast data
            daily_df = pd.DataFrame(daily_data)
            
            # Extract the 3 temperatures
            yesterday_temp = daily_df['temperature_2m_mean'][0]
            today_temp = daily_df['temperature_2m_mean'][1]
            tomorrow_temp = daily_df['temperature_2m_mean'][2]

            print("\nSuccessfully fetched 3-day temperature data:")
            print(f"  - Yesterday (Observed):    {yesterday_temp}°C")
            print(f"  - Today (Observed/Fcst): {today_temp}°C")
            print(f"  - Tomorrow (Forecast):   {tomorrow_temp}°C")

            # --- Append to the merged_df from Section 3 ---
            if merged_df is not None:
                print(f"\nAppending 3-day temperatures as new columns to the merged DataFrame...")
                
                merged_df['yesterday_observed_mean_temp'] = yesterday_temp
                merged_df['today_forecast_mean_temp'] = today_temp
                merged_df['tomorrow_forecast_mean_temp'] = tomorrow_temp
                
                print("Successfully added new columns. Tail of the final DataFrame:")
                print(merged_df.tail())
            else:
                print("\n'merged_df' was not created (likely due to an error in Section 3), skipping append.")
                # Just print the 3-day data if we can't append it
                print(daily_df.to_string()) # .to_string() ensures it prints nicely formatted

    except requests.exceptions.RequestException as e:
        print(f"\nError fetching 3-day forecast data: {e}")
    except json.JSONDecodeError:
        print("\nError: Could not decode JSON response from forecast API.")
    except Exception as e:
        print(f"\nAn error occurred during 3-day forecast processing: {e}")


print("\n--- Python script execution finished. ---")
