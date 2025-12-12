# %%
import requests
import json
import pandas as pd
import time

# %%
headers = {"User-Agent": "DataFetcher/1.0", "Accept": "application/json"}


# %%
def fetch_daily_weather(api, lat, lon, date):
    """Original function - fetches weather for a single date"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "models": 'era5',
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "wind_direction_10m_dominant",
            "precipitation_sum",
            "precipitation_hours",
            "et0_fao_evapotranspiration",
            "weather_code"
        ]),
        "timezone": "UTC"
    }

    r = requests.get(api, params=params, headers=headers)
    data = r.json()

    if "daily" not in data:
        return None  # no data for this day/location

    daily = data["daily"]

    # They always return lists → convert to one row
    row = {
        "temperature_2m_mean": daily["temperature_2m_mean"][0],
        "temperature_2m_max": daily["temperature_2m_max"][0],
        "temperature_2m_min": daily["temperature_2m_min"][0],
        "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"][0],
        "wind_speed_10m_mean": daily["wind_speed_10m_mean"][0],
        "wind_direction_10m_dominant": daily["wind_direction_10m_dominant"][0],
        "precipitation_sum": daily["precipitation_sum"][0],
        "precipitation_hours": daily["precipitation_hours"][0],
        "et0_fao_evapotranspiration": daily["et0_fao_evapotranspiration"][0],
        "weather_code": daily["weather_code"][0]
    }

    return row


# NEW FUNCTION: Batch fetch by location
def fetch_weather_batch_by_location(api, airquality_df):
    """
    Batch fetches weather by grouping dates per location.
    Much more efficient than fetching one date at a time.
    
    For 68 locations: ~68 API calls instead of 9,902
    """
    # Group by location
    location_groups = airquality_df.groupby(['latitude', 'longitude']).agg({
        'date': ['min', 'max']
    }).reset_index()
    location_groups.columns = ['latitude', 'longitude', 'start_date', 'end_date']
    
    all_weather_rows = []
    total = len(location_groups)
    
    print(f"Fetching weather for {total} unique locations...")
    
    for idx, row in location_groups.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        start_date = row['start_date']
        end_date = row['end_date']
        
        print(f"[{idx+1}/{total}] Fetching {lat:.4f}, {lon:.4f} ({start_date} to {end_date})")
        
        # Fetch date range for this location
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "models": 'era5',
            "daily": ",".join([
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_mean",
                "wind_speed_10m_mean",
                "wind_direction_10m_dominant",
                "precipitation_sum",
                "precipitation_hours",
                "et0_fao_evapotranspiration",
                "weather_code"
            ]),
            "timezone": "UTC"
        }
        
        try:
            r = requests.get(api, params=params, headers=headers)
            data = r.json()
            
            if "daily" not in data:
                print(f"  ✗ No data")
                continue
            
            daily = data["daily"]
            
            # Create a row for each date
            for i in range(len(daily["time"])):
                weather_row = {
                    "latitude": lat,
                    "longitude": lon,
                    "date": daily["time"][i],
                    "temperature_2m_mean": daily["temperature_2m_mean"][i],
                    "temperature_2m_max": daily["temperature_2m_max"][i],
                    "temperature_2m_min": daily["temperature_2m_min"][i],
                    "relative_humidity_2m_mean": daily["relative_humidity_2m_mean"][i],
                    "wind_speed_10m_mean": daily["wind_speed_10m_mean"][i],
                    "wind_direction_10m_dominant": daily["wind_direction_10m_dominant"][i],
                    "precipitation_sum": daily["precipitation_sum"][i],
                    "precipitation_hours": daily["precipitation_hours"][i],
                    "et0_fao_evapotranspiration": daily["et0_fao_evapotranspiration"][i],
                    "weather_code": daily["weather_code"][i]
                }
                all_weather_rows.append(weather_row)
            
            print(f"  ✓ Got {len(daily['time'])} days")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Rate limiting
        time.sleep(0.2)
    
    return pd.DataFrame(all_weather_rows)
