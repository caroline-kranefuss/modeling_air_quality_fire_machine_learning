"""
Alternative approach: Step-by-step functions you can run interactively
"""
import pandas as pd
import weather_func

# ============================================================================
# FUNCTION 1: Identify and save missing data
# ============================================================================

def separate_missing_data(df, output_file='missing_weather_rows.csv'):
    """
    Separates rows with complete weather data from those missing data.
    
    Returns:
        df_complete: DataFrame with weather data
        df_missing: DataFrame without weather data
    """
    print(f"Total rows: {len(df)}")
    print(f"Rows with weather: {df['temperature_2m_mean'].notna().sum()}")
    print(f"Rows missing weather: {df['temperature_2m_mean'].isna().sum()}")
    
    df_complete = df[df['temperature_2m_mean'].notna()].copy()
    df_missing = df[df['temperature_2m_mean'].isna()].copy()
    
    # Save missing rows
    df_missing.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df_missing)} missing rows to: {output_file}")
    
    # Show which locations are missing
    if len(df_missing) > 0:
        print("\nMissing data by location:")
        missing_summary = df_missing.groupby(['site_name', 'latitude', 'longitude']).size()
        print(missing_summary.sort_values(ascending=False).head(10))
    
    return df_complete, df_missing


# ============================================================================
# FUNCTION 2: Retry fetching weather for missing rows
# ============================================================================

def retry_weather_fetch(df_missing, weather_api):
    """
    Attempts to fetch weather data for rows that are missing it.
    
    Returns:
        weather_df: DataFrame with weather data for previously missing rows
    """
    print(f"\nRetrying weather fetch for {len(df_missing)} rows...")
    print(f"Unique locations: {df_missing.groupby(['latitude', 'longitude']).ngroups}")
    
    # Try batch fetch first
    weather_df = weather_func.fetch_weather_batch_by_location(weather_api, df_missing)
    
    print(f"✓ Retrieved weather for {len(weather_df)} date-location combinations")
    
    return weather_df


# ============================================================================
# FUNCTION 3: Merge retry results back in
# ============================================================================

def merge_retry_results(df_complete, df_missing, weather_retry):
    """
    Merges the retry weather results with the missing data, 
    then combines with complete data.
    
    Returns:
        df_final: Complete dataset with retry results merged in
    """
    # Weather column names
    weather_cols = [
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'relative_humidity_2m_mean', 'wind_speed_10m_mean', 
        'wind_direction_10m_dominant', 'precipitation_sum', 
        'precipitation_hours', 'et0_fao_evapotranspiration', 'weather_code'
    ]
    
    # Drop weather columns from missing data (they're all NaN)
    df_missing_clean = df_missing.drop(columns=weather_cols, errors='ignore')
    
    # Merge retry results
    df_missing_updated = df_missing_clean.merge(
        weather_retry,
        on=['latitude', 'longitude', 'date'],
        how='left'
    )
    
    # Combine with complete data
    df_final = pd.concat([df_complete, df_missing_updated], ignore_index=True)
    df_final = df_final.sort_values(['date', 'site_id']).reset_index(drop=True)
    
    print(f"\nFinal results:")
    print(f"  Total rows: {len(df_final)}")
    print(f"  Rows with weather: {df_final['temperature_2m_mean'].notna().sum()}")
    print(f"  Rows still missing: {df_final['temperature_2m_mean'].isna().sum()}")
    
    return df_final


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def complete_retry_workflow(input_file='air_quality_with_weather.csv',
                           output_file='air_quality_with_weather_final.csv'):
    """
    Complete workflow to handle missing weather data.
    """
    weather_api = 'https://archive-api.open-meteo.com/v1/archive'
    
    # Step 1: Load and separate
    print("="*70)
    print("STEP 1: Loading data and identifying missing rows")
    print("="*70)
    df = pd.read_csv(input_file)
    df_complete, df_missing = separate_missing_data(df)
    
    if len(df_missing) == 0:
        print("\nNo missing data! Nothing to retry.")
        return df
    
    # Step 2: Retry fetch
    print("\n" + "="*70)
    print("STEP 2: Retrying weather fetch")
    print("="*70)
    weather_retry = retry_weather_fetch(df_missing, weather_api)
    
    # Step 3: Merge results
    print("\n" + "="*70)
    print("STEP 3: Merging results")
    print("="*70)
    df_final = merge_retry_results(df_complete, df_missing, weather_retry)
    
    # Step 4: Save
    print("\n" + "="*70)
    print("STEP 4: Saving results")
    print("="*70)
    df_final.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    # Check if any still missing
    still_missing = df_final[df_final['temperature_2m_mean'].isna()]
    if len(still_missing) > 0:
        still_missing.to_csv('still_missing_weather.csv', index=False)
        print(f"⚠ {len(still_missing)} rows still missing - saved to still_missing_weather.csv")
    else:
        print("✓ All rows now have weather data!")
    
    return df_final


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the complete workflow
    df_final = complete_retry_workflow(
        input_file='air_quality_with_weather.csv',
        output_file='air_quality_with_weather_final.csv'
    )
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


# ============================================================================
# OR use step-by-step interactively:
# ============================================================================
"""
# In a notebook or interactive session:

import pandas as pd
import retry_missing_weather as retry

# Load your data
df = pd.read_csv('air_quality_with_weather.csv')

# Step 1: Separate complete vs missing
df_complete, df_missing = retry.separate_missing_data(df)

# Step 2: Retry fetching
weather_api = 'https://archive-api.open-meteo.com/v1/archive'
weather_retry = retry.retry_weather_fetch(df_missing, weather_api)

# Step 3: Merge back together
df_final = retry.merge_retry_results(df_complete, df_missing, weather_retry)

# Step 4: Save
df_final.to_csv('air_quality_with_weather_final.csv', index=False)
"""