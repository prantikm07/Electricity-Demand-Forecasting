import pandas as pd
import numpy as np

def read_demand(filepath):
    df = pd.read_csv(filepath)
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
    df = df.groupby('x_Timestamp').agg({'t_kWh': 'sum'}).reset_index()

    df = df.set_index('x_Timestamp').resample('1h').sum()
    df = df.reset_index()
    df.columns = ['timestamp', 'kwh']
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def impute_gaps(df, method="interpolate"):
    df = df.set_index("timestamp")
    df = df.asfreq("1h")
    n_missing = df['kwh'].isna().sum()
    if method == "interpolate":
        df["kwh"] = df["kwh"].interpolate(limit_direction="both")
    elif method == "zero":
        df["kwh"] = df["kwh"].fillna(0)
    elif method == "backfill":
        df["kwh"] = df["kwh"].fillna(method="bfill")
    else:
        raise ValueError(f"Unknown gap fill method: {method}")
    print(f"Imputed {n_missing} missing hourly values using {method} method.")

    df = df.reset_index()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df

def cap_outliers(df):
    series = df["kwh"].copy()
    low, high = np.percentile(series.dropna(), 1), np.percentile(series.dropna(), 99)
    df["kwh_capped"] = np.clip(series, low, high)
    capped_count = (series != df["kwh_capped"]).sum()
    print(f"Capped {capped_count} outlier points outside [{low:.2f}, {high:.2f}] kWh.")
    
    df["kwh"] = df["kwh_capped"]
    df = df.drop(columns="kwh_capped")
    return df

def get_weather_data(start_date, end_date, lat=28.36, lon=79.43):
    import requests
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&timezone=auto"
    response = requests.get(url)
    data = response.json()
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(data['hourly']['time']),
        "temperature": data['hourly']['temperature_2m']
    })
    return weather_df
