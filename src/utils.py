import pandas as pd
import numpy as np

def read_demand(file_path):
    df = pd.read_csv(file_path)
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
    df = df.groupby(['x_Timestamp']).agg({'t_kWh':'sum'}).reset_index()
    df = df.set_index('x_Timestamp').resample('1h').sum()
    df = df.reset_index()
    df.columns = ['timestamp', 'kwh']
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def impute_gaps(df):
    df = df.set_index('timestamp')
    df = df.asfreq('1h')
    df['kwh'] = df['kwh'].interpolate(limit_direction='both')
    df = df.reset_index()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def cap_outliers(df):
    low, high = np.percentile(df['kwh'].dropna(), [1, 99])
    df['kwh'] = np.clip(df['kwh'], low, high)
    return df

def get_weather_data(start_date, end_date, lat=28.36, lon=79.43):
    import requests
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m&timezone=auto"
    response = requests.get(url)
    data = response.json()
    weather_df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m']
    })
    return weather_df
