import pandas as pd
import numpy as np

def read_demand(file_path):
    df = pd.read_csv(file_path)
    df['x_Timestamp'] = pd.to_datetime(df['x_Timestamp'])
    df = df.groupby(['x_Timestamp']).agg({'t_kWh':'sum'}).reset_index()
    df = df.set_index('x_Timestamp').resample('1h').sum()
    df = df.reset_index()
    df['hour'] = df['x_Timestamp'].dt.hour
    df['dayofweek'] = df['x_Timestamp'].dt.dayofweek
    return df

def impute_gaps(df):
    df = df.set_index('x_Timestamp')
    df = df.asfreq('1h')
    df['t_kWh'] = df['t_kWh'].interpolate(limit_direction='both')
    df = df.reset_index()
    df['hour'] = df['x_Timestamp'].dt.hour
    df['dayofweek'] = df['x_Timestamp'].dt.dayofweek
    return df

def cap_outliers(df):
    low, high = np.percentile(df['t_kWh'].dropna(), [1, 99])
    df['t_kWh'] = np.clip(df['t_kWh'], low, high)
    return df
