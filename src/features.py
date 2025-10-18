import numpy as np
import pandas as pd

def create_features(df):
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['lag1'] = df['t_kWh'].shift(1)
    df['lag2'] = df['t_kWh'].shift(2)
    df['lag3'] = df['t_kWh'].shift(3)
    df['roll24'] = df['t_kWh'].rolling(24).mean()
    df = df.dropna()
    return df
