import numpy as np

def create_features(df, with_weather=False):
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['lag1'] = df['kwh'].shift(1)
    df['lag2'] = df['kwh'].shift(2)
    df['lag3'] = df['kwh'].shift(3)
    df['roll24'] = df['kwh'].rolling(24).mean()
    df = df.dropna()
    return df
