import numpy as np

def create_features(df, with_weather=False):
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["lag1"] = df["kwh"].shift(1)
    df["lag2"] = df["kwh"].shift(2)
    df["lag3"] = df["kwh"].shift(3)
    df["roll24"] = df["kwh"].rolling(24).mean()
    if with_weather and "temperature" in df.columns:
        df["temp_lag1"] = df["temperature"].shift(1)
        df["temp_roll24"] = df["temperature"].rolling(24).mean()
        df = df.dropna(subset=["lag1", "lag2", "lag3", "roll24", "temp_lag1", "temp_roll24"])
    else:
        df = df.dropna(subset=["lag1", "lag2", "lag3", "roll24"])
    return df
