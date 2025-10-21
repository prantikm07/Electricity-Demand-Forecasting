import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import QuantileRegressor

def seasonal_naive(df, forecast_start):
    pred = []
    for i in range(24):
        idx = forecast_start - pd.Timedelta(24, "h") + pd.Timedelta(i, "h")
        val = df.loc[df["timestamp"] == idx, "kwh"].values
        if len(val) > 0:
            pred.append(val[0])
        else:
            pred.append(0)
    return np.array(pred)

def ridge_forecast(df, X_cols, forecast_start):
    train = df[df["timestamp"] < forecast_start]
    X_train = train[X_cols]
    y_train = train["kwh"]
    model = Ridge()
    model.fit(X_train, y_train)
    future = df[(df["timestamp"] >= forecast_start) & (df["timestamp"] < forecast_start + pd.Timedelta(hours=24))]
    if len(future) == 0:
        return np.zeros(24)
    X_future = future[X_cols]
    pred = model.predict(X_future)
    return pred

def quantile_forecast(df, X_cols, forecast_start, quantiles=[0.1, 0.5, 0.9]):
    preds = {}
    train = df[df['timestamp'] < forecast_start]
    future = df[(df['timestamp'] >= forecast_start) & (df['timestamp'] < forecast_start + pd.Timedelta(hours=24))]
    X_train = train[X_cols]
    y_train = train['kwh']
    X_future = future[X_cols]
    for q in quantiles:
        model = QuantileRegressor(quantile=q, alpha=1)
        model.fit(X_train, y_train)
        preds[q] = model.predict(X_future)
    return preds

def calibrate_daily_energy(yhat, df, history_days=1):
    last_day = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=history_days))]
    true_total = last_day["kwh"].sum()
    forecast_total = yhat.sum()
    if forecast_total == 0:
        return yhat
    calibration_factor = true_total / forecast_total
    return yhat * calibration_factor
