import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def seasonal_naive(df, forecast_start_col):
    pred = []
    for i in range(24):
        idx = forecast_start_col - pd.Timedelta('24h') + pd.Timedelta(f'{i}h')
        val = df.loc[df['x_Timestamp'] == idx, 't_kWh'].values[0]
        pred.append(val)
    return np.array(pred)

def ridge_forecast(df, X_cols, forecast_start_col):
    train = df[df['x_Timestamp'] < forecast_start_col]
    X_train = train[X_cols]
    y_train = train['t_kWh']
    model = Ridge()
    model.fit(X_train, y_train)
    future = df[(df['x_Timestamp'] >= forecast_start_col) & (df['x_Timestamp'] < forecast_start_col + pd.Timedelta('24h'))]
    X_future = future[X_cols]
    pred = model.predict(X_future)
    return pred
