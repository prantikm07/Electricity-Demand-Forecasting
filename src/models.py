import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def seasonal_naive(df, forecast_start):
    pred = []
    for i in range(24):
        idx = forecast_start - pd.Timedelta('24h') + pd.Timedelta(f'{i}h')
        val = df.loc[df['timestamp'] == idx, 'kwh'].values
        if len(val) > 0:
            pred.append(val[0])
        else:
            pred.append(0)
    return np.array(pred)

def ridge_forecast(df, X_cols, forecast_start):
    train = df[df['timestamp'] < forecast_start]
    X_train = train[X_cols]
    y_train = train['kwh']
    model = Ridge()
    model.fit(X_train, y_train)
    future = df[(df['timestamp'] >= forecast_start) & (df['timestamp'] < forecast_start + pd.Timedelta('24h'))]
    if len(future) == 0:
        return np.zeros(24)
    X_future = future[X_cols]
    pred = model.predict(X_future)
    return pred
