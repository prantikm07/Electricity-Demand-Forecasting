import argparse
import pandas as pd
from utils import read_demand, impute_gaps, cap_outliers
from features import create_features
from models import seasonal_naive, ridge_forecast
from evaluation import mae, wmape, smape
from plot import plot_actuals_forecast, plot_mae
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--city')
parser.add_argument('--history_window')
parser.add_argument('--with_weather', type=bool)
parser.add_argument('--make_plots', type=bool)
parser.add_argument('--save_report', type=bool)
args = parser.parse_args()

df = read_demand(f'data/bareilly.csv')
df = impute_gaps(df)
df = cap_outliers(df)
df = create_features(df)

forecast_origin = df['x_Timestamp'].iloc[-25]
forecast_start = forecast_origin + pd.Timedelta('1h')
actual = df[(df['x_Timestamp'] >= forecast_start) & (df['x_Timestamp'] < forecast_start + pd.Timedelta('24h'))]['t_kWh'].values

naive_pred = seasonal_naive(df, forecast_start)
ridge_pred = ridge_forecast(df, ['sin_hour','cos_hour','dayofweek','lag1','lag2','lag3','roll24'], forecast_start)

mae_naive = mae(actual, naive_pred)
mae_ridge = mae(actual, ridge_pred)
wmape_naive = wmape(actual, naive_pred)
wmape_ridge = wmape(actual, ridge_pred)
smape_naive = smape(actual, naive_pred)
smape_ridge = smape(actual, ridge_pred)

metrics = pd.DataFrame({
    'model': ['naive','ridge'],
    'MAE': [mae_naive, mae_ridge],
    'WAPE': [wmape_naive, wmape_ridge],
    'sMAPE': [smape_naive, smape_ridge]
})
metrics.to_csv('artifacts/fast_track/metrics.csv', index=False)

forecast_df = pd.DataFrame({
    'timestamp': [forecast_start + pd.Timedelta(f'{i}h') for i in range(24)],
    'yhat': ridge_pred
})
forecast_df.to_csv('artifacts/fast_track/forecast_T_plus_24.csv', index=False)

if args.make_plots:
    plot_actuals_forecast(df, ridge_pred, forecast_start, 'artifacts/fast_track/plots/actual_vs_forecast.png')
    plot_mae(np.abs(actual - ridge_pred), 'artifacts/fast_track/plots/horizon_mae.png')

