import argparse
import os
import pandas as pd
import numpy as np
from utils import read_demand, impute_gaps, cap_outliers, get_weather_data
from features import create_features
from models import seasonal_naive, ridge_forecast
from evaluation import mae, wmape, smape
from plot import plot_actuals_forecast, plot_horizon_mae
from report import create_report

parser = argparse.ArgumentParser()
parser.add_argument('--city', required=True)
parser.add_argument('--history_window', required=True)
parser.add_argument('--with_weather', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--make_plots', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--save_report', type=lambda x: x.lower() == 'true', default=False)
args = parser.parse_args()

os.makedirs('artifacts/fast_track/plots/', exist_ok=True)
os.makedirs('reports/', exist_ok=True)

city_file = f'data/{args.city.lower()}.csv'
df = read_demand(city_file)
df = impute_gaps(df)
df = cap_outliers(df)

history_days = int(args.history_window.split(':')[1])

if args.with_weather:
    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = df['timestamp'].max().strftime('%Y-%m-%d')
    lat = 28.36 if args.city.lower() == 'bareilly' else 27.49
    lon = 79.43 if args.city.lower() == 'bareilly' else 77.67
    weather_df = get_weather_data(start_date, end_date, lat, lon)
    df = df.merge(weather_df, on='timestamp', how='left')
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())

df = create_features(df, args.with_weather)

forecast_origin = df['timestamp'].iloc[-(history_days*24 + 25)]
forecast_start = forecast_origin + pd.Timedelta('1h')

actual = df[(df['timestamp'] >= forecast_start) & (df['timestamp'] < forecast_start + pd.Timedelta('24h'))]['kwh'].values

if len(actual) < 24:
    forecast_start = df['timestamp'].iloc[-25]
    actual = df[(df['timestamp'] >= forecast_start) & (df['timestamp'] < forecast_start + pd.Timedelta('24h'))]['kwh'].values

naive_pred = seasonal_naive(df, forecast_start)
X_cols = ['sin_hour','cos_hour','dayofweek','lag1','lag2','lag3','roll24']
if args.with_weather and 'temperature' in df.columns:
    X_cols.append('temperature')
ridge_pred = ridge_forecast(df, X_cols, forecast_start)

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
    horizon_errors = np.abs(actual - ridge_pred)
    plot_horizon_mae(horizon_errors, 'artifacts/fast_track/plots/horizon_mae.png')

if args.save_report:
    create_report(args.city, metrics, forecast_start, history_days, args.with_weather, 'reports/fast_track_report.pdf')

print(f"Forecast complete for {args.city}")
print(f"MAE - Naive: {mae_naive:.2f}, Ridge: {mae_ridge:.2f}")
print(f"WAPE - Naive: {wmape_naive*100:.1f}%, Ridge: {wmape_ridge*100:.1f}%")
