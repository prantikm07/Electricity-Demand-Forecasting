import matplotlib.pyplot as plt
import pandas as pd

def plot_actuals_forecast(df, forecast, forecast_start, save_path):
    plt.figure(figsize=(10,4))
    mask = df['x_Timestamp'] >= forecast_start - pd.Timedelta('72h')
    plt.plot(df[mask]['x_Timestamp'], df[mask]['t_kWh'], label='Actual')
    forecast_times = [forecast_start + pd.Timedelta(f'{i}h') for i in range(24)]
    plt.plot(forecast_times, forecast, label='Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_mae(mae_list, save_path):
    plt.figure(figsize=(8,4))
    plt.plot(range(1,25), mae_list)
    plt.xlabel('horizon')
    plt.ylabel('MAE')
    plt.tight_layout()
    plt.savefig(save_path)
