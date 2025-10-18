import matplotlib.pyplot as plt
import pandas as pd

def plot_actuals_forecast(df, forecast, forecast_start, save_path):
    plt.figure(figsize=(12,5))
    mask = df['timestamp'] >= forecast_start - pd.Timedelta('72h')
    plt.plot(df[mask]['timestamp'], df[mask]['kwh'], label='Actual', marker='o', markersize=3)
    forecast_times = [forecast_start + pd.Timedelta(f'{i}h') for i in range(len(forecast))]
    plt.plot(forecast_times, forecast, label='Forecast', marker='x', markersize=4, linewidth=2)
    plt.xlabel('Timestamp')
    plt.ylabel('kWh')
    plt.title('Last 3 Days Actuals with 24-Hour Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_horizon_mae(mae_list, save_path):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(mae_list)+1), mae_list, marker='o')
    plt.xlabel('Forecast Horizon (hours ahead)')
    plt.ylabel('MAE (kWh)')
    plt.title('Horizon-Wise MAE for 24-Hour Forecast')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
