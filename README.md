# 24-Hour Electricity Demand Forecasting

A fast-track electricity demand forecasting solution for Indian cities using smart meter data and weather information.

## Overview

This project forecasts the next 24 hours of electricity demand for Bareilly using historical smart meter readings and weather data. It processes 3-minute smart meter data into hourly forecasts using Ridge Regression with engineered features.

## Features

- **Data Processing**: Converts 3-minute smart meter readings to clean hourly data
- **Weather Integration**: Fetches weather forecasts from Open-Meteo API
- **Multiple Models**: Seasonal naive baseline + Ridge Regression with engineered features
- **Uncertainty Quantification**: Provides 10th, 50th, and 90th percentile forecasts
- **Daily Calibration**: Aligns forecasts with recent consumption patterns
- **Automated Reporting**: Generates PDF reports with plots and metrics

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

**Main dependencies:**
- pandas
- numpy
- scikit-learn
- requests
- matplotlib
- fpdf

## Quick Start

Run the complete forecast pipeline with a single command:

```
python run_forecast.py --city Bareilly --history_window days:7 --with_weather true --make_plots true --save_report true
```

### Command-line Arguments

- `--city`: City name (Bareilly or Mathura)
- `--history_window`: Training data window (format: `days:7`)
- `--with_weather`: Include weather data (true/false)
- `--make_plots`: Generate visualization plots (true/false)
- `--save_report`: Create PDF report (true/false)

## Project Structure
```
├── src/
│   ├── run_forecast.py
│   ├── utils.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── plot.py
│   └── report.py
├── data/
│   ├── bareilly.csv
│   └── mathura.csv
├── artifacts/
│   └── fast_track/
│       ├── forecast_T_plus_24.csv
│       ├── metrics.csv
│       └── plots/
│           ├── actual_vs_forecast.png
│           └── horizon_mae.png
├── reports/
│   └── fast_track_report.pdf
├── .gitignore
├── Pipfile
├── Pipfile.lock
├── requirements.txt
└── README.md
```

## Data Sources

- **Smart Meter Data**: [Kaggle - Smart Meter Data (Mathura and Bareilly)](https://www.kaggle.com/datasets/jehanbhathena/smart-meter-datamathura-and-bareilly)
- **Weather Data**: [Open-Meteo API](https://open-meteo.com/en/docs)

## Methodology

### Data Preparation
1. Aggregates 3-minute readings to hourly consumption (kWh)
2. Handles missing values using forward fill
3. Removes outliers using 99th percentile capping
4. Merges weather forecasts by timestamp

### Feature Engineering
- **Temporal features**: Hour-of-day (sine/cosine encoding), day-of-week
- **Lag features**: 1-hour, 2-hour, 3-hour, 24-hour lags
- **Rolling statistics**: 24-hour rolling mean
- **Weather features**: Temperature (when available)

### Models
1. **Baseline**: Seasonal naive (previous day, same hour)
2. **Ridge Regression**: Engineered features with L2 regularization
3. **Quantile Forecasts**: 10th, 50th, 90th percentiles using residual scaling

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **WAPE** (Weighted Absolute Percentage Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)

## Output Files

### metrics.csv
Contains model performance metrics for baseline and Ridge Regression models.

### forecast_T_plus_24.csv
24-hour forecast with columns:
- `timestamp`: Hourly timestamps for next 24 hours
- `yhat`: Point forecast
- `y_p10`, `y_p50`, `y_p90`: Quantile forecasts

### Plots
- **Forecast Overlay**: Last 3 days of actual demand + 24-hour forecast
- **Horizon MAE**: Error analysis across all 24 forecast horizons

## Results Summary

The model achieves consistent performance with strong day-to-day demand patterns. The Ridge Regression model provides reliable forecasts suitable for grid operations and capacity planning.

## Reproducibility

The entire pipeline is fully reproducible with a single command. All data processing, model training, forecasting, and report generation happen automatically.

## Author

    Prantik Mukhopadhyay
- Email: [prantik25m@gmail.com](mailto:prantik25m@gmail.com)
- LinkedIn: [Prantik Mukhopadhyay](https://www.linkedin.com/in/prantikm07/)