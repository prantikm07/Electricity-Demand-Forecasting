from fpdf import FPDF
import pandas as pd

def create_report(city, metrics_df, forecast_start, history_days, with_weather, save_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, '24-Hour Electricity Demand Forecasting Report', ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Problem Statement', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f'This project forecasts hourly electricity demand for {city} city 24 hours ahead using smart meter data. The objective is to provide accurate short-term forecasts using a fast-track, reproducible pipeline with minimal data requirements and a 7-day training window.')
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Preparation', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f'Raw 3-minute smart meter readings were aggregated to hourly totals by summing kWh across all meters. Missing values were imputed using linear interpolation, and extreme outliers were capped at the 1st and 99th percentiles to ensure robustness. The final dataset contains continuous hourly timestamps in IST timezone.')
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Methods', ln=True)
    pdf.set_font('Arial', '', 11)
    weather_text = 'Temperature data was merged from Open-Meteo API.' if with_weather else 'No weather data was used.'
    pdf.multi_cell(0, 6, f'Two forecasting models were implemented. The baseline seasonal naive model predicts each hour using the same hour from the previous day. The Ridge regression model uses engineered features including hour-of-day (sine/cosine encoded), day-of-week, lagged demand values (1, 2, 3 hours), and a 24-hour rolling mean. {weather_text} The training window covers the last {history_days} days before the forecast origin.')
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Results', ln=True)
    pdf.set_font('Arial', '', 11)
    
    naive_mae = metrics_df[metrics_df['model']=='naive']['MAE'].values[0]
    naive_wape = metrics_df[metrics_df['model']=='naive']['WAPE'].values[0]
    naive_smape = metrics_df[metrics_df['model']=='naive']['sMAPE'].values[0]
    ridge_mae = metrics_df[metrics_df['model']=='ridge']['MAE'].values[0]
    ridge_wape = metrics_df[metrics_df['model']=='ridge']['WAPE'].values[0]
    ridge_smape = metrics_df[metrics_df['model']=='ridge']['sMAPE'].values[0]
    
    pdf.multi_cell(0, 6, f'The seasonal naive baseline achieved MAE of {naive_mae:.2f} kWh, WAPE of {naive_wape*100:.1f}%, and sMAPE of {naive_smape:.1f}%. The Ridge regression model achieved MAE of {ridge_mae:.2f} kWh, WAPE of {ridge_wape*100:.1f}%, and sMAPE of {ridge_smape:.1f}%. The plots show actual vs forecast comparison and horizon-wise error distribution.')
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Takeaways and Next Steps', ln=True)
    pdf.set_font('Arial', '', 11)
    
    if ridge_mae < naive_mae:
        conclusion = 'The Ridge regression model outperformed the baseline, demonstrating that engineered features add predictive value.'
    else:
        conclusion = 'The baseline outperformed the Ridge model, suggesting that in this data-limited scenario, more sophisticated features or models are needed.'
    
    pdf.multi_cell(0, 6, f'{conclusion} Next steps include expanding the training window, incorporating additional weather variables, testing ensemble methods, and implementing quantile forecasting for uncertainty estimation. Daily energy calibration could further improve accuracy by aligning forecasts with recent consumption patterns.')
    
    pdf.output(save_path)
