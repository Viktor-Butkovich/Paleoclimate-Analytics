cd /d "%~dp0"
:: Set working directory to this script's directory
Rscript classical_regression.r
python3 torch_forecasting.py
python3 genetic_torch_forecasting.py
Rscript classical_time_series_forecasting.r
Rscript visualization.r