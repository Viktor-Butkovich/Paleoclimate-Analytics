cd /d "%~dp0"
:: Set working directory to this script's directory
python3 ETL.py
python3 view_creation.py
Rscript classical_regression.r
Rscript classical_time_series_forecasting.r
python3 torch_forecasting.py
python3 genetic_torch_forecasting.py
Rscript visualization.r