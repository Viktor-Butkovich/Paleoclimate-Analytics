cd "$(dirname "$0")"
# Set working directory to this script's directory
python3 ETL.py
python3 view_creation.py
Rscript classical_regression.r
python3 genetic_torch_forecasting.py
Rscript classical_time_series_forecasting.r
Rscript visualization.r