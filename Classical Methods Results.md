To run locally, clone the repository and perform the following:
1. Install `Python 3`, `pip`, and `R`
2. Using `pip`, set up a virtual environment and install the packages in `requirements.txt` to set up the Python environment.
3. Run `Rscript install_packages.r` to set up the R environment.
4. To run the entire pipeline, run `./src/orchestrator.bat` (Windows) or `./src/orchestrator.sh` (Linux) from the root directory of the repository. This will:
    - Pull remaining data that doesn't fit in the repository
    - Creates a local SQLite database with a star schema, acting as a data warehouse of temperature measurements with dimensions
    - Runs the ETL pipeline:
        - Extract Holocene to modern temperature measurements, long-term climate, CO2 and Beryllium-10 concentrations, geomagnetic field intensity, and simulated orbital parameters
        - Transform the data into a common format for analysis
        - Load the data to populate the SQLite data warehouse
    - Creates views from the data warehouse for different analyses:
        - `raw_global_anomaly_view.parquet`: Full, globally aggregated data with high resolution, original units, and missing values
            - Used for visualization in original form
        - `long_term_global_anomaly_view.parquet`: Globally aggregated data in the train/test/forecast window with low resolution, even frequency, no missing values, and standardized units
            - Used for visualization where missing values are not desired
        - `long_term_global_anomaly_view_enriched_training.parquet`: Globally aggregated data in the train/test/forecast window with low resolution, even frequency, no missing values, standardized units, and engineered features
            - Used for machine learning
    - Performs linear regression forecasting:
        - `linear_model`: Linear regression model with no lagged anomaly values as context (no time series elements)
        - `linear_model_lagged`: Linear regression model with lagged anomaly values as context (time series elements)
    - Performs ARIMA forecasting:
        - `arima_model`: ARIMA model purely based on past anomaly values
        - `arimax_model`: ARIMAX model using both past anomaly values and other features
5. Alternatively, run `./src/orchestrator_no_etl.bat` (Windows) or `./src/orchestrator.sh` (Linux) to only perform the steps after view creation, using pre-computed ETL outputs from the repository.

Recent temperature anomaly:
![Recent Temperature Anomaly](Outputs/modern_temperature_anomaly.png)

Temperature anomaly since 12,000 BC:
![Since Ice Age Temperature Anomaly](Outputs/since_ice_age_temperature_anomaly.png)

Temperature anomaly since 800,000 BC:
![Long Term Temperature Anomaly](Outputs/long_term_temperature_anomaly.png)

Interactions between anomaly and orbital parameters:
![Orbital parameters interactions](Outputs/orbital_parameters_glacial_cycles_trends.png)

Linear model results:
![Linear Model](Outputs/linear_model_predictions.png)

Lagged Linear model results:
![Lagged Linear Model](Outputs/lagged_linear_model_predictions.png)

ARIMA model results:
![ARIMA Model](Outputs/arima_model_predictions.png)

ARIMAX model results:
![ARIMAX Model](Outputs/arimax_model_predictions.png)