# Paleoclimate Analytics Project
 
Study on the Earth's climate in the last million years

Temperature anomaly since 1 million years ago:
![Long Term Temperature Anomaly](Outputs/long_term_temperature_anomaly.png)

Note the very similar patterns of carbon dioxide levels over the last million years:
![Long Term CO2 Levels](Outputs/long_term_co2_ppm.png)

Comparison of variations in orbital "wobble" with carbon dioxide levels and climate anomaly:
![Orbital parameters vs CO2 vs Temperature](Outputs/orbital_parameters_glacial_cycles_trends.png)

Temperature anomaly since 12,000 BC:
![Since Ice Age Temperature Anomaly](Outputs/since_ice_age_temperature_anomaly.png)

Temperature anomaly since 1850:
![Modern Temperature Anomaly](Outputs/modern_temperature_anomaly_forecast.png)

Estimated regression of temperature anomaly:
![Temperature Anomaly Regression](Outputs/regression_anomaly_forecast.png)

Research AGGI Index:
https://gml.noaa.gov/aggi/aggi.html

CO2 levels in past 11,000 years dataset:
https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1382

More CO2 level datasets:
https://www.ncei.noaa.gov/access/paleo-search/

Milankovitch Cycles dataset:
http://www.climatedata.info/forcing/data-downloads/

Milankovitch Simulator:
https://biocycle.atmos.colostate.edu/shiny/Milankovitch/

Modern CO2 levels dataset:
https://gml.noaa.gov/ccgg/trends/gl_data.html

2024 mean CO2 level:
https://www.statista.com/statistics/1091999/atmospheric-concentration-of-co2-historic/

Possibly incorporate recent GHG index data
Attempt at least ARIMAX, Linear Regression-based, and NHITS forecasting models in different possible future scenarios
    Possibly also include a basic TensorFlow neural network and use an evolutionary algorithm for hyperparameter optimization
Also look into using smoothed variables with s() to reduce noise and for easier plotting
    s(variable) returns a smoothed version - from the mgcv package
    Try a GAM - generalized additive model, allows a linear response variable to depend linearly on predictor variables
        Simple method of time series prediction with future exogenous variables
    Try diff-based regression, using changes in each variable to predict change in anomaly, rather than absolute amounts
