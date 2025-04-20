suppressPackageStartupMessages({
    library(tidyverse)
    library(jsonlite)
    library(forecast)
})

# Read in the JSON configuration file
config <- fromJSON("prediction_config.json")

anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view_enriched_training.csv") %>%
    filter(year_bin <= config$forecast_end)

train_anomaly_df <- anomaly_df %>%
    filter(year_bin <= config$present) %>%
    select(year_bin, anomaly)

anomaly_ts <- ts(train_anomaly_df$anomaly, start = train_anomaly_df$year_bin[1], frequency = 1 / 2000)
arima_model <- auto.arima(anomaly_ts, stepwise = FALSE, approximation = FALSE, seasonal = FALSE)

print(summary(arima_model))

horizon_length <- nrow(anomaly_df) - nrow(train_anomaly_df)
arima_forecast <- forecast(arima_model, h = horizon_length)

arima_pred_anomaly <- c(train_anomaly_df$anomaly, arima_forecast$mean)

pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = arima_pred_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write_csv(pred_anomaly_df, "Outputs/arima_model_predictions.csv")

print("Saved predictions to csv")
