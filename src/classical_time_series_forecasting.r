suppressPackageStartupMessages({
    suppressWarnings(library(tidyverse))
    suppressWarnings(library(jsonlite))
    suppressWarnings(library(forecast))
    suppressWarnings(library(olsrr))
    suppressWarnings(library(here))
    suppressWarnings(library(arrow))
})
options(warn = -1) # Suppress warnings

# Read in the JSON configuration file
config <- fromJSON(here("prediction_config.json"))

anomaly_df <- read_parquet(here("Outputs", "long_term_global_anomaly_view_enriched_training.parquet")) %>%
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
write.csv(pred_anomaly_df, here("Outputs", "arima_model_predictions.csv"))


train_anomaly_exog_df <- anomaly_df %>%
    filter(year_bin <= config$present) %>%
    select(-year_bin, -anomaly)
forecast_exog_df <- anomaly_df %>%
    filter(year_bin > config$present) %>%
    select(-year_bin, -anomaly)

stepwise_exog <- ols_step_both_p(
    lm(train_anomaly_df$anomaly ~ ., data = train_anomaly_exog_df),
    details = FALSE
)$model
stepwise_exog_features <- attr(terms(stepwise_exog$model), "term.labels") # Only use the exogeneous features selected by the stepwise model

train_anomaly_exog_df <- train_anomaly_exog_df %>% select(all_of(stepwise_exog_features))
forecast_exog_df <- forecast_exog_df %>% select(all_of(stepwise_exog_features))

arimax_model <- auto.arima(
    anomaly_ts,
    xreg = as.matrix(train_anomaly_exog_df),
    stepwise = FALSE,
    approximation = FALSE,
    seasonal = FALSE
)

print(summary(arimax_model))

arimax_forecasts <- forecast(
    arimax_model,
    xreg = as.matrix(forecast_exog_df),
    h = horizon_length
)
arimax_pred_anomaly <- c(train_anomaly_df$anomaly, arimax_forecasts$mean)

pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = arimax_pred_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write.csv(pred_anomaly_df, here("Outputs", "arimax_model_predictions.csv"))

print("Saved predictions to csv")
