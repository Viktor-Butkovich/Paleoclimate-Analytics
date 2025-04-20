suppressPackageStartupMessages({
    library(tidyverse)
    library(olsrr)
    library(jsonlite)
})

# Read in the JSON configuration file
config <- fromJSON("prediction_config.json")

anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view_enriched_training.csv") %>%
    filter(year_bin <= config$forecast_end)

test_anomaly_df <- anomaly_df %>% filter(year_bin > config$present | (year_bin <= config$test_start & year_bin <= config$test_end))
train_anomaly_df <- anomaly_df %>% anti_join(test_anomaly_df, by = "year_bin") # Train data is non-test data

produce_stepwise_model <- function(model_type, data) {
    # Predict anomaly from all non-delta attributes
    stepwise_model <- ols_step_both_p(
        model_type(anomaly ~ ., data = data %>% select(-year_bin)),
        details = FALSE
    )
    return(stepwise_model$model)
}

produce_model <- function(model_type, data) {
    # Predict anomaly from all non-delta attributes
    model <- model_type(anomaly ~ ., data = data %>% select(-year_bin))
    return(model)
}

predict_model <- function(model, data) {
    # Use raw model predictions of anomaly
    predicted_anomalies <- predict(model, newdata = data)
    return(predicted_anomalies)
}

linear_model <- produce_stepwise_model(lm, train_anomaly_df %>% select(-contains("lagged")))
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = linear_model_predicted_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write_csv(pred_anomaly_df, "Outputs/linear_model_predictions.csv")

linear_model <- produce_stepwise_model(lm, train_anomaly_df)
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = linear_model_predicted_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write_csv(pred_anomaly_df, "Outputs/linear_model_predictions_lagged.csv")

print("Saved predictions to csv")
