library(tidyverse)
library(reshape2)
library(scales)
library(patchwork)
library(olsrr)

present_line <- 2025
prediction_line <- 200000 # Don't attempt predictions past year 200,000
test_start <- -500000
test_end <- -300000
anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view_enriched_training.csv") %>%
    filter(year_bin <= prediction_line)

omit_enriched <- FALSE
if (omit_enriched) {
    anomaly_df <- anomaly_df %>%
        select(-contains("delta"), -contains("squared"))
}

test_anomaly_df <- anomaly_df %>% filter(year_bin > present_line | (year_bin <= test_start & year_bin <= test_end))
train_anomaly_df <- anomaly_df %>% anti_join(test_anomaly_df, by = "year_bin") # Train data is non-test data

produce_stepwise_model <- function(model_type, data) {
    # Predict anomaly from all non-delta attributes
    stepwise_model <- ols_step_both_p(
        lm(anomaly ~ ., data = data %>% select(-year_bin)),
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
    select(year_bin, anomaly, pred_anomaly)
write_csv(pred_anomaly_df, "Outputs/linear_model_predictions.csv")

linear_model <- produce_stepwise_model(lm, train_anomaly_df)
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = linear_model_predicted_anomaly) %>%
    select(year_bin, anomaly, pred_anomaly)
write_csv(pred_anomaly_df, "Outputs/linear_model_predictions_lagged.csv")

print("Saved predictions to csv")
