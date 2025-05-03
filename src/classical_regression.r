suppressPackageStartupMessages({
    suppressWarnings(library(tidyverse))
    suppressWarnings(library(olsrr))
    suppressWarnings(library(jsonlite))
    suppressWarnings(library(caret))
    suppressWarnings(library(here))
    suppressWarnings(library(arrow))
})
options(warn = -1) # Suppress warnings
set.seed(42)

# Read in the JSON configuration file
config <- fromJSON(here("prediction_config.json"))

anomaly_df <- read_parquet(here("Outputs", "long_term_global_anomaly_view_enriched_training.parquet")) %>%
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

k_fold_train_eval <- function(model_type, data, num_folds) {
    folds <- createFolds(data$anomaly, k = k, list = TRUE, returnTrain = TRUE)
    val_mse_values <- c()
    pred_values <- c()

    for (i in seq_along(folds)) {
        train_indices <- folds[[i]]
        fold_train_anomaly_df <- train_anomaly_df[train_indices, ]
        fold_validation_anomaly_df <- train_anomaly_df[-train_indices, ]

        model <- produce_stepwise_model(model_type, fold_train_anomaly_df)
        validation_anomaly <- predict_model(model, fold_validation_anomaly_df)

        val_mse <- mean((fold_validation_anomaly_df$anomaly - validation_anomaly)^2, na.rm = TRUE)
        val_mse_values <- c(val_mse_values, val_mse)

        pred_anomaly <- predict_model(model, anomaly_df)
        pred_values[[i]] <- pred_anomaly
    }

    aggregated_pred_anomaly <- Reduce(`+`, pred_values) / num_folds
    return(list(val_mse = mean(val_mse_values), pred_anomaly = aggregated_pred_anomaly))
}

k <- 5

linear_model_results <- k_fold_train_eval(lm, train_anomaly_df %>% select(-contains("lagged")), k)
print(sprintf("Linear model MSE: %.5f", linear_model_results$val_mse))
linear_model_pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = linear_model_results$pred_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write_parquet(linear_model_pred_anomaly_df, here("Outputs", "linear_model_predictions.parquet"))


lagged_linear_model_results <- k_fold_train_eval(lm, train_anomaly_df, k)
print(sprintf("Lagged linear model MSE: %.5f", lagged_linear_model_results$val_mse))
lagged_linear_model_pred_anomaly_df <- anomaly_df %>%
    mutate(pred_anomaly = lagged_linear_model_results$pred_anomaly) %>%
    mutate(
        anomaly = round(anomaly, config$anomaly_decimal_places),
        pred_anomaly = round(pred_anomaly, config$anomaly_decimal_places)
    ) %>%
    select(year_bin, anomaly, pred_anomaly)
write_parquet(lagged_linear_model_pred_anomaly_df, here("Outputs", "lagged_linear_model_predictions.parquet"))

print("Saved predictions to parquet")

# Update scoreboard.json with the best individual's fitness
scoreboard_path <- here("Outputs", "scoreboard.json")

# Load the existing scoreboard
scoreboard <- fromJSON(scoreboard_path)

# Update the "linear_model" and "lagged_linear_model" entries
scoreboard$linear_model <- round(linear_model_results$val_mse, config$anomaly_decimal_places)
scoreboard$lagged_linear_model <- round(lagged_linear_model_results$val_mse, config$anomaly_decimal_places)

# Save the updated scoreboard
write(toJSON(scoreboard, pretty = TRUE, auto_unbox = TRUE), scoreboard_path)

print("Updated scoreboard with linear_model and lagged_linear_model MSE")
