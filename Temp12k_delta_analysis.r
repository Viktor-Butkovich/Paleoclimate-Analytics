# Deprecated

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(glmnet)
library(mgcv)
library(scales)


anomaly_df <- read.csv("Data/anomaly_training_ready_with_deltas.csv")

anomaly <- anomaly_df$anomaly

original_anomaly <- anomaly_df$anomaly[1] - anomaly_df$delta_anomaly[1]

anomaly_df <- anomaly_df %>%
    select(
        year_bin,
        anomaly,
        delta_anomaly,
        # delta_eccentricity_wave,
        delta_eccentricity,
        delta_co2_radiative_forcing,
        delta_obliquity,
        # delta_insolation,
        # delta_global_insolation,
        delta_perihelion
    )

validation_line <- -100000
anomaly_train <- anomaly_df %>% filter(year_bin <= validation_line)
anomaly_validation <- anomaly_df %>% filter(year_bin > validation_line)

print("Training...")
ctrl <- trainControl(method = "cv", number = 10, savePredictions = "final", classProbs = FALSE)


model_config <- list(
    lm = TRUE,
    nnet = FALSE,
    rpart = FALSE,
    rf = FALSE
) # Choose models to include

models <- list()

# Train individual models
if (model_config$lm) {
    model_lm <- train(delta_anomaly ~ . - year_bin - anomaly, data = anomaly_train, method = "lm", trControl = ctrl)
    models[["lm"]] <- model_lm
}
if (model_config$nnet) {
    model_nnet <- train(delta_anomaly ~ . - year_bin - anomaly, data = anomaly_train, method = "nnet", trControl = ctrl, tuneGrid = expand.grid(.size = seq(5, 20, 5), .decay = seq(0.1, 0.5, 0.1)), preProcess = c("center", "scale"), linout = TRUE, trace = FALSE)
    models[["nnet"]] <- model_nnet
}
if (model_config$rpart) {
    model_rpart <- train(delta_anomaly ~ . - year_bin - anomaly, data = anomaly_train, method = "rpart", trControl = ctrl, tuneLength = 200)
    models[["rpart"]] <- model_rpart
}
if (model_config$rf) {
    model_rf <- train(delta_anomaly ~ . - year_bin - anomaly, data = anomaly_train, method = "rf", trControl = ctrl, tuneGrid = expand.grid(.mtry = c(1:10)))
    models[["rf"]] <- model_rf
}

predict_ensemble <- function(models, newdata, ...) {
    predictions <- lapply(models, function(model) predict(model, newdata = newdata, ...))
    return(as.data.frame(predictions))
}

weighted_predict_ensemble <- function(ensemble_weight_model, models, newdata, ...) {
    predictions <- predict_ensemble(models, newdata = newdata, ...)
    combined_predictions <- predict(ensemble_weight_model, newdata = predictions)
    return(combined_predictions)
}

ensemble_training_predictions <- predict_ensemble(models, newdata = anomaly_train)
ensemble_training_predictions$delta_anomaly <- anomaly_train$delta_anomaly
ensemble_model <- train(delta_anomaly ~ ., data = ensemble_training_predictions, method = "lm", trControl = ctrl)

print(summary(ensemble_model))

anomaly_train <- anomaly_train %>%
    mutate(predicted_delta_anomaly = weighted_predict_ensemble(ensemble_model, models, newdata = anomaly_train)) %>%
    mutate(predicted_anomaly = lag(anomaly) + weighted_predict_ensemble(ensemble_model, models, newdata = anomaly_train))

cumulative_delta_forecast <- function(last_observed_anomaly, future_exogenous) {
    predicted_delta_anomaly <- weighted_predict_ensemble(ensemble_model, models, newdata = future_exogenous)
    return(last_observed_anomaly + cumsum(predicted_delta_anomaly))
}

anomaly_validation$predicted_delta_anomaly <- weighted_predict_ensemble(ensemble_model, models, newdata = anomaly_validation)
anomaly_validation$predicted_anomaly <- cumulative_delta_forecast(tail(anomaly_train$anomaly, 1), anomaly_validation)
# Each predicted anomaly within previous data is predicted from the previous anomaly
# Future anomalies are extrapolated from the last known observation and the cumulative sum of predicted deltas, based on deltas of global parameters

anomaly_df <- bind_rows(anomaly_train, anomaly_validation)


plot <- ggplot(anomaly_df, aes(x = year_bin)) +
    geom_line(aes(y = anomaly, color = "Actual Anomaly")) +
    geom_line(aes(y = predicted_anomaly, color = "Predicted Anomaly")) +
    geom_vline(xintercept = validation_line, linetype = "dashed", color = "blue") +
    annotate("rect", xmin = validation_line, xmax = max(anomaly_df$year_bin), ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "grey") +
    annotate("text", x = validation_line + 10000, y = 8, label = "Validation", hjust = 0, color = "black") +
    annotate("text", x = validation_line - 10000, y = 8, label = "Training", hjust = 1, color = "black") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
    labs(
        title = "Forecasted Climate Anomaly (-650 kya to 1700)",
        x = "Year Bin",
        y = "Anomaly",
        color = "Legend"
    ) +
    theme_classic() +
    scale_y_continuous(limits = c(-10, 10)) +
    scale_x_continuous(labels = scales::comma)
ggsave("Outputs/delta_based_anomaly_forecast.png", plot, width = 10, height = 6)

plot <- ggplot(anomaly_df, aes(x = year_bin)) +
    geom_line(aes(y = delta_anomaly, color = "Actual Delta Anomaly")) +
    geom_line(aes(y = predicted_delta_anomaly, color = "Predicted Delta Anomaly")) +
    geom_vline(xintercept = validation_line, linetype = "dashed", color = "blue") +
    annotate("rect", xmin = validation_line, xmax = max(anomaly_df$year_bin), ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "grey") +
    annotate("text", x = validation_line + 10000, y = 2.5, label = "Validation", hjust = 0, color = "black") +
    annotate("text", x = validation_line - 10000, y = 2.5, label = "Training", hjust = 1, color = "black") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
    labs(
        title = "Forecasted Climate Anomaly (-650 kya to 1700)",
        x = "Year Bin",
        y = "Anomaly",
        color = "Legend"
    ) +
    theme_classic() +
    # scale_y_continuous(limits = c(-10, 10)) +
    scale_x_continuous(labels = scales::comma)
ggsave("Outputs/delta_anomaly_forecast.png", plot, width = 10, height = 6)
