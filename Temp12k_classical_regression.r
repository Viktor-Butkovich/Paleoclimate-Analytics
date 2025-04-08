library(tidyverse)
library(reshape2)
library(scales)
library(patchwork)
library(olsrr)

anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view_enriched_training.csv") %>% filter(year_bin <= 2025)
anomaly_df_raw <- read.csv("Outputs/raw_global_anomaly_view.csv") %>% filter(year_bin <= 2025)

omit_enriched <- TRUE
if (omit_enriched) {
    anomaly_df <- anomaly_df %>% select(-contains("delta"), -contains("squared"))
}

validation_line <- -100000
train_anomaly_df <- anomaly_df %>% filter(year_bin < validation_line)
validation_anomaly_df <- anomaly_df %>% filter(year_bin >= validation_line)

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

plot_predictions <- function(data, predictions, validation_line, file_path) {
    plot_df <- data %>% mutate(predictions = predictions)
    ggplot(data, aes(x = year_bin)) +
        geom_line(aes(y = anomaly, color = "Actual Anomaly")) +
        geom_line(aes(y = predictions, color = "Predicted Anomaly")) +
        labs(
            x = "Time Step",
            y = "Anomaly",
            color = "Legend",
            title = "Actual vs Predicted Anomaly"
        ) +
        theme_classic() +
        scale_y_continuous(limits = c(-10, 10)) +
        scale_x_continuous(labels = scales::comma) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
        annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
        annotate("rect", xmin = validation_line, xmax = max(data$year_bin), ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "grey") +
        annotate("text", x = validation_line + 10000, y = 8, label = "Validation", hjust = 0, color = "black") +
        annotate("text", x = validation_line - 10000, y = 8, label = "Training", hjust = 1, color = "black")
    ggsave(paste("Outputs/", file_path, ".png", sep = ""), width = 10, height = 6)
}

linear_model <- produce_stepwise_model(lm, train_anomaly_df %>% select(-contains("lagged")))
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
plot_predictions(anomaly_df, linear_model_predicted_anomaly, validation_line, "linear_model_predictions")

linear_model <- produce_stepwise_model(lm, train_anomaly_df)
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
plot_predictions(anomaly_df, linear_model_predicted_anomaly, validation_line, "linear_model_predictions_lagged")
