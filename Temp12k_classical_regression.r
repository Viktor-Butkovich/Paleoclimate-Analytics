library(tidyverse)
library(reshape2)
library(scales)
library(patchwork)

anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view.csv")
anomaly_df_raw <- read.csv("Outputs/raw_global_anomaly_view.csv")

omit_co2 <- TRUE
if (omit_co2) {
    anomaly_df <- anomaly_df %>% select(-c(co2_ppm, co2_radiative_forcing, delta_co2_ppm, delta_co2_radiative_forcing))
}

validation_line <- -100000
train_anomaly_df <- anomaly_df %>% filter(year_bin < validation_line)
validation_anomaly_df <- anomaly_df %>% filter(year_bin >= validation_line)

produce_delta_model <- function(model_type, data) {
    # Predict delta anomaly from all delta attributes
    return(model_type(delta_anomaly ~ ., data = data %>% select(starts_with("delta"))))
}

predict_delta_model <- function(model, data, validation_line) {
    train_df <- data %>% filter(year_bin < validation_line)
    validation_df <- data %>% filter(year_bin >= validation_line)

    predicted_train_anomalies <- cumsum(predict(model, newdata = train_df)) + train_df$anomaly[1]
    # Predict as cumulative sum of delta anomalies plus first anomaly value (accuracy will decrease farther from start)

    predicted_validation_anomalies <- cumsum(predict(model, newdata = validation_df)) + tail(train_df$anomaly, 1)
    # Predict as cumulative sum of delta anomalies plus last known anomaly value (accuracy will decrease farther from start)

    return(c(predicted_train_anomalies, predicted_validation_anomalies))
}

produce_model <- function(model_type, data) {
    # Predict anomaly from all non-delta attributes
    return(model_type(anomaly ~ ., data = data %>% select(-c(starts_with("delta_"), year_bin))))
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

linear_delta_model <- produce_delta_model(lm, train_anomaly_df)
print(summary(linear_delta_model))
linear_delta_model_predicted_anomaly <- predict_delta_model(linear_delta_model, anomaly_df, validation_line)
plot_predictions(anomaly_df, linear_delta_model_predicted_anomaly, validation_line, "linear_delta_model_predictions")

linear_model <- produce_model(lm, train_anomaly_df)
print(summary(linear_model))
linear_model_predicted_anomaly <- predict_model(linear_model, anomaly_df)
plot_predictions(anomaly_df, linear_model_predicted_anomaly, validation_line, "linear_model_predictions")
