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
    select(-c(co2_ppm, co2_radiative_forcing, eccentricity, obliquity, perihelion, global_insolation, insolation, delta_co2_ppm, delta_insolation, delta_global_insolation))

validation_line <- -100000
anomaly_train <- anomaly_df %>% filter(year_bin <= validation_line)
anomaly_validation <- anomaly_df %>% filter(year_bin > validation_line)

print("Training...")
ctrl <- trainControl(method = "cv", number = 10)
model_type <- "lm"
if (model_type == "lm") {
    model <- lm(delta_anomaly ~ . - year_bin - anomaly, data = anomaly_train)
} else if (model_type == "nn") {
    grid <- expand.grid(.size = seq(5, 20, 5), .decay = seq(0.1, 0.5, 0.1))
    model <- train(delta_anomaly ~ . - year_bin - anomaly, method = "nnet", trControl = ctrl, tuneGrid = grid, preProcess = c("center", "scale"), linout = TRUE, trace = FALSE, data = anomaly_train)
} else if (model_type == "regression_tree") {
    model <- train(delta_anomaly ~ . - year_bin - anomaly, method = "rpart", trControl = ctrl, tuneLength = 200, data = anomaly_train)
} else if (model_type == "random_forest") {
    grid <- expand.grid(.mtry = c(1:10))
    model <- train(delta_anomaly ~ . - year_bin - anomaly, method = "rf", trControl = ctrl, tuneGrid = grid, data = anomaly_train)
} else {
    stop("Invalid model type")
}
print(summary(model))

anomaly_train <- anomaly_train %>%
    mutate(predicted_anomaly = lag(anomaly) + predict(model, newdata = anomaly_train))

cumulative_delta_forecast <- function(last_observed_anomaly, model, future_exogenous) {
    predicted_delta_anomaly <- predict(model, newdata = future_exogenous)
    return(last_observed_anomaly + cumsum(predicted_delta_anomaly))
}

anomaly_validation$predicted_anomaly <- cumulative_delta_forecast(tail(anomaly_train$anomaly, 1), model, anomaly_validation)
# Each predicted anomaly within previous data is predicted from the previous anomaly
# Future anomalies are extrapolated from the last known observation and the cumulative sum of predicted deltas, based on deltas of global parameters

anomaly_df <- rbind(anomaly_train, anomaly_validation)


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
ggsave("Outputs/delta_anomaly_forecast.png", plot, width = 10, height = 6)
