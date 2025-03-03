library(tidyverse)
library(scales)
library(olsrr)

mse <- function(model) {
    mean(summary(model)$residuals^2)
}

mae <- function(model) {
    mean(abs(summary(model)$residuals))
}

mape <- function(model) {
    mean(abs(summary(model)$residuals) / anomaly_df$anomaly)
}

error_metrics <- function(model) {
    list(
        mse = mse(model),
        mae = mae(model),
        mape = mape(model)
    )
}
anomaly_df <- read.csv("Data/anomaly_year.csv")

unscale <- function(x, original_min, original_max) {
    return(x * (original_max - original_min) + original_min)
}
anomaly_df <- anomaly_df %>% filter(year_bin >= -650000 & year_bin < 1700) # Include years from quaternary glaciation intensification to industrial revolution

min_anomaly <- min(anomaly_df$anomaly)
max_anomaly <- max(anomaly_df$anomaly)

anomaly_df <- anomaly_df %>%
    select(-c(insolation)) %>% # Remove any variables that worsen validation performance
    mutate(across(-c(year_bin), rescale)) %>% # Scale all variables from 0 to 1
    mutate(
        across(-c(year_bin, anomaly), list(
            squared = ~ rescale(.^2),
            times_obliquity = ~ rescale(. * obliquity)
        ))
    ) %>%
    na.omit()

# Fit a linear regression model
validation_line <- -100000
model <- lm(anomaly ~ . - year_bin, data = anomaly_df %>% filter(year_bin < validation_line))

optimization_type <- "stepwise"
print(paste("Using", optimization_type, "optimization"))
# Optimizing chosen parameters tends to worsen training performance but improve validation performance
if (optimization_type == "stepwise") { # 1.681 validation MAE
    stepwise <- ols_step_both_p(model, pent = 0.05, prem = 0.05, progress = TRUE, details = FALSE)
    model <- stepwise$model
}
print(summary(model))

anomaly_df$predicted_anomaly <- predict(model, newdata = anomaly_df)

anomaly_df$anomaly <- unscale(anomaly_df$anomaly, min_anomaly, max_anomaly)
anomaly_df$predicted_anomaly <- unscale(anomaly_df$predicted_anomaly, min_anomaly, max_anomaly)

validation_df <- anomaly_df %>% filter(year_bin >= validation_line)
print(paste("Unscaled MAE on Validation Data:", round(mean(abs(validation_df$anomaly - validation_df$predicted_anomaly)), 3), "°C"))
print(paste("Unscaled MSE on Validation Data:", round(mean((validation_df$anomaly - validation_df$predicted_anomaly)^2), 3), "°C^2"))

# Plot the anomaly and predicted anomaly on the same graph
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
# x11()
# print(plot)
ggsave("Outputs/regression_anomaly_forecast.png", plot, width = 10, height = 6)

# Try adding features based on squares of features and products of features
