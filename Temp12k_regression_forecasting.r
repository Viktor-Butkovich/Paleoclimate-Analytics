library(tidyverse)
library(scales)

anomaly_df <- read.csv("Data/anomaly_year.csv")

anomaly_df <- anomaly_df %>%
    filter(year_bin >= -650000 & year_bin < 1700) %>% # Include years from quaternary glaciation intensification to industrial revolution
    mutate(across(-c(year_bin, anomaly), rescale)) # Scale all variables from 0 to 1

# Fit a linear regression model
model <- lm(anomaly ~ . - year_bin, data = anomaly_df)

# Print the summary of the model
print(summary(model))

anomaly_df <- anomaly_df %>% # Remove non-statistically significant variables
    select(-c(perihelion, insolation)) # Scale all variables from 0 to 1

# Fit a linear regression model
trimmed_model <- lm(anomaly ~ . - year_bin, data = anomaly_df)

# Print the summary of the model
print(summary(trimmed_model))
# Use the trimmed model to make predictions
anomaly_df$predicted_anomaly <- predict(trimmed_model, newdata = anomaly_df)

# Plot the anomaly and predicted anomaly on the same graph
plot <- ggplot(anomaly_df, aes(x = year_bin)) +
    geom_line(aes(y = anomaly, color = "Actual Anomaly")) +
    geom_line(aes(y = predicted_anomaly, color = "Predicted Anomaly")) +
    labs(
        title = "Forecasted Temperature Anomalies (-650 kya to 1700)",
        x = "Year Bin",
        y = "Anomaly",
        color = "Legend"
    ) +
    theme_classic() +
    scale_y_continuous(limits = c(-10, 10)) +
    scale_x_continuous(labels = scales::comma)
x11()
print(plot)
ggsave("Outputs/regression_anomaly_forecast.png", plot, width = 10, height = 6)

# Try adding features based on squares of features and products of features
