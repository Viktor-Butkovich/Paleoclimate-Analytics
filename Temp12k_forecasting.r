library(tidyverse)

# Check if the file Data/anomaly_year.csv exists
file_path <- "Data/anomaly_year.csv"

if (!file.exists(file_path)) {
    stop("The file of aggregated temperature data does not exist. Run Temp12k_analytics.r to generate the file")
}

exp_offset <- 2 # Offset to prevent taking the logarithm of zero
anomaly_year_df <- read.csv(file_path)
recent_anomaly_df <- anomaly_year_df %>%
    filter(year_bin >= 1850) %>%
    mutate(timestep = row_number() - 1, timestep_2 = timestep^2, timestep_3 = timestep^3, timestep_4 = timestep^4, ln_anomaly = log(anomaly + exp_offset))
latest_anomaly <- tail(recent_anomaly_df$anomaly, 1)

anomaly_forecast_df <- recent_anomaly_df %>%
    bind_rows(data.frame(year_bin = 2025:2050, timestep = 176:201, anomaly = NA)) %>%
    mutate(timestep = row_number() - 1, timestep_2 = timestep^2, timestep_3 = timestep^3, timestep_4 = timestep^4, ln_anomaly = log(anomaly + exp_offset))

simple_linear_regression <- lm(data = recent_anomaly_df, formula = anomaly ~ timestep) # Anomaly as a linear function of years since 1850
anomaly_forecast_df <- anomaly_forecast_df %>%
    mutate(slr_anomaly = predict(simple_linear_regression, newdata = anomaly_forecast_df))

quadratic_regression <- lm(data = recent_anomaly_df, formula = anomaly ~ timestep_2 + timestep_3) # Anomaly as a quadratic function of years since 1850
anomaly_forecast_df <- anomaly_forecast_df %>%
    mutate(qr_anomaly = predict(quadratic_regression, newdata = anomaly_forecast_df))
# Years since 1850 ^2 and ^3 are both statistically significant, and the R^2 = 0.8968 means this model explains 89.68% of the variance in temperature anomalies

exponential_regression <- lm(data = recent_anomaly_df, formula = ln_anomaly ~ timestep) # Logarithm of anomaly as a linear function of years since 1850
anomaly_forecast_df <- anomaly_forecast_df %>%
    mutate(exp_anomaly = exp(predict(exponential_regression, newdata = anomaly_forecast_df)) - exp_offset)

plot <- ggplot(recent_anomaly_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2025: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-2, 2)) +
    x11()
print(plot)

plot <- ggplot() +
    geom_line(data = recent_anomaly_df, aes(x = year_bin, y = anomaly, color = "Observed Anomaly"), linewidth = 1.2) +
    geom_line(data = anomaly_forecast_df, aes(x = year_bin, y = slr_anomaly, color = "Forecasted Anomaly (SLR)"), linewidth = 1.2, linetype = "dashed") +
    geom_line(data = anomaly_forecast_df, aes(x = year_bin, y = qr_anomaly, color = "Forecasted Anomaly (QR)"), linewidth = 1.2, linetype = "dotted") +
    # geom_line(data = anomaly_forecast_df, aes(x = year_bin, y = exp_anomaly, color = "Forecasted Anomaly (EXP)"), linewidth = 1.2, linetype = "dotdash") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#df00a7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2025: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_manual(values = c("Observed Anomaly" = "blue", "Forecasted Anomaly (SLR)" = "red", "Forecasted Anomaly (QR)" = "orange", "Forecasted Anomaly (EXP)" = "purple")) +
    labs(x = "Year", y = "Temperature Anomaly (°C)", color = "Legend") +
    theme_classic() +
    scale_y_continuous(limits = c(-1, 3))
x11()
print(plot)

ggsave("Outputs/modern_temperature_anomaly_forecast.png", width = 15, height = 9)
