library(tidyverse)

# Check if the file Data/anomaly_year.csv exists
file_path <- "Data/anomaly_year.csv"

if (!file.exists(file_path)) {
    stop("The file of aggregated temperature data does not exist. Run Temp12k_analytics.r to generate the file")
}

anomaly_year_df <- read.csv(file_path)
recent_anomaly_df <- anomaly_year_df %>%
    filter(year_bin >= 1850) %>%
    mutate(timestep = row_number() - 1)
latest_anomaly <- tail(recent_anomaly_df$anomaly, 1)

plot <- ggplot(recent_anomaly_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2025: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-2, 2)) +
    scale_x_continuous(labels = scales::comma)
x11()
print(plot)
