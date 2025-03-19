library(tidyverse)
library(scales)

# Check if the file Data/anomaly_year.csv exists
file_path <- "Data/anomaly_year.csv"

if (!file.exists(file_path)) {
    stop("The file of aggregated temperature data does not exist. Run Temp12k_analytics.r to generate the file")
}

df <- read.csv(file_path)
df <- df %>% filter(year_bin >= -650000 & year_bin < 1700) # Include years from quaternary glaciation intensification to industrial revolution

# Aggregate data to have a constant frequency of 2000 years
df_aggregated <- df %>%
    mutate(year_bin = floor(year_bin / 2000) * 2000) %>%
    group_by(year_bin) %>%
    summarize(across(everything(), ~ mean(.x, na.rm = TRUE))) %>%
    ungroup() %>%
    mutate(across(-c(year_bin, anomaly), ~ rescale(.x)))

# Print the first few rows of the aggregated data
print(tail(df_aggregated))

local_maxima <- function(x) {
    which(diff(sign(diff(x))) == -2) + 1
}

wave_fitting_enrichment <- function(original_df) {
    eccentricity_maxima <- original_df %>% # Get indices where eccentricity peaks
        pull(eccentricity) %>%
        local_maxima()

    eccentricity_maxima_years <- original_df %>% # Get years where eccentricity peaks
        slice(eccentricity_maxima) %>%
        pull(year_bin)
    average_period <- mean(diff(eccentricity_maxima_years)) # Get average distance between peaks

    year_offset <- eccentricity_maxima_years[1] - min(df$year_bin)

    enriched_df <- original_df %>%
        mutate(eccentricity_wave = sin(2 * pi * (year_bin - year_offset) / average_period) / 2 + 0.5)

    df_long <- enriched_df %>%
        select(eccentricity, eccentricity_wave, anomaly, year_bin) %>%
        mutate(anomaly = rescale(anomaly)) %>%
        pivot_longer(cols = -year_bin, names_to = "variable", values_to = "value")

    plot <- ggplot(df_long, aes(x = year_bin, y = value, color = variable)) +
        geom_line() +
        geom_vline(xintercept = eccentricity_maxima_years, linetype = "dashed", color = "red") +
        labs(title = "Fields vs Year Bin", x = "Year Bin", y = "Value") +
        theme_classic()
    ggsave("Outputs/eccentricity_wave_fitting.png", plot, width = 10, height = 6)

    return(enriched_df)
}

df_aggregated <- df_aggregated %>%
    wave_fitting_enrichment()

write.csv(df_aggregated, "Data/anomaly_training_ready.csv", row.names = FALSE)

# Calculate the delta for each variable
df_deltas <- df_aggregated %>%
    arrange(year_bin) %>%
    mutate(across(-c(year_bin), ~ .x - lag(.x), .names = "delta_{col}")) %>%
    mutate(across(starts_with("delta_") & !starts_with("delta_anomaly"), ~ rescale(.x))) %>%
    filter(across(everything(), ~ !is.na(.)))

# Print the first few rows of the data with deltas
print(tail(df_deltas))

write.csv(df_deltas, "Data/anomaly_training_ready_with_deltas.csv", row.names = FALSE)
