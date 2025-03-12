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
