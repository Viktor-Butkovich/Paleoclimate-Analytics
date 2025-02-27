library(tidyverse)
library(reshape2)

file_path <- "Data/Milankovitch-Insolation-Extracted.csv"

milankovitch_df <- read_csv(file_path)

anomaly_df <- read_csv("Data/anomaly_year.csv")

milankovitch_df <- milankovitch_df %>%
    mutate(year = 1950 + BP, insolation = annual_insolation_65N - mean(annual_insolation_65N), july_insolation = july_insolation - mean(july_insolation)) %>% # Note that this dataset uses negative BP, so we have to add instead of subtract
    mutate(delta_july_insolation = july_insolation - lag(july_insolation)) %>%
    mutate(annual_insolation_global_cos_weighted = annual_insolation_global_cos_weighted - mean(annual_insolation_global_cos_weighted)) %>%
    mutate(delta_global_insolation = annual_insolation_global_cos_weighted - lag(annual_insolation_global_cos_weighted)) %>%
    mutate(delta_global_insolation_per_year = delta_global_insolation / (year - lag(year))) %>%
    select(year, insolation, july_insolation, delta_july_insolation, annual_insolation_global_cos_weighted, delta_global_insolation, delta_global_insolation_per_year)


# Bin the years in milankovitch_df according to the year_bins in anomaly_df
milankovitch_df <- milankovitch_df %>%
    mutate(year_bin = cut(year, breaks = anomaly_df$year_bin, labels = FALSE, include.lowest = TRUE))

# Aggregate the data by the year bins
milankovitch_binned_df <- milankovitch_df %>%
    group_by(year_bin) %>%
    summarize(
        insolation = mean(insolation, na.rm = TRUE),
        july_insolation = mean(july_insolation, na.rm = TRUE),
        delta_july_insolation = mean(delta_july_insolation, na.rm = TRUE),
        annual_insolation_global_cos_weighted = mean(annual_insolation_global_cos_weighted, na.rm = TRUE),
        delta_global_insolation = mean(delta_global_insolation, na.rm = TRUE),
        delta_global_insolation_per_year = mean(delta_global_insolation_per_year, na.rm = TRUE)
    )

# Merge the binned data with the anomaly_df to get the actual year ranges
milankovitch_binned_df <- milankovitch_binned_df %>%
    mutate(year_bin = anomaly_df$year_bin[year_bin])

# Join anomaly_df and milankovitch_binned_df on year_bin
combined_df <- left_join(anomaly_df, milankovitch_binned_df, by = "year_bin")

correlation <- cor(combined_df$annual_insolation_global_cos_weighted, combined_df$anomaly, use = "complete.obs")
print("Correlation between global insolation and temperature anomaly:")
print(correlation)

correlation <- cor(combined_df$annual_insolation_global_cos_weighted, combined_df$co2_ppm, use = "complete.obs")
print("Correlation between global insolation and co2_ppm:")
print(correlation)

# Not much correlation observed involving insolation, but they are clearly related based on the plots - investigate the relationships of the delta values for each variable
#   The absolute values are not closely related, but they usually increase and decrease together

correlation <- cor(combined_df$co2_ppm, combined_df$anomaly, use = "complete.obs")
print("Correlation between co2_ppm and temperature anomaly:")
print(correlation)

melted_df <- combined_df %>%
    filter(year_bin <= 1850) %>%
    rename(global_insolation = annual_insolation_global_cos_weighted) %>%
    mutate(group = (row_number() - 1) %/% 5) %>%
    group_by(group) %>% # Summarize every 5 rows as 1 row
    summarize(
        year_bin = first(year_bin),
        anomaly = mean(anomaly, na.rm = TRUE),
        global_insolation = mean(global_insolation, na.rm = TRUE),
        co2_ppm = mean(co2_ppm, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    mutate(anomaly = (anomaly - min(anomaly, na.rm = TRUE)) / (max(anomaly, na.rm = TRUE) - min(anomaly, na.rm = TRUE)) * 2 - 1) %>%
    mutate(global_insolation = (global_insolation - min(global_insolation, na.rm = TRUE)) / (max(global_insolation, na.rm = TRUE) - min(global_insolation, na.rm = TRUE)) * 2 - 1) %>%
    mutate(co2_ppm = (co2_ppm - min(co2_ppm, na.rm = TRUE)) / (max(co2_ppm, na.rm = TRUE) - min(co2_ppm, na.rm = TRUE)) * 2 - 1) %>%
    select(year_bin, global_insolation, anomaly, co2_ppm) %>%
    melt("year_bin")

plot <- ggplot(melted_df %>% filter(year_bin >= -650000), aes(x = year_bin, y = value, color = variable, linetype = variable)) +
    geom_line(linewidth = 1.3) +
    labs(x = "Year", y = "Standardized Values") +
    annotate("rect", xmin = -75000, xmax = -11000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -43000, y = 1.3, label = "Wisconsin\nGlaciation", color = "blue", vjust = 1.5) +
    annotate("rect", xmin = -191000, xmax = -130000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -160000, y = 1.3, label = "Illinoian\nGlaciation", color = "blue", vjust = 1.5) +
    annotate("rect", xmin = -300000, xmax = -250000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -385000, xmax = -345000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -475000, xmax = -430000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -560000, xmax = -530000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -650000, xmax = -620000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -400000, y = 1.3, label = "Pre-Illinoian Glaciations", color = "blue", vjust = 1.5) +
    annotate("text", x = 15000, y = 1.35, label = "Holocene\nInterglacial\n(Modern)", color = "blue", vjust = 1.5) +
    scale_linetype_manual(values = c(global_insolation = "solid", anomaly = "longdash", co2_ppm = "longdash")) +
    theme_classic() +
    scale_x_continuous(labels = scales::comma) +
    coord_cartesian(xlim = c(-650000, 20000))
ggsave("Outputs/insolation_co2_ppm_anomaly_trends.png", width = 15, height = 9)
