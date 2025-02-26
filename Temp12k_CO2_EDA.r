library(tidyverse)

file_path <- "Data/ice_core_800k_co2_extracted.csv"

# Data extracted from antarctica2015co2composite-noaa.txt, from https://www.ncei.noaa.gov/access/paleo-search/study/17975

anomaly_year_df <- read.csv("Data/anomaly_year.csv")
year_bins <- anomaly_year_df$year_bin

co2_df <- read.csv(file_path)
co2_df <- co2_df %>%
    mutate(year = 1950 - age_gas_calBP) %>%
    rowwise() %>%
    mutate(year_bin = year_bins[which.min(abs(year_bins - year))]) %>%
    group_by(year_bin) %>%
    summarize(co2_ppm = mean(co2_ppm, na.rm = TRUE)) %>%
    ungroup() %>%
    select(year_bin, co2_ppm)

anomaly_co2_df <- inner_join(anomaly_year_df, co2_df, by = "year_bin")

plot <- ggplot(anomaly_co2_df, aes(x = year_bin, y = co2_ppm)) +
    geom_line(linewidth = 1.2) +
    coord_cartesian(ylim = c(0, 500)) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df, aes(x = year_bin, y = anomaly)) +
    geom_line(linewidth = 1.2) +
    coord_cartesian(ylim = c(-30, 30)) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df, aes(x = co2_ppm, y = anomaly, color = year_bin >= 1850)) +
    geom_point(size = 2.2) +
    geom_smooth(method = "lm", se = FALSE) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df %>% filter(year_bin >= -12000), aes(x = year_bin, y = anomaly)) +
    geom_line(linewidth = 1.2) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df %>% filter(year_bin >= -12000), aes(x = year_bin, y = co2_ppm)) +
    geom_line(linewidth = 1.2) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df %>% filter(year_bin >= 1850), aes(x = year_bin, y = anomaly)) +
    geom_line(linewidth = 1.2) +
    theme_minimal()
x11()
print(plot)

plot <- ggplot(anomaly_co2_df %>% filter(year_bin >= 1850), aes(x = year_bin, y = co2_ppm)) +
    geom_line(linewidth = 1.2) +
    theme_minimal()
x11()
print(plot)
