library(tidyverse)
library(reshape2)
library(scales)
library(patchwork)

anomaly_df <- read.csv("Outputs/long_term_global_anomaly_view_enriched.csv") %>% filter(year_bin <= 2025)
print(head(anomaly_df, 1))
get_year_bin_plot <- function() {
    plot <- ggplot(anomaly_df, aes(x = year_bin, x = anomaly)) +
        geom_point() +
        theme_classic()
}

get_co2_ppm_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = co2_ppm)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_co2_radiative_forcing_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = co2_radiative_forcing)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_eccentricity_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = eccentricity)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_perihelion_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = perihelion)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_obliquity_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = obliquity)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_insolation_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = insolation)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_global_insolation_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = global_insolation^0.5)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
    # Found that global insolation has a square root relationship with anomaly
    # Square root of global insolation has linear relationship with anomaly
}

get_be_ppm_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = be_ppm)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_VADM_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = VADM)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_solar_modulation_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = solar_modulation)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

get_custom_plot <- function() {
    plot <- ggplot(anomaly_df, aes(y = anomaly, x = eccentricity * delta_solar_modulation)) +
        geom_point() +
        geom_smooth(method = "lm", se = FALSE) +
        theme_classic()
}

year_bin <- 1
co2_ppm <- 2
co2_radiative_forcing <- 3
eccentricity <- 4
perihelion <- 5
obliquity <- 6
insolation <- 7
global_insolation <- 8
be_ppm <- 9
VADM <- 10
solar_modulation <- 11
custom <- 12

plot_type <- custom
results_window <- FALSE
if (plot_type == year_bin) {
    plot <- get_year_bin_plot()
} else if (plot_type == co2_ppm) {
    plot <- get_co2_ppm_plot()
} else if (plot_type == co2_radiative_forcing) {
    plot <- get_co2_radiative_forcing_plot()
} else if (plot_type == eccentricity) {
    plot <- get_eccentricity_plot()
} else if (plot_type == perihelion) {
    plot <- get_perihelion_plot()
} else if (plot_type == obliquity) {
    plot <- get_obliquity_plot()
} else if (plot_type == insolation) {
    plot <- get_insolation_plot()
} else if (plot_type == global_insolation) {
    plot <- get_global_insolation_plot()
} else if (plot_type == be_ppm) {
    plot <- get_be_ppm_plot()
} else if (plot_type == VADM) {
    plot <- get_VADM_plot()
} else if (plot_type == solar_modulation) {
    plot <- get_solar_modulation_plot()
} else if (plot_type == custom) {
    plot <- get_custom_plot()
}
if (results_window) {
    x11()
    print(plot)
}

ggsave("Outputs/EDA_plot.png", width = 10, height = 6)
