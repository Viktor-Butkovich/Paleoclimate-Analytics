suppressPackageStartupMessages({
    library(tidyverse)
    library(reshape2)
    library(scales)
    library(patchwork)
    library(jsonlite)
    library(here)
})
options(warn = -1) # Suppress warnings
config <- fromJSON(here("prediction_config.json"))
scoreboard <- fromJSON(here("Outputs", "scoreboard.json"))
evolution_log <- fromJSON(here("Outputs", "genetic_torch_model_evolution_log.json"))

anomaly_df <- read.csv(here("Outputs", "long_term_global_anomaly_view.csv")) %>% filter(year_bin <= 2025)
# Preprocessed, normalized, missing values interpolated, etc. with constant frequency

anomaly_df_raw <- read.csv(here("Outputs", "raw_global_anomaly_view.csv")) %>% filter(year_bin <= 2025 & year_bin >= -700000)
# Raw data, good for actual attribute values and high-frequency time periods (recent)

ggplot(anomaly_df %>% filter(year_bin >= -800000), aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -600000, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#DF00A7") +
    geom_vline(xintercept = -700000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -440000, y = 5, label = "Quaternary Glaciation Intensifies", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = -110000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -110000, y = 5, label = "Last Ice Age Starts", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 8, limits = c(-10, 10)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-10, 10)) +
    scale_x_continuous(labels = scales::comma)
ggsave(here("Outputs", "long_term_temperature_anomaly.png"), width = 10, height = 6)

ggplot(anomaly_df_raw %>% filter(year_bin >= -800000), aes(x = year_bin, y = co2_ppm, color = anomaly)) +
    geom_line() +
    geom_vline(xintercept = -700000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -440000, y = 350, label = "Quaternary Glaciation Intensifies", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = -110000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -110000, y = 350, label = "Last Ice Age Starts", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 8, limits = c(-10, 10)) +
    labs(x = "Year", y = "CO2 Concentration (parts per million)") +
    theme_classic() +
    scale_y_continuous(limits = c(175, 450)) +
    scale_x_continuous(labels = scales::comma)
ggsave(here("Outputs", "long_term_co2_ppm.png"), width = 10, height = 6)

ggplot(anomaly_df_raw %>% filter(year_bin >= -12000), aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#DF00A7") +
    geom_vline(xintercept = -10000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -10000, y = 1.5, label = "Agricultural\nRevolution", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = -8000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -8000, y = 1.5, label = "Ice Age Ends", hjust = -0.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = 1760, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 1760, y = 1.5, label = "Industrial\nRevolution", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 3, limits = c(-3, 3)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-3, 3))
ggsave(here("Outputs", "since_ice_age_temperature_anomaly.png"), width = 10, height = 6)

ggplot(anomaly_df_raw %>% filter(year_bin >= -12000), aes(x = year_bin, y = co2_ppm, color = anomaly)) +
    geom_line() +
    geom_vline(xintercept = -10000, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -10000, y = 350, label = "Agricultural\nRevolution", hjust = 1.1, vjust = -0.5, color = "#DF00A7") +
    geom_vline(xintercept = -8000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -8000, y = 350, label = "Ice Age Ends", hjust = -0.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = 1760, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 1760, y = 350, label = "Industrial\nRevolution", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 3, limits = c(-3, 3)) +
    labs(x = "Year", y = "CO2 Concentration (parts per million)") +
    theme_classic() +
    scale_y_continuous(limits = c(175, 450)) +
    scale_x_continuous(labels = scales::comma)
ggsave(here("Outputs", "since_ice_age_co2_ppm.png"), width = 10, height = 6)

latest_anomaly <- tail(anomaly_df_raw$anomaly, 1)
ggplot(anomaly_df_raw, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -3500, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#DF00A7") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2024: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-2, 2)) +
    scale_x_continuous(limits = c(1850, 2025), labels = scales::comma)
ggsave(here("Outputs", "modern_temperature_anomaly.png"), width = 10, height = 6)

ggplot(anomaly_df_raw, aes(x = year_bin, y = co2_ppm, color = anomaly)) +
    geom_line() +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = 415, label = "2024: CO2 Concentration of 424.61 ppm", hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "CO2 Concentration (parts per million)") +
    theme_classic() +
    scale_y_continuous(limits = c(175, 450)) +
    scale_x_continuous(limits = c(1850, 2025), labels = scales::comma)
ggsave(here("Outputs", "modern_co2_ppm.png"), width = 10, height = 6)

ggplot(anomaly_df_raw %>% mutate(color_bin = case_when(
    year_bin < 1850 ~ 0,
    year_bin >= 1850 & year_bin < 1980 ~ 1,
    year_bin >= 1980 ~ 2
)), aes(x = co2_ppm, y = anomaly, color = factor(color_bin))) +
    geom_point(size = 2.2) +
    geom_smooth(formula = y ~ x, method = "lm", se = FALSE) +
    theme_classic() +
    scale_color_manual(values = c("0" = "blue", "1" = "green", "2" = "red"), labels = c("Before 1850", "1850-1979", "1980-Present"))
ggsave(here("Outputs", "anomaly_vs_co2_ppm.png"), width = 10, height = 6)

group_size <- 1
glaciation_orbit_df <- anomaly_df_raw %>%
    filter(year_bin <= 1850) %>% # Exclude post-industrial data
    select(year_bin, anomaly, co2_ppm, eccentricity, obliquity, perihelion, insolation, global_insolation) %>%
    mutate(across(-year_bin, rescale)) %>% # Scale all variables from 0 to 1
    select(-perihelion, -insolation, -obliquity) %>%
    mutate(group_number = row_number() %/% group_size) %>% # Aggregate every set of group_size bins into 1
    group_by(group_number) %>%
    summarise(across(everything(), mean)) %>%
    ungroup() %>%
    select(-group_number)
glaciation_orbit_df_melted <- melt(glaciation_orbit_df, id.vars = "year_bin", variable.name = "variable", value.name = "value")

glaciation_orbit_plot <- ggplot(glaciation_orbit_df_melted %>% filter(year_bin >= -650000), aes(x = year_bin, y = value, color = variable, linetype = variable)) +
    geom_line(linewidth = 1.3) +
    labs(x = "Year", y = "Standardized Scale") +
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
    scale_linetype_manual(values = c(eccentricity = "solid", global_insolation = "solid", anomaly = "longdash", co2_ppm = "longdash")) +
    scale_color_manual(values = c(eccentricity = "#7700ff", global_insolation = "#e5ff00", anomaly = "#ff0000", co2_ppm = "#161616")) +
    theme_classic() +
    scale_x_continuous(labels = scales::comma) +
    ggtitle("Glacial Cycles Due to Orbital Variation (Pre-Industrial)") +
    coord_cartesian(xlim = c(-650000, 20000))

anomaly_orbit_plot <- ggplot(glaciation_orbit_df, aes(x = eccentricity, y = anomaly, color = year_bin >= -12000)) +
    geom_point() +
    labs(x = "Eccentricity", y = "Anomaly") +
    ggtitle("Climate Anomaly vs Orbital Eccentricity") +
    theme_classic() +
    theme(legend.position = "none")

anomaly_insolation_plot <- ggplot(glaciation_orbit_df, aes(x = global_insolation, y = anomaly, color = year_bin >= -12000)) +
    geom_point() +
    labs(x = "Global Insolation", y = "Anomaly", color = "Year >= -12000 (Post-Ice Age)") +
    ggtitle("Climate Anomaly vs Global Insolation") +
    theme_classic()
combined_plot <- glaciation_orbit_plot / (anomaly_orbit_plot | anomaly_insolation_plot)
combined_plot
ggsave(here("Outputs", "orbital_parameters_glacial_cycles_trends.png"), width = 15, height = 9)

solar_modulation_plot <- ggplot(anomaly_df, aes(x = year_bin)) +
    geom_line(aes(y = rescale(solar_modulation), color = "Solar Modulation (Φ)"), linewidth = 1.2) +
    geom_line(aes(y = rescale(anomaly), color = "Temperature Anomaly (°C)"), linewidth = 1.2) +
    labs(x = "Year", y = "Value", color = "Legend") +
    annotate("rect", xmin = -75000, xmax = -11000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -43000, y = max(anomaly_df$solar_modulation, na.rm = TRUE) * 0.9, label = "Wisconsin\nGlaciation", color = "blue", vjust = 1.5) +
    annotate("rect", xmin = -191000, xmax = -130000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -160000, y = max(anomaly_df$solar_modulation, na.rm = TRUE) * 0.9, label = "Illinoian\nGlaciation", color = "blue", vjust = 1.5) +
    annotate("rect", xmin = -300000, xmax = -250000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -385000, xmax = -345000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -475000, xmax = -430000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -560000, xmax = -530000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("rect", xmin = -650000, xmax = -620000, ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#58008b") +
    annotate("text", x = -400000, y = max(anomaly_df$solar_modulation, na.rm = TRUE) * 0.9, label = "Pre-Illinoian Glaciations", color = "blue", vjust = 1.5) +
    theme_classic() +
    scale_x_continuous(labels = scales::comma) +
    scale_color_manual(values = c("Solar Modulation (Φ)" = "orange", "Temperature Anomaly (°C)" = "red")) +
    ggtitle("Solar Modulation and Temperature Anomaly with Ice Age Overlays")

solar_modulation_plot
ggsave(here("Outputs", "long_term_solar_modulation_plot.png"), solar_modulation_plot, width = 12, height = 6)

plot_predictions <- function(prediction_type) {
    file_path_base <- here("Outputs", paste(prediction_type, "_predictions", sep = ""))
    if (prediction_type == "default") {
        data <- read.csv(here("Outputs", "long_term_global_anomaly_view.csv")) %>% filter(year_bin <= config$forecast_end)
    } else {
        data <- read.csv(here(paste(file_path_base, ".csv", sep = ""))) %>% filter(year_bin <= config$forecast_end)
    }

    present_line <- config$present # Include these in a shared configuration file rather than hardcoding
    train1_bounds <- c(min(data$year_bin), config$test_start)
    test_bounds <- c(config$test_start, config$test_end)
    train2_bounds <- c(config$test_end, present_line)
    forecast_bounds <- c(present_line, config$forecast_end)
    data <- data %>% mutate(anomaly = ifelse(year_bin > present_line, NA, anomaly)) # Set future anomalies to null for plotting
    extra_layers <- list()
    if (prediction_type == "default") {
        extra_layers <- c(
            extra_layers,
            list(
                labs(
                    x = "Year",
                    y = "Anomaly (°C)",
                    color = "Legend",
                    title = "Climate Anomaly by Year",
                )
            )
        )
    } else {
        extra_layers <- c(
            extra_layers,
            list(
                labs(
                    x = "Year",
                    y = "Anomaly (°C)",
                    color = "Legend",
                    title = paste(str_to_title(str_replace_all(prediction_type, "_", " ")), " Predictions - Actual and Predicted Climate Anomaly by Year", sep = "")
                ),
                geom_line(aes(x = year_bin, y = data$pred_anomaly, color = "Predicted Anomaly"))
            )
        )
    }

    if (grepl("arima", file_path_base)) {
        data <- data %>% mutate(pred_anomaly = ifelse(year_bin > present_line, pred_anomaly, NA)) # Set past anomalies to null for plotting - trivial predictions
        extra_layers <- c(
            extra_layers,
            list(
                annotate("text", x = mean(c(train1_bounds[1], train2_bounds[2])), y = -10, label = "Train", hjust = 0.5, color = "black")
            ) # Pure forecasting methods don't have a train/test split
        )
    } else {
        extra_layers <- c(
            extra_layers,
            list(
                annotate("rect", xmin = test_bounds[1], xmax = test_bounds[2], ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "grey"),
                annotate("text", x = mean(test_bounds), y = -10, label = "Test", hjust = 0.5, color = "black"),
                annotate("text", x = mean(train1_bounds), y = -10, label = "Train", hjust = 0.5, color = "black"),
                annotate("text", x = mean(train2_bounds), y = -10, label = "Train", hjust = 0.5, color = "black")
            )
        )
        if (prediction_type != "default") {
            extra_layers <- c(
                extra_layers,
                list(
                    labs(caption = paste("Validation MSE: ", scoreboard[[prediction_type]], sep = ""))
                )
            )
        }
    }

    ggplot(data, aes(x = year_bin)) +
        geom_line(aes(y = anomaly, color = "Actual Anomaly")) +
        extra_layers +
        theme_classic() +
        scale_y_continuous(limits = c(-10, 4), breaks = seq(-10, 4, by = 2)) +
        scale_x_continuous(labels = label_number(scale_cut = cut_short_scale()), breaks = seq(ceiling(min(data$year_bin) / 100000) * 100000, max(data$year_bin), by = 100000)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
        annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "#df00a7") +
        annotate("rect", xmin = present_line, xmax = max(data$year_bin), ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#889AFF") +
        geom_vline(xintercept = present_line, linetype = "dotted", color = "blue") +
        annotate("text", x = present_line, y = 3, label = "Present", hjust = 1.1, color = "black") +
        annotate("text", x = mean(forecast_bounds), y = -10, label = "Forecast", hjust = 0.5, color = "black")
    ggsave(here("Outputs", paste(prediction_type, "_predictions.png", sep = "")), width = 10, height = 6)
}


for (prediction_type in c(
    "linear_model",
    "lagged_linear_model",
    "torch_model",
    "genetic_torch_model",
    "arima_model",
    "arimax_model",
    "default"
)) {
    plot_predictions(prediction_type)
}

generations_fitness_df <- evolution_log %>%
    select(generation, fitness) %>%
    filter(generation >= 0) %>%
    mutate(fitness = fitness)
ggplot(generations_fitness_df, aes(x = generation, y = fitness)) +
    geom_line(color = "blue", linewidth = 1.2) +
    labs(x = "Generation", y = "Fitness (MSE)", title = "Fitness vs Generation") +
    theme_classic()
ggsave(here("Outputs", "genetic_torch_model_evolution_progress.png"), width = 10, height = 6)

scoreboard_df <- as.data.frame(scoreboard) %>%
    pivot_longer(cols = everything(), names_to = "model", values_to = "mse") %>%
    mutate(model = str_replace_all(model, "_", " ")) %>%
    mutate(model = str_to_title(model)) %>%
    mutate(model = reorder(model, mse, decreasing = TRUE))

ggplot(scoreboard_df, aes(x = model, y = mse, fill = model)) +
    geom_bar(stat = "identity") +
    labs(x = "Model", y = "Validation MSE", title = "Validation MSE by Model") +
    theme_classic() +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_brewer(palette = "Set3")

ggsave(here("Outputs", "validation_mse_by_model.png"), width = 10, height = 6)

print("Plots saved successfully")
