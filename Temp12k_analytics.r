library(ggplot2)
library(dplyr)

conn_str <- "DRIVER={ODBC Driver 17 for SQL Server};SERVER=(LocalDB)\\MSSQLLocalDB;DATABASE=Temp12k;Trusted_Connection=yes;"

conn <- DBI::dbConnect(odbc::odbc(), .connection_string = conn_str)
print(conn)
cat("Connected to database\n")

tables <- DBI::dbGetQuery(conn, "
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
        AND TABLE_SCHEMA = 'dbo'
        AND TABLE_NAME NOT LIKE 'sys%'
        AND TABLE_NAME NOT LIKE 'MS%'
")
cat("User tables in database:\n")
cat(tables$TABLE_NAME)

for (table_name in tables$TABLE_NAME) {
    cat("\nColumns in", table_name, ":\n")
    columns <- DBI::dbGetQuery(conn, sprintf("
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '%s'
        ORDER BY ORDINAL_POSITION", table_name))
    print(columns)
}

DBI::dbExecute(conn, "
    ALTER DATABASE Temp12k SET RECOVERY SIMPLE;
    DBCC SHRINKFILE(Temp12k_log);
") # Reduce log file size

cat("\nRetrieving records from database...\n")
anomaly_df <- as.data.frame(DBI::dbGetQuery(conn, "
    SELECT t.anomaly, dt.year_bin, da.co2_ppm
    FROM fact_temperature t
    INNER JOIN dim_time dt ON t.time_id = dt.time_id
    INNER JOIN dim_atmosphere da ON t.time_id = da.time_id
")) # Get corresponding anomaly per year_bin`

colnames(anomaly_df) <- c("anomaly", "year_bin", "co2_ppm")

cat("Shape of anomaly_df:", nrow(anomaly_df), "rows x", ncol(anomaly_df), "columns\n")
cat("\nAggregating...\n")
anomaly_df <- anomaly_df %>%
    group_by(year_bin, co2_ppm) %>% # Aggregate anomalies per year_bin
    summarise(anomaly = mean(anomaly, na.rm = TRUE)) %>%
    ungroup()

cat("Shape of anomaly_df:", nrow(anomaly_df), "rows x", ncol(anomaly_df), "columns\n")

cat("\nCreating plots...\n")
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
ggsave("Outputs/long_term_temperature_anomaly.png", width = 10, height = 6)

ggplot(anomaly_df %>% filter(year_bin >= -800000), aes(x = year_bin, y = co2_ppm, color = anomaly)) +
    geom_line() +
    geom_vline(xintercept = -700000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -440000, y = 350, label = "Quaternary Glaciation Intensifies", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    geom_vline(xintercept = -110000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -110000, y = 350, label = "Last Ice Age Starts", hjust = 1.1, vjust = -0.5, color = "#0072F5") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 8, limits = c(-10, 10)) +
    labs(x = "Year", y = "CO2 Concentration (parts per million)") +
    theme_classic() +
    scale_y_continuous(limits = c(150, 450)) +
    scale_x_continuous(labels = scales::comma)
ggsave("Outputs/long_term_co2_ppm.png", width = 10, height = 6)

ggplot(anomaly_df %>% filter(year_bin >= -12000), aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = -10000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -10000, y = 1.5, label = "Agricultural\nRevolution", hjust = 1.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = -8000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -8000, y = 1.5, label = "Ice Age Ends", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = 1760, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 1760, y = 1.5, label = "Industrial\nRevolution", hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 3, limits = c(-3, 3)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-3, 3))
ggsave("Outputs/since_ice_age_temperature_anomaly.png", width = 10, height = 6)

latest_anomaly <- tail(anomaly_df$anomaly, 1)
modern_plot_config <- ggplot(anomaly_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2025: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-2, 2)) +
    scale_x_continuous(limits = c(1850, 2025), labels = scales::comma)
ggsave("Outputs/modern_temperature_anomaly.png", width = 10, height = 6)


ggplot(anomaly_df, aes(x = co2_ppm, y = anomaly, color = year_bin >= 1850)) +
    geom_point(size = 2.2) +
    geom_smooth(method = "lm", se = FALSE) +
    theme_classic()
ggsave("Outputs/anomaly_vs_co2_ppm.png", width = 10, height = 6)

write.csv(anomaly_df, "Data/anomaly_year.csv", row.names = FALSE)

DBI::dbDisconnect(conn)
cat("\nDisconnected from database\n")
