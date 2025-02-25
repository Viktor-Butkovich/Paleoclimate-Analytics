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
anomaly_year_df <- as.data.frame(DBI::dbGetQuery(conn, "
    SELECT t.anomaly, dt.year_bin
    FROM fact_temperature t
    INNER JOIN dim_time dt ON t.temperature_id = dt.temperature_id
")) # Get corresponding anomaly per year_bin
colnames(anomaly_year_df) <- c("anomaly", "year_bin")

cat("Shape of anomaly_year_df:", nrow(anomaly_year_df), "rows x", ncol(anomaly_year_df), "columns\n")
cat("\nAggregating...\n")
anomaly_year_df <- anomaly_year_df %>%
    group_by(year_bin) %>% # Aggregate anomalies per year_bin
    summarise(anomaly = mean(anomaly, na.rm = TRUE)) %>%
    ungroup()

cat("Shape of anomaly_year_df:", nrow(anomaly_year_df), "rows x", ncol(anomaly_year_df), "columns\n")

cat("\nCreating plots...\n")
long_term_plot_config <- ggplot(anomaly_year_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = -700000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -700000, y = 5, label = "Quaternary Glaciation Intensifies", hjust = 1.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = -110000, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = -110000, y = 5, label = "Last Ice Age Starts", hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 8, limits = c(-10, 10)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-10, 10)) +
    scale_x_continuous(labels = scales::comma)

long_term_plot_config + geom_point()
ggsave("Outputs/long_term_temperature_anomaly_scatter_plot.png", width = 10, height = 6)

long_term_plot_config + geom_line()
ggsave("Outputs/long_term_temperature_anomaly_line_plot.png", width = 10, height = 6)

since_ice_age_plot_config <- ggplot(anomaly_year_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
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
    scale_y_continuous(limits = c(-3, 3)) +
    scale_x_continuous(limits = c(-12000, 2025), labels = scales::comma)

since_ice_age_plot_config + geom_line()
ggsave("Outputs/since_ice_age_temperature_anomaly_line_plot.png", width = 10, height = 6)

since_ice_age_plot_config + geom_point()
ggsave("Outputs/since_ice_age_temperature_anomaly_scatter_plot.png", width = 10, height = 6)

latest_anomaly <- tail(anomaly_year_df$anomaly, 1)

modern_plot_config <- ggplot(anomaly_year_df, aes(x = year_bin, y = anomaly, color = anomaly)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#DF00A7") +
    annotate("text", x = -Inf, y = 0, label = "Long-term Climate Average", hjust = -0.1, vjust = -0.5, color = "black") +
    geom_vline(xintercept = 2024, linetype = "dashed", color = "#0072F5") +
    annotate("text", x = 2024, y = latest_anomaly, label = sprintf("2025: Anomaly of +%.2f°C", latest_anomaly), hjust = 1.1, vjust = -0.5, color = "black") +
    scale_color_gradient2(low = "blue", mid = "red", high = "red", midpoint = 2, limits = c(-2, 2)) +
    labs(x = "Year", y = "Temperature Anomaly (°C)") +
    theme_classic() +
    scale_y_continuous(limits = c(-2, 2)) +
    scale_x_continuous(limits = c(1850, 2025), labels = scales::comma)

modern_plot_config + geom_line()
ggsave("Outputs/modern_temperature_anomaly_line_plot.png", width = 10, height = 6)

modern_plot_config + geom_point()
ggsave("Outputs/modern_temperature_anomaly_scatter_plot.png", width = 10, height = 6)

write.csv(anomaly_year_df, "Data/anomaly_year.csv", row.names = FALSE)

DBI::dbDisconnect(conn)
cat("\nDisconnected from database\n")
