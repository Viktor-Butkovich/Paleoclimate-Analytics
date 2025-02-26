library(tidyverse)

file_path <- "Data/forcing/Milankovitch/Milankovitch-Insolation-Extracted.csv"

milankovitch_df <- read_csv(file_path)

milankovitch_df <- milankovitch_df %>%
    mutate(year = 1950 + BP, insolation = abs(annual_insolation_65N - mean(annual_insolation_65N))) %>% # Note that this dataset uses negative BP, so we have to add instead of subtract
    select(year, insolation)

plot <- ggplot(milankovitch_df %>% filter(year >= -800000), aes(x = year, y = insolation)) +
    geom_line() +
    scale_x_continuous(labels = scales::comma) +
    theme_minimal()
x11()
print(plot)
