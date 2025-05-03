# %%
# Imports
import polars as pl
from sklearn.preprocessing import StandardScaler
import json
import os

config = json.load(open("../prediction_config.json"))
if config["db_mode"] == "sql_server":
    from modules import db_sqlalchemy as db
elif config["db_mode"] == "sqlite":
    from modules import db_sqlite as db


# %%
# Setup
def round_columns(df, num_places, exclude=None):
    if not exclude:
        exclude = []
    return df.with_columns(
        [
            pl.col(col).round(num_places).alias(col)
            for col in df.columns
            if col not in exclude
        ]
    )


def normalize(df, exclude=None):
    # Normalize most columns to have a mean of 0 and standard deviation of 1
    if not exclude:
        exclude = []
    scaler = StandardScaler()
    columns_to_normalize = [col for col in df.columns if col not in exclude]
    normalized_data = scaler.fit_transform(df.select(columns_to_normalize).to_numpy())
    normalized_df = pl.DataFrame(normalized_data, schema=columns_to_normalize)
    return df.with_columns(
        [pl.Series(name, normalized_df[name]) for name in columns_to_normalize]
    )


db.connect()

# %%
# Load in the data
try:
    print(f"Loading in fact_temperature")
    fact_temperature = pl.DataFrame(db.read_table("fact_temperature"))
    print(f"Loading in dimensions")
    dimensions_dict = {
        dim: round_columns(
            pl.DataFrame(db.read_table(dim)),
            config["anomaly_decimal_places"],
            exclude=["year_bin"],
        )
        for dim in ["dim_time", "dim_atmosphere", "dim_orbital", "dim_cosmic"]
    }
except Exception as e:
    print(e)
db.close()
# %%
# Aggregate data records by year, regardless of location
print(f"Aggregating fact_temperature across locations by year")
fact_temperature = (
    fact_temperature.group_by(["time_id"]).agg(
        [pl.col("anomaly").mean().alias("anomaly")]
    )
).sort("time_id")
print(fact_temperature)

# %%
# Join fact_temperature with the dimensions
print(f"Joining fact_temperature with dimensions")
fact_temperature = fact_temperature.join(
    dimensions_dict["dim_time"], on="time_id", how="right"
)
# Populate all time ID's, even if not all data is available for future times

for dim_name, dim_df in dimensions_dict.items():
    if dim_name == "dim_time":
        continue  # Skip dim_time, which has already been joined
    fact_temperature = fact_temperature.join(dim_df, on="time_id", how="left")
fact_temperature = fact_temperature.select(pl.exclude("time_id"))


# %%
# Storing raw data for visualizations (basic preprocessing to minimize floating point errors and false diffs)
round_columns(
    normalize(fact_temperature, exclude=["year_bin", "anomaly", "co2_ppm"]),
    config["anomaly_decimal_places"],
    exclude=["year_bin"],
).sort("year_bin").write_parquet(
    "../Outputs/raw_global_anomaly_view.parquet",
)

# %%
# Preprocess the data for analysis

# Aggregate data to have a constant frequency of 2000 years
preprocessed = (
    fact_temperature.with_columns((pl.col("year_bin") // 2000 * 2000).alias("year_bin"))
    .group_by("year_bin")
    .agg(pl.all().mean())
).sort("year_bin")

# Apply a rolling mean to the solar_modulation column - helps with sparse data
preprocessed = preprocessed.with_columns(
    pl.col("solar_modulation")
    .rolling_mean(window_size=3, center=True)
    .alias("solar_modulation")
)

first_populated_2000_increment = (
    preprocessed.filter(pl.col("year_bin") % 50000 != 0)["year_bin"].min() - 2000
)

preprocessed = preprocessed.filter(  # Remove rows before the first fully defined 2000-increment row
    pl.col("year_bin") >= first_populated_2000_increment
)

orbital_lags = [20, 50]
anomaly_lags = []

# Pad the preprocessed dataframe with duplicates of the earliest year bin row, ensuring that lags are defined
padding_row = preprocessed.filter(
    pl.col("year_bin") == preprocessed["year_bin"].min()
).with_columns(pl.lit(None, dtype=pl.Int64).alias("year_bin"))

padding_rows = pl.concat(
    [padding_row] * (max(orbital_lags + anomaly_lags)), how="vertical"
)
preprocessed = padding_rows.vstack(preprocessed)

# Add delta columns for each column except 'year_bin'
preprocessed = preprocessed.with_columns(
    [
        (pl.col(col) - pl.col(col).shift(1)).alias(f"delta_{col}")
        for col in preprocessed.columns
        if col != "year_bin"
    ]
)

preprocessed = normalize(preprocessed, exclude=["year_bin", "anomaly"])


# %%
# Add lagged features from 40,000 and 100,000 years before
lagged_columns = (
    [
        pl.col("anomaly").shift(lag).alias(f"anomaly_lagged_{lag}")
        for lag in anomaly_lags
    ]
    + [
        pl.col("perihelion").shift(lag).alias(f"perihelion_lagged_{lag}")
        for lag in orbital_lags
    ]
    + [
        pl.col("eccentricity").shift(lag).alias(f"eccentricity_lagged_{lag}")
        for lag in orbital_lags
    ]
    + [
        pl.col("obliquity").shift(lag).alias(f"obliquity_lagged_{lag}")
        for lag in orbital_lags
    ]
    + [
        pl.col("insolation").shift(lag).alias(f"insolation_lagged_{lag}")
        for lag in orbital_lags
    ]
)

preprocessed = preprocessed.with_columns(lagged_columns).filter(
    pl.col("year_bin").is_not_null()
)

# Add _squared columns for non-year-bin and non-anomaly fields
squared_columns = [
    (pl.col(col) ** 2).alias(f"{col}_squared")
    for col in preprocessed.columns
    if col not in ["year_bin", "anomaly"] and "lagged" not in col
]
preprocessed = preprocessed.with_columns(squared_columns)

# %%
# Apply linear interpolation to fill null values in all past rows
interpolated = (
    preprocessed.filter(pl.col("year_bin") < 2025).fill_nan(None).interpolate()
)
non_interpolated = preprocessed.filter(pl.col("year_bin") >= 2025)

# Ensure interpolated has the same data types as non_interpolated (interpolate and None values can interfere with types)
interpolated = interpolated.with_columns(
    [pl.col(col).cast(non_interpolated.schema[col]) for col in interpolated.columns]
)

preprocessed = pl.concat([interpolated, non_interpolated], how="vertical").sort(
    "year_bin"
)

# %%
# Storing preprocessed data
preprocessed = round_columns(
    preprocessed, config["anomaly_decimal_places"], exclude=["year_bin"]
)

training = (
    preprocessed.drop(
        [
            col
            for col in preprocessed.columns
            if "co2" in col
            or "be_ppm" in col
            or "VADM" in col
            or "delta_anomaly" in col
            or "squared" in col
            or "solar_modulation" in col
        ]
    )
    .filter(  # Records below -740000 are low resolution and cause overfitting to noise
        pl.col("year_bin") >= -740000
    )
    .with_columns(pl.col("anomaly").fill_null(0))
)

training.write_parquet(
    "../Outputs/long_term_global_anomaly_view_enriched_training.parquet",
)

preprocessed = preprocessed.drop(
    [
        col
        for col in preprocessed.columns
        if col.endswith("_squared") or col.startswith("delta_") or "lagged" in col
    ]
)
print("Finished saving preprocessed views")
print(preprocessed)
preprocessed.write_parquet(
    "../Outputs/long_term_global_anomaly_view.parquet"
)  # Only contains original columns

# Create an empty JSON file for the MSE scoreboard
scoreboard_path = "../Outputs/scoreboard.json"
if not os.path.exists(scoreboard_path):
    with open(scoreboard_path, "w") as f:
        json.dump({}, f)

# %%
