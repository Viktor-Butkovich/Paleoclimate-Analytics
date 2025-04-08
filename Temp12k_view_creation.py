# %%
# Imports
db_mode = "sqlite"
if db_mode == "sql_server":
    from modules import db_sqlalchemy as db
elif db_mode == "sqlite":
    from modules import db_sqlite as db
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# Load in the data
db.connect()
try:
    print(f"Loading in fact_temperature")
    fact_temperature = pl.DataFrame(db.read_table("fact_temperature"))
    print(f"Loading in dimensions")
    dimensions_dict = {
        dim: pl.DataFrame(db.read_table(dim))
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


# Round all non-year-bin columns to 5 decimal places - avoid float errors from causing different output each run
fact_temperature = round_columns(fact_temperature, 5, exclude=["year_bin"])
print(fact_temperature)

# %%
# Storing raw data
fact_temperature.sort("year_bin").write_csv("Outputs/raw_global_anomaly_view.csv")

# %%
# Preprocess the data for analysis

# Filter data for the specified year range
preprocessed = fact_temperature.filter((pl.col("year_bin") >= -700000))

# Aggregate data to have a constant frequency of 2000 years
preprocessed = (
    preprocessed.with_columns((pl.col("year_bin") // 2000 * 2000).alias("year_bin"))
    .group_by("year_bin")
    .agg(pl.all().mean())
).sort("year_bin")

# Apply a 5-row rolling mean to the solar_modulation column
preprocessed = preprocessed.with_columns(
    pl.col("solar_modulation").rolling_mean(window_size=5).alias("solar_modulation")
)

# Use normalized solar modulation, as in Interglacials, Milankovitch Cycles, Solar Activity, and Carbon Dioxide (Marsh 2014)
scaler = StandardScaler()
solar_modulation_normalized = scaler.fit_transform(
    preprocessed.select(["solar_modulation"]).to_numpy()
)
preprocessed = preprocessed.with_columns(
    pl.Series("solar_modulation", solar_modulation_normalized.flatten())
)

# Add delta columns for each column except 'year_bin'
delta_columns = [
    (pl.col(col) - pl.col(col).shift(1)).alias(f"delta_{col}")
    for col in preprocessed.columns
    if col != "year_bin"
]
preprocessed = preprocessed.with_columns(delta_columns).filter(
    pl.col("year_bin") != preprocessed["year_bin"].min()
)

# Add lagged anomaly for 40,000 and 100,000 years before
lagged_columns = [
    pl.col("anomaly").shift(lag).alias(f"anomaly_lagged_{lag}") for lag in [20, 50]
]
preprocessed = preprocessed.with_columns(lagged_columns)

# Add _squared columns for non-year-bin and non-anomaly fields
squared_columns = [
    (pl.col(col) ** 2).alias(f"{col}_squared")
    for col in preprocessed.columns
    if col not in ["year_bin", "anomaly"]
]
preprocessed = preprocessed.with_columns(squared_columns)

# Rescale columns except 'year_bin' and 'anomaly'
scaler = MinMaxScaler()
columns_to_rescale = [
    col
    for col in preprocessed.columns
    if not (
        col in ["solar_modulation", "year_bin", "anomaly", "delta_anomaly"]
        or "degC" in col
    )
]

# Exclude outlier year bin 2000 from min/max consideration
filtered_data = preprocessed.filter(pl.col("year_bin") != 2000)
scaler.fit(filtered_data.select(columns_to_rescale).to_numpy())
rescaled_data = scaler.transform(preprocessed.select(columns_to_rescale).to_numpy())
rescaled_df = pl.DataFrame(rescaled_data, schema=columns_to_rescale)

preprocessed = preprocessed.with_columns(
    [pl.Series(name, rescaled_df[name]) for name in columns_to_rescale]
)
preprocessed = preprocessed.filter(
    pl.col("year_bin") > -600000
)  # Remove first 100,000 years with null lagged values

preprocessed = round_columns(preprocessed, 5, exclude=["year_bin"])

print(preprocessed)

# %%
# Storing preprocessed data
preprocessed.write_csv("Outputs/long_term_global_anomaly_view_enriched.csv")

training = preprocessed.drop(
    [
        col
        for col in preprocessed.columns
        if "co2" in col or "be_ppm" in col or "VADM" in col or "delta_anomaly" in col
    ]
)
training.write_csv("Outputs/long_term_global_anomaly_view_enriched_training.csv")

preprocessed = preprocessed.drop(
    [
        col
        for col in preprocessed.columns
        if col.endswith("_squared") or col.startswith("delta_") or "lagged" in col
    ]
)
preprocessed.write_csv(
    "Outputs/long_term_global_anomaly_view.csv"
)  # Only contains original columns

# %%
