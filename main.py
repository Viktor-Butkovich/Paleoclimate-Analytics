# %%
# Imports
from modules import data_types, db, modern_temperature
import matplotlib.pyplot as plt
import pickle as pkl
import polars as pl
import json
import pprint
import warnings
import numpy as np
import os

# %%
# Read in the data
warnings.filterwarnings("ignore", category=DeprecationWarning)

data: data_types.Temp12k_data = pkl.load(open("Data/Temp12k_v1_0_0.pkl", "rb"))
# The data has "TS" and "D" sections
json.dump(data["TS"][0], open("Data/example_ts_sample.json", "w"))
json.dump(
    data["D"][next(iter(data["D"].keys()))], open("Data/example_d_sample.json", "w")
)
# A D sample seems to have a collection of data types (temperature, age, depth, material, sensorSpecies, etc.) - tracks all variables over time from a particular study
# A TS sample seems to have a single data type (temperature, depth, etc.) paired with age - tracks a single variable over time from a particular study
# For our purposes, we should isolate the TS samples that contain relevant data types

# TS Sample 0 is a series of samples from ages 950 to 19,130 years BP (Before Present)
#   The sample has geographic metadata for the sample location
#   The main part of the sample is how sediment depth has changed over time (Not temperature)

# D Sample 0 is a collection of data types (temperature, age, depth, material, sensorSpecies, etc.), including the depth data from TS Sample 0
month_indexes = {
    "january": 0,
    "february": 1,
    "march": 2,
    "april": 3,
    "may": 4,
    "june": 5,
    "july": 6,
    "august": 7,
    "september": 8,
    "october": 9,
    "november": 10,
    "december": 11,
    "winter": 0,
    "spring": 3,
    "summer": 6,
    "fall": 9,
    "cold season": 0,
    "coldest month": 0,
    "coldest": 0,
    "mean summer": 6,
    "1; summer": 6,
    "1 (summer)": 6,
    "warmest month": 6,
    "warmest": 6,
}
temperature_data = []
units = set()
num_samples = 0
ages = []
outliers = []
rewritten_samples = []


def include_sample(sample: data_types.Temp12k_TS_sample) -> bool:
    if sample.get("paleoData_units") != "degC":
        return False
    elif sample.get("paleoData_useInGlobalTemperatureAnalysis", "TRUE") == "FALSE":
        return False
    elif "DELETE" in sample.get("paleoData_QCnotes", ""):
        return False
    elif sample.get("age", None) == None:
        return False
    elif sample.get("paleoData_values")[0] == "nan" or sample.get("age")[0] == "nan":
        return False
    else:
        return True


# %%
# Pre-processing
for sample in data["TS"]:
    units.add(sample.get("paleoData_units"))
    if include_sample(sample):
        if (
            sample.get("paleoData_interpretation")
            and sample["paleoData_interpretation"][0].get("seasonality", "").lower()
            in month_indexes
        ):
            month_idx = month_indexes[
                sample["paleoData_interpretation"][0]["seasonality"].lower()
            ]
        else:
            # Some samples labeled with month as habitatSeason, some labeled in interpretation
            month_idx = month_indexes.get(
                sample.get("paleoData_habitatSeason", None), None
            )
        lat, lon = round(sample.get("geo_meanLat")), round(sample.get("geo_meanLon"))

        num_samples += 1
        if month_idx == None:
            climate = modern_temperature.get_climate(lat, lon)
        else:
            climate = modern_temperature.get_climate_month(lat, lon, month_idx)

        anomalies = []
        most_recent_age = sample["age"][0]
        most_recent_temperature = sample["paleoData_values"][0]
        temperature_offset = 0
        if most_recent_age != "nan" and most_recent_temperature != "nan":
            if most_recent_age < -50:
                # If most recent age is in the future, adjust to be present
                #   Most samples use age as years before 1950, but some seem to use a different convention
                shift_forward = -50 - most_recent_age
                sample["age"] = [
                    age + shift_forward for age in sample["age"] if age != "nan"
                ]
                most_recent_age = sample["age"][0]
            most_recent_year = 1950 - int(most_recent_age)
            if most_recent_year > 0:
                if not "T anomalies" in sample.get("paleoData_description", ""):
                    if most_recent_year >= 1850:
                        if month_idx == None:
                            most_recent_anomaly = modern_temperature.get_anomaly_when(
                                lat, lon, most_recent_year
                            )
                        else:
                            most_recent_anomaly = (
                                modern_temperature.get_anomaly_when_month(
                                    lat, lon, most_recent_year, month_idx
                                )
                            )
                    else:
                        most_recent_anomaly = modern_temperature.get_anomaly_when(
                            lat, lon, 1850
                        )
                    # Add most recent temperature to each data point, such that the most recent temperature is assumed to be the modern average
                    #   Thus, even if the most recent temperature is different from the modern average (local variation, error), we still get an accurate anomaly vs age
        for age, temperature in zip(sample["age"], sample["paleoData_values"]):
            # Convert age (years BP) to a date (assuming current year is 1950 for BP conversion)
            if age != "nan" and temperature != "nan":
                diff = float(temperature) - most_recent_temperature
                anomaly = (
                    diff + most_recent_anomaly
                )  # Total anomaly is difference from most recent + anomaly of most recent
                temperature = climate + anomaly
                temperature_data.append(
                    {
                        "sample_id": num_samples,
                        "year": round(
                            most_recent_year - (float(age) - most_recent_age)
                        ),
                        "degC": temperature,
                        "anomaly": anomaly,
                        "geo_meanLat": lat,
                        "geo_meanLon": lon,
                    }
                )
                ages.append(age)
                anomalies.append(temperature_data[-1]["anomaly"])
        if abs(temperature_data[-1]["anomaly"]) > 20:
            sample["anomaly"] = anomalies
            rewritten_samples.append(sample)
    # According to original paper, any records that aren't units degC, variable name temperature are not calibrated
    #   C37.concentration: set of compounds produced by algae, used to estimate past sea surface temperatures
print("Unique units:", units)
print("Oldest age:", max(ages))
# The data samples include various units such as m (meters), degC (degrees Celsius), kelvin, etc.
#   We want to find samples regarding temperature
json.dump(temperature_data, open("Data/temperature_data.json", "w"))
print(
    f"{num_samples} samples include degC temperature data, resulting in {len(temperature_data)} time series data points"
)
json.dump(rewritten_samples, open("Data/anomaly_outlier_samples.json", "w"))
# We have 1506 samples with temperature data, each of which is a time series to ~20,000 years BP
# Temperature sample 0 tracks degC temperature from ages 500 to 22,260 years BP
# As shown in the below plot, this sample was taken near the east coast of the Arabian Peninsula

# %%
# Convert temperature data to a Polars DataFrame
temperature_df = pl.DataFrame(temperature_data)
# Remove temperature outliers that are more than 1.5 IQR away
q1 = temperature_df["degC"].quantile(0.25)
q3 = temperature_df["degC"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

temperature_df = temperature_df.filter(
    (temperature_df["degC"] >= lower_bound) & (temperature_df["degC"] <= upper_bound)
)
print(temperature_df)
print(
    f"Filtering outliers beyond ({lower_bound}, {upper_bound}) - check if these are the correct units"
)
max_degC = temperature_df["degC"].max()
print(f"Maximum degC: {max_degC}")

# %%
# Add recent climate data (since 1850)
recompute = False
if recompute or not os.path.exists("Data/precomputed_modern_temperature.csv"):
    array_data = modern_temperature.modern_temperature_grid.variables["temperature"][:]
    # Get the dimensions
    months, latitudes, longitudes = array_data.shape

    # Create arrays for month, latitude, and longitude
    total_elements = months * latitudes * longitudes
    indices = np.arange(total_elements)

    # Calculate the original indices
    month_array = indices // (latitudes * longitudes)
    latitude_array = ((indices % (latitudes * longitudes)) // longitudes) - 89
    longitude_array = (indices % longitudes) - 179

    # Flatten the 3-dimensional array into a 1-dimensional array
    flattened_array = array_data.flatten()

    flattened_array: np.ma.masked_array = flattened_array

    filled_array = flattened_array.filled(np.nan)

    # Replace NaN values with the item at the index 1555200 indexes later
    nan_indices = np.where(np.isnan(filled_array))[0]
    for idx in reversed(nan_indices):
        filled_array[idx] = filled_array[idx + 1555200]

    # Create a Polars DataFrame
    modern_temperature_df = pl.DataFrame(
        {
            "month": month_array,
            "geo_meanLat": latitude_array,
            "geo_meanLon": longitude_array,
            "anomaly": filled_array,
        }
    )
    # Remove odd latitudes and longitudes
    modern_temperature_df = modern_temperature_df.filter(
        (modern_temperature_df["geo_meanLat"] % 2 == 0)
        & (modern_temperature_df["geo_meanLon"] % 2 == 0)
    )

    # Group by latitude and longitude, and calculate the average anomaly for each group of 12 months
    modern_temperature_df = (
        modern_temperature_df.with_columns((1850 + pl.col("month") // 12).alias("year"))
        .group_by(["geo_meanLat", "geo_meanLon", "year"])
        .agg(pl.col("anomaly").mean().alias("anomaly"))
    )

    modern_temperature.modern_temperature_grid.variables["climatology"].shape
    # Calculate the average annual temperature for each latitude and longitude combination
    climatology_data = modern_temperature.modern_temperature_grid.variables[
        "climatology"
    ][:]
    # Get the dimensions
    months, latitudes, longitudes = climatology_data.shape

    # Calculate the average annual temperature
    average_climate = climatology_data.mean(axis=0)
    # Create arrays for latitude and longitude
    latitude_array = np.arange(latitudes) - 89
    longitude_array = np.arange(longitudes) - 179

    # Create a Polars DataFrame
    climatology_df = pl.DataFrame(
        {
            "geo_meanLat": np.repeat(latitude_array, longitudes),
            "geo_meanLon": np.tile(longitude_array, latitudes),
            "climate": average_climate.flatten(),
        }
    )
    # Remove odd latitudes and longitudes
    climatology_df = climatology_df.filter(
        (climatology_df["geo_meanLat"] % 2 == 0)
        & (climatology_df["geo_meanLon"] % 2 == 0)
    )

    # Merge climatology_df and grouped_df on latitude and longitude
    modern_temperature_df = climatology_df.join(
        modern_temperature_df, on=["geo_meanLat", "geo_meanLon"], how="inner"
    )
    modern_temperature_df = modern_temperature_df.with_columns(
        (pl.col("climate") + pl.col("anomaly")).alias("degC")
    ).drop("climate")

    # Assign a unique sample_id to each unique pair of geo_meanLat and geo_meanLon
    unique_locations = modern_temperature_df.select(
        ["geo_meanLat", "geo_meanLon"]
    ).unique()
    unique_locations = unique_locations.with_row_count(name="sample_id")

    # Join the unique locations with the merged_df to assign sample_id
    modern_temperature_df = modern_temperature_df.join(
        unique_locations, on=["geo_meanLat", "geo_meanLon"], how="left"
    )
    print(modern_temperature_df)

    modern_temperature_df = modern_temperature_df.with_columns(
        pl.col("sample_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int64),
        pl.col("degC").cast(pl.Float64),
        pl.col("anomaly").cast(pl.Float64),
        pl.col("geo_meanLat").cast(pl.Int64),
        pl.col("geo_meanLon").cast(pl.Int64),
    )
    modern_temperature_df.write_csv("Data/precomputed_modern_temperature.csv")
else:
    modern_temperature_df = pl.read_csv("Data/precomputed_modern_temperature.csv")

# Ensure the column order matches
temperature_df = pl.concat(
    [
        temperature_df,
        modern_temperature_df.select(
            [
                "sample_id",
                "year",
                "degC",
                "anomaly",
                "geo_meanLat",
                "geo_meanLon",
            ]
        ),
    ]
).with_row_count(name="temperature_id")

# %%
# Write the temperature data to the SQL Server database
print("Writing temperature records to database")
print(temperature_df)
update_db = True
if update_db:
    db.connect()
    # db.drop_table("TS_Sample")
    table_name = "TS_Sample"
    if not db.table_exists(table_name):
        db.create_table(
            table_name,
            {
                "temperature_id": "INT PRIMARY KEY",
                "sample_id": "INT",
                "year": "INT",
                "degC": "FLOAT",
                "anomaly": "FLOAT",
                "geo_meanLat": "FLOAT",
                "geo_meanLon": "FLOAT",
            },
        )
    temperature_df.write_database("TS_Sample", db.conn, if_table_exists="replace")
    # 2 methods to insert into the database
    db.close()

# %%
