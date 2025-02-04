# %%
# Imports
from modules import data_types, db, modern_temperature
import matplotlib.pyplot as plt
import pickle as pkl
import polars as pl
import json
import pprint
import numpy as np
from sklearn.utils import resample

# %%
# Read in the data - basic Exploratory Data Analysis (EDA)
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

temperature_data = []
units = set()
missing_ages = 0
num_samples = 0
num_data_points = 0
ages = []
for sample in data["TS"]:
    units.add(sample.get("paleoData_units"))
    if sample.get("paleoData_units") == "degC":
        if sample.get("age"):
            num_samples += 1
            for age, temperature in zip(
                sample.get("age"), sample.get("paleoData_values")
            ):
                num_data_points += 1
                # Convert age (years BP) to a date (assuming current year is 1950 for BP conversion)
                if age != "nan" and temperature != "nan":
                    temperature_data.append(
                        {
                            "temperature_id": num_data_points,
                            "sample_id": num_samples,
                            "year": 1950 - int(age),
                            "degC": float(temperature),
                            "geo_meanLat": float(sample.get("geo_meanLat")),
                            "geo_meanLon": float(sample.get("geo_meanLon")),
                        }
                    )
                    ages.append(age)
        else:
            missing_ages += 1
    # According to original paper, any records that aren't units degC, variable name temperature are not calibrated
    #   C37.concentration: set of compounds produced by algae, used to estimate past sea surface temperatures
print("Samples missing ages:", missing_ages)
print("Unique units:", units)
print("Oldest age:", max(ages))
# The data samples include various units such as m (meters), degC (degrees Celsius), kelvin, etc.
#   We want to find samples regarding temperature
json.dump(temperature_data, open("Data/temperature_data.json", "w"))
print(
    f"{num_samples} samples include degC temperature data, resulting in {len(temperature_data)} time series data points"
)
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
# Function to resample coordinates to evenly distribute latitudes and longitudes

allow_resample = False
# Re-sampling takes the median latitude from 43 (very biased) to 6 (slightly biased)
# Similarly, longitude changes from to -2 to 6
#   Initial data has notable Northwest Hemisphere bias
if allow_resample:
    # Separate latitudes into bins of 10 degrees each
    latitude_bins = np.arange(-90, 100, 10)
    latitude_bin_expr = pl.when(pl.col("geo_meanLat") < latitude_bins[0]).then(
        latitude_bins[0] - 10
    )
    for i in range(len(latitude_bins) - 1):
        latitude_bin_expr = latitude_bin_expr.when(
            (pl.col("geo_meanLat") >= latitude_bins[i])
            & (pl.col("geo_meanLat") < latitude_bins[i + 1])
        ).then(latitude_bins[i])
    latitude_bin_expr = latitude_bin_expr.otherwise(latitude_bins[-1])

    temperature_df = temperature_df.with_columns(
        latitude_bin_expr.alias("latitude_bin")
    )

    # Count the number of samples in each bin
    bin_counts = (
        temperature_df.group_by(["latitude_bin"]).agg(pl.len()).sort("latitude_bin")
    )
    max_bin_count = max(bin_counts["len"])
    resampled_data = []

    for bin_value in latitude_bins:
        bin_data = temperature_df.filter(pl.col("latitude_bin") == bin_value)
        if len(bin_data) > 0:
            resampled_bin_data = resample(
                bin_data.to_pandas(), n_samples=max_bin_count, replace=True
            )
            resampled_data.append(pl.DataFrame(resampled_bin_data))

    temperature_df = pl.concat(resampled_data)
    temperature_df = temperature_df.drop("latitude_bin")
    print(temperature_df)

# %%
# Optionally analyze only data in a specific area
allow_sector = False
if allow_sector:
    temperature_df = temperature_df.filter(
        (temperature_df["geo_meanLat"] >= 20) & (temperature_df["geo_meanLat"] <= 80)
    )
    print(temperature_df)

# %%
# Plot the temperature time series of the first sample
plot_temperature = False
if plot_temperature:
    # Collect the temperature data from the first sample
    temperature = temperature_data[0].get("paleoData_values")

    # Convert the temperature from Celsius to Fahrenheit
    temperature = [(temp * 9 / 5) + 32 for temp in temperature]

    ages = temperature_data[0].get("age")
    # Create a line plot of the temperature data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ages, temperature, color="blue", label="Temperature")
    ax.set_title("Temperature Time Series")
    ax.set_xlabel("Age (years BP)")
    ax.set_ylabel("Temperature (degF)")
    ax.legend()
    ax.grid(True)
    ax.invert_xaxis()  # Invert the x-axis to show farther ages on the left
    plt.show()

    # Save the plot to an image file
    fig.savefig("Data/sample_0_temperature_time_series.png")

    # The temperature data is noisy, but it shows a general trend of decreasing temperature over time
    #   The temperature data is in degrees Celsius, and the ages are in years Before Present (BP)

# %%
# Plot the geolocation of the samples
plot_locations = False
if plot_locations:
    # Collect the latitudes and longitudes of each sample from temperature_data
    import matplotlib.image as mpimg

    latitudes = [sample.get("geo_meanLat") for sample in temperature_data]
    longitudes = [sample.get("geo_meanLon") for sample in temperature_data]

    # Load the world map image
    img = mpimg.imread("Data/world_map.jpg")
    # This map doesn't fully line up with our scatter-plot, but it is a decent approximation
    # Use Tableau or similar for a more accurate geographic visualization

    # Create a scatter plot of the latitudes and longitudes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, extent=[-180, 180, -90, 90])

    ax.scatter(longitudes, latitudes, alpha=0.5, edgecolors="w", linewidth=0.5)
    # Highlight the first sample point in red
    ax.scatter(
        longitudes[0],
        latitudes[0],
        color="red",
        edgecolors="w",
        linewidth=0.5,
        label="Sample 0",
    )
    ax.legend()
    ax.set_title("Sample Location Coordinates")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)  # Set the bounds for longitude
    ax.set_ylim(-90, 90)  # Set the bounds for latitude
    ax.grid(True)
    plt.show()

    # Save the plot to an image file
    fig.savefig("Data/sample_locations.png")

    # Note Northern-hemisphere bias in number of samples - be cautious of this during analysis
    # We have enough data points for even a scatter plot to resemble a world map

# %%
# Write the temperature data to the SQL Server database
update_db = True
if update_db:
    db.connect()
    db.drop_table("TS_Sample")
    db.create_table(
        "TS_Sample",
        {
            "temperature_id": "INT PRIMARY KEY",
            "sample_id": "INT",
            "year": "INT",
            "degC": "FLOAT",
            "geo_meanLat": "FLOAT",
            "geo_meanLon": "FLOAT",
        },
    )

    temperature_df.write_database("TS_Sample", db.conn, if_table_exists="replace")
    # 2 methods to insert into the database

    db.close()

# %%
# Maybe clean data by using difference from today's average, rather than absolute amount - we care more about time variance than absolute amount
# Also check the month metadata of each sample if it is provided, and show that instead
# Make sure temperature API works for different months as expected in MO
