# %%
# Imports
from data_types import Temp12k_data
import matplotlib.pyplot as plt
import pickle as pkl
import polars as pl
import json
import pprint
import sqlite3
import pyodbc

# %%
# Read in the data - basic Exploratory Data Analysis (EDA)
data: Temp12k_data = pkl.load(open("Data/Temp12k_v1_0_0.pkl", "rb"))
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
for sample in data["TS"]:
    units.add(sample.get("paleoData_units"))
    if sample.get("paleoData_units") in ["degC", "kelvin"]:
        temperature_data.append(sample)

print("Unique units:", units)
# The data samples include various units such as m (meters), degC (degrees Celsius), kelvin, etc.
#   We want to find samples regarding temperature
json.dump(temperature_data, open("Data/temperature_data.json", "w"))

pprint.pprint(temperature_data[0])
print(f"{len(temperature_data)} samples include degC or kelvin temperature data")
# We have 1506 samples with temperature data, each of which is a time series to ~20,000 years BP
# Temperature sample 0 tracks degC temperature from ages 500 to 22,260 years BP
# As shown in the below plot, this sample was taken near the east coast of the Arabian Peninsula

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
    # Note Northern-hemisphere bias in number of samples - be cautious of this during analysis
    # We have enough data points for even a scatter plot to resemble a world map

# %%
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=(LocalDB)\\MSSQLLocalDB;"
    "DATABASE=Temp12k;"
    "Trusted_Connection=yes;"
)  # Use the connection string for your local SQL Server instance

conn = pyodbc.connect(conn_str, timeout=5)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS TS_Sample")
cursor.execute(
    """
    BEGIN
        CREATE TABLE TS_Sample (
            TS_Sample_Id INT PRIMARY KEY,
            Sample_Data NVARCHAR(MAX),
            Other_Sample_Data NVARCHAR(MAX)
        )
    END
"""
)
cursor.execute(
    """
    INSERT INTO TS_Sample (TS_Sample_Id, Sample_Data, Other_Sample_Data)
    VALUES (1, 500, 1000)
"""
)
cursor.execute(
    """
    INSERT INTO TS_Sample (TS_Sample_Id, Sample_Data, Other_Sample_Data)
    VALUES (2, 1000, 2000)
"""
)
cursor.execute(
    """
    INSERT INTO TS_Sample (TS_Sample_Id, Sample_Data, Other_Sample_Data)
    VALUES (3, 1200, 3000)
"""
)
cursor.execute(
    """
    INSERT INTO TS_Sample (TS_Sample_Id, Sample_Data, Other_Sample_Data)
    VALUES (4, 1300, 4000)
"""
)
cursor.execute(
    """
    INSERT INTO TS_Sample (TS_Sample_Id, Sample_Data, Other_Sample_Data)
    VALUES (5, 1400, 8000)
"""
)
cursor.execute("SELECT * FROM TS_Sample")
rows = cursor.fetchall()

for row in rows:
    print(row)
cursor.close()
conn.commit()
conn.close()
# %%
