import netCDF4 as nc
import numpy as np
import warnings
import os
import math

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
warnings.filterwarnings("ignore", category=UserWarning)
if os.getcwd().endswith(
    "modules"
):  # If running directly from modules, go to directory containing Data
    os.chdir("..")
modern_temperature_grid = nc.Dataset("Data/Land_and_Ocean_LatLong1.nc", mode="r")
"""
print(dataset) outputs
root group (NETCDF4 data model, file format HDF5):
    Conventions: Berkeley Earth Internal Convention (based on CF-1.5)
    title: Native Format Berkeley Earth Surface Temperature Anomaly Field
    history: 09-Jan-2025 20:24:26
    institution: Berkeley Earth Surface Temperature Project
    land_source_history: 04-Jan-2025 19:11:11
    ocean_source_history: 06-Jan-2025 11:47:59
    comment: This file contains Berkeley Earth surface temperature anomaly field in our native equal-area format.
    dimensions(sizes): map_points(15984), time(2100), month_number(12)
    variables(dimensions): float32 longitude(map_points), float32 latitude(map_points), float64 time(time), float64 land_mask(map_points), float32 temperature(time, map_points), float32 climatology(month_number, map_points)
    groups:
"""


def find_nearest(array, value):
    """
    Finds the nearest value in an array.

    Parameters:
    array (numpy array): The array to search.
    value (float): The value to find.

    Returns:
    int: The index of the nearest value.
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def get_average_temperature(lat, lon, month=None):
    """
    Gets the average temperature for the nearest latitude and longitude.

    Parameters:
    dataset (Dataset): The netCDF4 dataset.
    lat (float): The latitude.
    lon (float): The longitude.

    Returns:
    float: The average temperature.
    """
    # temperatures = dataset.variables['temperature']
    lat_idx = find_nearest(modern_temperature_grid.variables["latitude"][:], lat)
    lon_idx = find_nearest(modern_temperature_grid.variables["longitude"][:], lon)
    # The temperature variable holds anomaly delta degC for each month since 1850, while the climatology variable holds the historical average degC for each month
    last_year = modern_temperature_grid.variables["climatology"][:, lat_idx, lon_idx]
    if month:
        month_idx = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ].index(month)
        return last_year[month_idx]
    else:
        return np.mean(last_year)


def get_temperature_when(lat, lon, year, month=None):
    month_time_series = []
    index_year = year - 1850
    if month:
        indexes = [index_year * 12 + months.index(month)]
    else:
        indexes = range(index_year * 12, (index_year + 1) * 12)
    for time in indexes:
        temperature_when = modern_temperature_grid.variables["temperature"][
            time, lat, lon
        ]
        forward_offset = 0
        while (
            type(temperature_when) == np.ma.core.MaskedConstant
        ):  # Missing values are masked constants
            forward_offset += 12
            temperature_when = modern_temperature_grid.variables["temperature"][
                time + forward_offset, lat, lon
            ]
        month_time_series.append(temperature_when)
    # print(month_time_series)
    return np.mean(month_time_series)


def missouri_example():
    latitude = 37  # Example latitude of Missouri
    longitude = -91  # Example longitude of Missouri
    missouri_temperature = get_average_temperature(
        modern_temperature_grid, latitude, longitude
    )
    print(missouri_temperature)
    missouri_temperatures = modern_temperature_grid.variables["climatology"][:, 37, -91]
    print(len(missouri_temperatures))
    print(missouri_temperatures)

    # for latitude in range(-90, 91):
    #    print(f"Latitude: {latitude}, Average Temperature: {get_average_temperature(dataset, latitude, longitude)}")
    # It's working!

avg = get_temperature_when(37, -91, 1950)
averages = []
# for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
#    avg = get_temperature_when(37, -91, 1950, month)

year_time_series = []
for year in range(1850, 2025):
    averages = []
    for lat, lon in zip(range(-90, 91), range(-180, 181)):
        averages.append(get_temperature_when(lat, lon, year))
    year_time_series.append(np.mean(averages))
print(len(year_time_series))
# Reference https://berkeleyearth.org/data/ for dataset
