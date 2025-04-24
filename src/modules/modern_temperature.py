import netCDF4 as nc
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
if os.getcwd().endswith(
    "modules"
):  # If running directly from modules, go to directory containing Data
    os.chdir("../..")
modern_temperature_grid = nc.Dataset("../Data/Land_and_Ocean_LatLong1.nc", mode="r")
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


def get_climate_month(lat, lon, month_idx):
    """
    Gets the average temperature for the nearest latitude and longitude.

    Parameters:
    dataset (Dataset): The netCDF4 dataset.
    lat (float): The latitude.
    lon (float): The longitude.

    Returns:
    float: The average temperature.
    """
    # The temperature variable holds anomaly delta degC for each month since 1850, while the climatology variable holds the historical average degC for each month
    return float(
        modern_temperature_grid.variables["climatology"][
            :, round(lat) + 89, round(lon) + 179
        ][month_idx]
    )


def get_climate(lat, lon):
    # The temperature variable holds anomaly delta degC for each month since 1850, while the climatology variable holds the historical average degC for each month
    return float(
        np.mean(
            modern_temperature_grid.variables["climatology"][
                :, round(lat) + 89, round(lon) + 179
            ]
        )
    )


def get_weighted_global_average_climate():
    # Latitudes and longitudes are not proportional to surface area.
    # To get a more accurate representation of the Earth's surface, we need to account for the cosine of the latitude.
    # This is because the distance between lines of longitude decreases as you move towards the poles.
    latitudes = np.arange(-90, 91, 1)
    weights = np.cos(np.radians(latitudes))
    weighted_sum = 0
    total_weight = 0
    for lat, weight in zip(latitudes, weights):
        for lon in range(-180, 181, 1):
            weighted_sum += get_climate(lat, lon) * weight
            total_weight += weight
    return weighted_sum / total_weight


def get_anomaly_when_month(lat, lon, year, month_idx):
    temperature = modern_temperature_grid.variables["temperature"][
        (year - 1850) * 12 + month_idx, lat, lon
    ]
    if (
        type(temperature) == np.ma.core.MaskedConstant
    ):  # Replace missing value with next year]
        return get_anomaly_when_month(lat, lon, year + 10, month_idx)
    else:
        return float(temperature)


def get_anomaly_when(lat, lon, year, month_idx=None):
    if month_idx == None:
        month_time_series = [
            get_anomaly_when_month(lat, lon, year, month_idx) for month_idx in range(12)
        ]
        return float(np.mean(month_time_series))
    else:
        return get_anomaly_when_month(lat, lon, year, month_idx)


# Reference https://berkeleyearth.org/data/ for dataset
