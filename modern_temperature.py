import netCDF4 as nc
import numpy as np
import warnings

def read_nc_file(file_path):
    """
    Reads a .nc file and returns the dataset.

    Parameters:
    file_path (str): The path to the .nc file.

    Returns:
    Dataset: The netCDF4 dataset.
    """
    dataset = nc.Dataset(file_path, mode='r')
    return dataset

"""
The above outputs
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

def get_average_temperature(dataset, lat, lon):
    """
    Gets the average temperature for the nearest latitude and longitude.

    Parameters:
    dataset (Dataset): The netCDF4 dataset.
    lat (float): The latitude.
    lon (float): The longitude.

    Returns:
    float: The average temperature.
    """
    latitudes = dataset.variables['latitude']
    longitudes = dataset.variables['longitude']
    # temperatures = dataset.variables['temperature']
    # climatology = dataset.variables['climatology']
    lat_idx = find_nearest(latitudes[:], lat)
    lon_idx = find_nearest(longitudes[:], lon)
    # The temperature variable holds anomaly, while the climatology variable holds the average temperature
    return dataset.variables['climatology'][:, lat_idx, lon_idx][-13:].mean()

warnings.filterwarnings("ignore", category=UserWarning)

dataset = read_nc_file("Data/Land_and_Ocean_LatLong1.nc")
latitude = 37  # Example latitude of Missouri
longitude = -91  # Example longitude of Missouri
missouri_temperature = get_average_temperature(dataset, latitude, longitude)
print(missouri_temperature)

for latitude in range(-90, 91):
    print(f"Latitude: {latitude}, Average Temperature: {get_average_temperature(dataset, latitude, longitude)}")
    # It's working!

# Reference https://berkeleyearth.org/data/ for dataset
