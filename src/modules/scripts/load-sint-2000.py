# %%
# Imports
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

# 1-time script to extract Sint2000 magnetic field strength dataset from .mat file

# %%
# Load the .mat file
data = loadmat(
    "../Data/raw/Sint2000.mat"
)  # Adjust path as necessary for directory structure

# Access the variables in the file
print(data.keys())  # Lists all variables in the .mat file

# %%
# Transformations
t_values = data["t"]
years = ((t_values * 1000) + 3000).flatten().round().astype(int)
print(f"Range of years: min={years.min()}, max={years.max()}")

data = data["d"]
data = data.flatten()

# Create a DataFrame with 'year' and 'data' as columns
df = pd.DataFrame({"year": years, "VADM": data})  # virtual axis dipole moment

# %%
# EDA plots

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df["year"], df["VADM"], label="Data over time")
plt.xlabel("Year")
plt.ylabel("VADM")
plt.title("Time Series of VADM")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Save the DataFrame to a CSV file
df.to_csv("../Data/VADM.csv", index=False)

# %%
