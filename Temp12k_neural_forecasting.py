# %%
print("Loading imports...")

# Usage sample provided in https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html

# import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

# %%
print("Loading data...")

df = (
    pl.read_csv("Data/anomaly_training_ready.csv")
    .with_columns(pl.lit(1).alias("unique_id"))
    .rename({"year_bin": "ds", "anomaly": "y"})
    .with_columns((pl.col("ds") / 2000).alias("ds"))
    .with_columns(pl.col("ds").cast(pl.Date))
)
Y_train = df[: int(len(df) * 0.9)]
Y_test = df[int(len(df) * 0.9) :]

# %%
print("Fitting model...")
# Documentation: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html
horizon = len(Y_test)
model = NHITS(
    h=horizon,
    input_size=horizon * 5,
    loss=DistributionLoss(distribution="StudentT", level=[80, 90], return_params=True),
    futr_exog_list=[
        "co2_ppm",
        "co2_radiative_forcing",
        "eccentricity",
        "obliquity",
        "perihelion",
        "insolation",
        "global_insolation",
    ],
    n_freq_downsample=[2, 1, 1],
    scaler_type="robust",
    max_steps=200,
    early_stop_patience_steps=2,
    inference_windows_batch_size=1,
    val_check_steps=10,
    learning_rate=1e-3,
)

fcst = NeuralForecast(models=[model], freq="1d")
fcst.fit(df=Y_train, val_size=horizon)

# %%
print("Making forecasts")
forecasts = fcst.predict(futr_df=Y_test)

# %%
print("Plotting results...")
# Plot quantile predictions

Y_plot = (
    Y_test.with_columns(forecasts["NHITS"].alias("NHITS"))
    .with_columns((pl.col("ds").cast(pl.Int32) * 2000).alias("year_bin"))
    .drop("ds")
)
plt.figure(figsize=(10, 6))

plt.plot(Y_plot["year_bin"], Y_plot["y"], label="Actual Anomaly")
plt.plot(Y_plot["year_bin"], Y_plot["NHITS"], label="NHITS Forecast", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Anomaly")
plt.title("Actual vs NHITS Forecast Anomaly")
plt.legend()
plt.show()

print("Completed!")

# %%
