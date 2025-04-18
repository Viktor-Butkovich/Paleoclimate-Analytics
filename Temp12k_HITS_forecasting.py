# Deprecated

# %%
print("Loading imports...")

import polars as pl
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_squared_error

# %%
print("Loading data...")

df = (
    pl.read_csv("Outputs/long_term_global_anomaly_view_enriched_training.csv")
    .with_columns(pl.lit(1).alias("unique_id"))
    .rename({"year_bin": "ds", "anomaly": "y"})
    .with_columns((pl.col("ds") / 2000).alias("ds"))
    .with_columns(pl.col("ds").cast(pl.Date))
    .drop_nans()
)

df = df.drop([col for col in df.columns if "co2" in col])  # Remove co2 columns


Y_train = df[: int(len(df) * 0.9)]
Y_test = df[int(len(df) * 0.9) :]
# Try making y_test the same size as the validation set from the linear regression trials - compare performance on the same data

# %%
print("Fitting model...")
# Documentation: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html
horizon = len(Y_test)
model = NHITS(
    h=horizon,
    input_size=horizon * 2,
    loss=DistributionLoss(distribution="StudentT", level=[80, 90], return_params=True),
    futr_exog_list=[
        column for column in df.columns if not column in ["ds", "y", "unique_id"]
    ],
    stack_types=["identity", "identity", "identity"],
    n_blocks=[1, 1, 1],
    mlp_units=60 * [[24, 24]],
    n_pool_kernel_size=[2, 2, 1],
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

print("Making forecasts")
forecasts = fcst.predict(
    futr_df=Y_test, static_df=Y_train.select(["unique_id"]).unique()
)

print("Plotting results...")
# Plot quantile predictions

Y_plot = (
    Y_test.with_columns(forecasts["NHITS"].alias("NHITS"))
    .with_columns((pl.col("ds").cast(pl.Int32) * 2000).alias("year_bin"))
    .drop("ds")
)

mse = mean_squared_error(Y_plot["y"], Y_plot["NHITS"])
print(f"Mean Squared Error: {mse}")
# No exogeneous - MSE 16.327
# Exogeneous, default size - ME 13.322
# Exogeneous, x128 size - ME 12.9
# Exogeneous, x24 size - ME 14.86
# Exogeneous, 3 * 24 size - ME 11.99
# Exogeneous, 60 * 24 size - ME 9.95
# Exogeneous, 60 * 24 size, input size horizon * 2 - ME 4.59

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
print("Plotting results...")
# Plot quantile predictions
import pandas as pd

Y_hat_df = (
    forecasts.to_pandas().reset_index(drop=False).drop(columns=["unique_id", "ds"])
)
plot_df = pd.concat([Y_test.to_pandas(), Y_hat_df], axis=1)
plot_df = pd.concat([Y_train.to_pandas(), plot_df])

plt.plot(plot_df["ds"], plot_df["y"], c="black", label="True")
plt.plot(plot_df["ds"], plot_df["NHITS-median"], c="blue", label="median")
plt.fill_between(
    x=plot_df["ds"][-horizon:],
    y1=plot_df["NHITS-lo-90"][-horizon:].values,
    y2=plot_df["NHITS-hi-90"][-horizon:].values,
    alpha=0.4,
    label="90% Confidence Interval",
)
plt.legend()
plt.grid()
plt.plot()
plt.show()

print("Completed!")

# %%
