
# %%
# Imports
print("Loading imports...")

# Usage sample provided in https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html

# import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

# %%
# Load data
print("Loading data...")

df = pl.read_csv("Data/anomaly_training_ready.csv")
lag = False
if lag:
    df = (
        df.with_columns(
            pl.col("anomaly").shift(5).alias("anomaly_lag_5"),
            pl.col("anomaly").shift(10).alias("anomaly_lag_10"),
            pl.col("anomaly").shift(15).alias("anomaly_lag_15"),
            pl.col("anomaly").shift(20).alias("anomaly_lag_20"),
            pl.col("anomaly").shift(25).alias("anomaly_lag_25"),
            pl.col("anomaly").shift(30).alias("anomaly_lag_30"),
        )
    ).drop_nulls()

X_train, X_test = df[:int(len(df) * 0.9)].drop("anomaly", "year_bin"), df[int(len(df) * 0.9):].drop("anomaly", "year_bin")
Y_train, Y_test = df[:int(len(df) * 0.9)][['anomaly']], df[int(len(df) * 0.9):][['anomaly']]

# %%
# Define model class
class model:
    def __init__(self, genome):
        tf.random.set_seed(0)
        self.epochs = genome["epochs"]
        learning_rate = genome["learning_rate"]
        # Define a simple sequential model
        layers = []
        for idx, layer in enumerate(genome["layers"]):
            if idx == 0:
                layers.append(tf.keras.layers.Dense(layer, activation='relu', input_shape=(X_train.shape[1],)))
            else:
                layers.append(tf.keras.layers.Dense(layer, activation='relu'))
        layers.append(tf.keras.layers.Dense(1))

        self.model = tf.keras.Sequential(layers)

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    def evaluate(self):
        # Train the model

        history = self.model.fit(X_train, Y_train, epochs=self.epochs, validation_split=0.1, verbose=0)

        # Evaluate the model
        loss = self.model.evaluate(X_test, Y_test)
        print(f"Test loss: {loss ** 0.5}")
        self.model.predict(df.drop("year_bin", "anomaly"))
        print("working")
        prediction_df = df.with_columns(pl.Series(name="predicted_anomaly", values=self.model.predict(df.drop("year_bin", "anomaly")).flatten()))
        return loss, prediction_df

    def plot(self, prediction_df):
        # Create a DataFrame for seaborn
        prediction_df = prediction_df.select(['year_bin', 'anomaly', 'predicted_anomaly'])

        # Plot the actual and predicted anomalies using seaborn
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=prediction_df, x='year_bin', y='anomaly', label='Actual Anomaly')
        sns.lineplot(data=prediction_df, x='year_bin', y='predicted_anomaly', label='Predicted Anomaly', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Anomaly')
        plt.ylim(-10, 10)  # Set the y-axis range from -10 to 10
        plt.title('Actual vs Predicted Anomalies')
        plt.legend()
        plt.show()

# %%
# Model training
sample_model = model({
    "epochs": 50,
    "learning_rate": 0.001,
    "layers": [8, 8]
})
fitness, predictions = sample_model.evaluate()
sample_model.plot(predictions)

# %%
