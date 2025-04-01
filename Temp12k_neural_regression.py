# Deprecated

# %%
# Imports
print("Loading imports...")

# Usage sample provided in https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html

import polars as pl
import matplotlib.pyplot as plt
import concurrent.futures
import tensorflow as tf
import seaborn as sns
import random
import json

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
df = (
    df.with_columns(
        pl.col("anomaly").shift(50).alias("anomaly_lag_50"),
    )
).drop_nulls()

X_train, X_test = df[: int(len(df) * 0.9)].drop("anomaly", "year_bin"), df[
    int(len(df) * 0.9) :
].drop("anomaly", "year_bin")
Y_train, Y_test = (
    df[: int(len(df) * 0.9)][["anomaly"]],
    df[int(len(df) * 0.9) :][["anomaly"]],
)
feature_names = X_train.columns

# %%
# Define model class
class model:
    def __init__(self, genome):
        self.fitness = 1000
        if sum(genome["feature_mask"]) == 0:
            genome["feature_mask"][
                random.randrange(len(genome["feature_mask"]))
            ] = 1  # Use at least 1 feature
        self.genome = genome
        self.epochs = genome["epochs"]
        self.learning_rate = genome["learning_rate"]
        self.feature_mask = genome["feature_mask"]
        # Define a simple sequential model
        layers = []
        for idx in range(genome["num_layers"]):
            if idx == 0:
                layers.append(tf.keras.layers.Input(shape=(sum(self.feature_mask),)))
                layers.append(
                    tf.keras.layers.Dense(genome["layer_size"], activation="relu")
                )
            else:
                layers.append(
                    tf.keras.layers.Dense(genome["layer_size"], activation="relu")
                )
        layers.append(tf.keras.layers.Dense(1))

        self.model = tf.keras.Sequential(layers)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

    def get_masked_features(self, X):
        return X.select(
            pl.col(col) for col, mask in zip(X.columns, self.feature_mask) if mask == 1
        )

    def evaluate(self):
        if self.fitness != 1000:
            return
        # Train the model
        tf.random.set_seed(0)
        self.model.fit(
            self.get_masked_features(X_train),
            Y_train,
            epochs=self.epochs,
            validation_split=0.1,
            verbose=0,
        )

        # Evaluate the model
        loss = self.model.evaluate(
            self.get_masked_features(X_test), Y_test
        )  # verbose = 0
        self.fitness = loss

    def predict(self, prediction_df):
        return prediction_df.with_columns(
            pl.Series(
                name="predicted_anomaly",
                values=self.model.predict(
                    prediction_df.drop("year_bin", "anomaly"), verbose=0
                ).flatten(),
            )
        )

    def plot(self, prediction_df, save=False, generation_number=None):
        # Create a DataFrame for seaborn
        prediction_df = prediction_df.select(
            ["year_bin", "anomaly", "predicted_anomaly"]
        )

        # Plot the actual and predicted anomalies using seaborn
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            data=prediction_df, x="year_bin", y="anomaly", label="Actual Anomaly"
        )
        sns.lineplot(
            data=prediction_df,
            x="year_bin",
            y="predicted_anomaly",
            label="Predicted Anomaly",
            linestyle="--",
        )
        plt.xlabel("Year")
        plt.ylabel("Anomaly")
        plt.ylim(-10, 10)  # Set the y-axis range from -10 to 10
        plt.title(
            f'Actual vs Predicted Anomalies (fitness {self.fitness:.3f}, {self.genome["layer_size"] * self.genome["num_layers"]} neurons)'
        )
        plt.legend()
        if save:
            plt.savefig(f"evolution_progress/generation_{generation_number}_best.png")

    def __str__(self):
        feature_names_selected = self.get_masked_features(X_train).columns
        return f"fitness: {self.fitness:.3f}, epochs: {self.epochs}, learning_rate: {self.learning_rate:.5f}, layer_size: {self.genome['layer_size']}, num_layers: {self.genome['num_layers']}, feature_mask: {feature_names_selected}"


# %%
# Model training


def initialize_individual():
    return {
        "epochs": random.randrange(5, 41),
        "learning_rate": random.uniform(1e-4, 5 * 1e-3),
        "num_layers": random.choice([1, 2, 3, 4, 5, 6]),
        "layer_size": random.choice([4, 8, 16, 24, 32, 40, 48]),
        "feature_mask": [
            1 for _ in range(X_train.shape[1])
        ],  # [random.choice([0, 1]) for _ in range(X_train.shape[1])]
    }


random.seed(0)

population_size = 10
num_generations = 20
mutation_rate = 0.5
population = [model(initialize_individual()) for _ in range(population_size)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda individual: individual.evaluate(), population)
# Sort population by fitness in ascending order
population.sort(key=lambda x: x.fitness)
best_individual = population[0]
print(f"Initial generation: Best individual {best_individual}")
best_individual.plot(best_individual.predict(df), save=True, generation_number=0)

for generation_number in range(1, num_generations + 1):
    new_population = [best_individual]
    parent_weights = [1 / individual.fitness for individual in population]
    child_genomes = []
    for _ in range(population_size - 1):
        parents = random.choices(population, k=2, weights=parent_weights)
        inheritance_mask = [
            random.choice([0, 1]) for _ in range(len(parents[0].genome))
        ]
        child_1_genome = {
            key: parents[0].genome[key] if mask == 0 else parents[1].genome[key]
            for mask, key in zip(inheritance_mask, parents[0].genome)
        }
        child_genomes.append(child_1_genome)

        child_2_genome = {
            key: parents[1].genome[key] if mask == 0 else parents[0].genome[key]
            for mask, key in zip(inheritance_mask, parents[0].genome)
        }
        child_genomes.append(child_2_genome)
    for child_genome in child_genomes:
        for key in child_genome:
            if random.random() < mutation_rate:
                if key == "feature_mask":
                    continue
                    # child_genome[key][random.randrange(len(child_genome["feature_mask"]))] = random.choice([0, 1])
                elif key == "num_layers":
                    child_genome[key] += random.choice([-1, 1])
                    child_genome[key] = max(3, child_genome[key])
                elif key == "layer_size":
                    child_genome[key] += random.choice([-4, 4])
                    child_genome[key] = max(8, child_genome[key])
                elif key == "epochs":
                    child_genome[key] += random.choice([-10, 10])
                    child_genome[key] = max(1, child_genome[key])
                elif key == "learning_rate":
                    child_genome[key] *= random.uniform(0.5, 1.5)
                    child_genome[key] = max(1e-2, child_genome[key])
        population.append(model(child_genome))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda individual: individual.evaluate(), population)

    # Sort population by fitness in ascending order
    population.sort(key=lambda x: x.fitness)
    if population[0].fitness < best_individual.fitness:
        best_individual = population[0]
        print(f"Generation {generation_number}: New best individual {population[0]}")
    else:
        print(f"Generation {generation_number}: Same best individual")
    best_individual.plot(
        best_individual.predict(df), save=True, generation_number=generation_number
    )
    population = population[:population_size]
    with open("best_individual_genome.json", "w") as f:
        json.dump(best_individual.genome, f)

# Not much improvement over time - possibly excessive elitism, insufficient data, or incorrect EC parameters
#   However, this still performs better than polynomial regression models, with MSE of 1.88 compared to 3.64
#   This is also fairly long-horizon forecasting - integrating more recent lagged data may improve performance
#   Evolution is stuck on the minimum # layers and layer size - models are not being sufficiently incentivized to increase complexity

# %%
# Manual tests for new parameter regions
manual_model = model(
    {
        "epochs": 50,
        "learning_rate": 0.01,
        "num_layers": 20,
        "layer_size": 24,
        "feature_mask": [1 for _ in range(X_train.shape[1])],
    }
)
manual_model.evaluate()
manual_model.plot(manual_model.predict(df))
# %%
