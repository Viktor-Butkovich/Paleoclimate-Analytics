# %%
# Imports
import time
import random
from copy import deepcopy

start_time = time.time()
print(f"Starting script at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor
from modules import constants
from typing import Dict, List, Any
import numpy as np
import json

# %%
# Load the dataset
config = json.load(open("../prediction_config.json"))

device = None
if config["gpu"] and torch.cuda.is_available():
    device = torch.device("cuda")
    # device = torch.device("cpu")
    print(f"GPU is available - using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

full_df = (
    pl.read_csv("../Outputs/long_term_global_anomaly_view_enriched_training.csv")
    .with_columns(
        pl.when(pl.col("year_bin") > config["present"])
        .then(None)
        .otherwise(pl.col("anomaly"))
        .alias("anomaly")
    )
    .filter(pl.col("year_bin") <= config["forecast_end"])
)
full_features = full_df.drop(
    "anomaly", "year_bin"
).to_numpy()  # All input features for entire time series, designed for final prediction
full_features_tensor = torch.tensor(full_features, dtype=torch.float32).to(device)

# %%
# Split the dataset into training and test sets
test_df = full_df.filter(
    (
        (pl.col("year_bin") >= config["test_start"])
        & (pl.col("year_bin") <= config["test_end"])
    )
    | (pl.col("year_bin") > config["present"])
)
train_df = full_df.filter(
    (
        (
            (pl.col("year_bin") < config["test_start"])
            | (pl.col("year_bin") > config["test_end"])
        )
    )
    & (pl.col("year_bin") <= config["present"])
)

# Prepare the data
features = train_df.drop("anomaly", "year_bin").to_numpy()
input_size = features.shape[1]
print(f"Training on {train_df.drop('anomaly', 'year_bin').columns}")

targets = train_df["anomaly"].to_numpy()

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device)


# %%
# Define training/evaluation functions
class AnomalyPredictor(nn.Module):
    def __init__(
        self, input_size: int, layer_sizes: List[int], device: torch.device
    ) -> None:
        super(AnomalyPredictor, self).__init__()
        layers = []
        previous_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(previous_size, size))
            layers.append(nn.ReLU())
            previous_size = size
        layers.append(nn.Linear(previous_size, 1))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def run_loss_train(
    model: AnomalyPredictor,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    regularization_lambda: float,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Description: Trains the inputted model for 1 epoch
    """
    model.train()
    total_loss = 0.0
    for batch_features, batch_targets in data_loader:
        predictions = model(batch_features)
        loss = criterion(predictions, batch_targets)
        l2_norm = sum(
            p.pow(2).sum() for p in model.parameters()
        )  # L2 (ridge) regularization - loss penalty for larger parameter sizes (automatically ignores important parameters)
        loss += regularization_lambda * l2_norm
        optimizer.zero_grad()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()


def evaluate_loss(
    model: AnomalyPredictor,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    regularization_lambda: float,
):
    """
    Description: Returns the loss of the inputted model on the inputted data
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += regularization_lambda * l2_norm
            total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_predictions(
    model: AnomalyPredictor, features_tensor: torch.tensor
) -> np.ndarray:
    """
    Description: Returns the predictions of the inputted model on the inputted data
    """
    model.eval()
    with torch.no_grad():
        fold_predictions = model(features_tensor).squeeze().cpu().numpy()
    return fold_predictions


def train_fold(
    fold: int,
    train_idx: int,
    val_idx: int,
    hyperparameters: Dict[str, Any],
    verbosity: int = 1,
) -> Dict[str, Any]:
    if verbosity == 2:
        print(f"Starting Fold {fold + 1}/{k_folds}")

    # Split data into training and validation sets
    train_features, val_features = features_tensor[train_idx], features_tensor[val_idx]
    train_targets, val_targets = targets_tensor[train_idx], targets_tensor[val_idx]

    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = AnomalyPredictor(input_size, hyperparameters[constants.LAYER_SIZES], device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparameters[constants.ADAM_LR]
    )

    prev_val_loss = float("inf")
    original_patience = hyperparameters[constants.PATIENCE]
    patience = original_patience
    for epoch in range(hyperparameters[constants.EPOCHS]):
        # Train the model
        run_loss_train(
            model,
            train_loader,
            criterion,
            hyperparameters[constants.REGULARIZATION_LAMBDA],
            optimizer,
        )

        # Early stopping: break if validation loss didn't improve
        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            hyperparameters[constants.REGULARIZATION_LAMBDA],
        )
        if val_loss > prev_val_loss:
            patience -= 1
            if patience <= 0:
                if verbosity == 2:
                    print(
                        f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss."
                    )
                break
        else:
            patience = original_patience
        prev_val_loss = val_loss

    # Show final fold performance
    val_loss = evaluate_loss(
        model, val_loader, criterion, hyperparameters[constants.REGULARIZATION_LAMBDA]
    )
    train_loss = evaluate_loss(
        model, train_loader, criterion, hyperparameters[constants.REGULARIZATION_LAMBDA]
    )

    return {
        constants.MODEL: model,
        constants.TRAIN_LOSS: train_loss,
        constants.VAL_LOSS: val_loss,
    }


class Individual:
    def __init__(self, genome: Dict[str, Any]) -> None:
        self.genome = genome
        if not self.genome:
            self.genome = {
                constants.LAYER_SIZES: [64, 32],
                constants.EPOCHS: 200,
                constants.PATIENCE: 10,
                constants.ADAM_LR: 0.001,
                constants.REGULARIZATION_LAMBDA: 0.05,
            }
        self.fitness: float = None
        self.state_dicts: List[Dict[str, Any]] = []

    def evaluate(self, kf: KFold) -> None:
        self.state_dicts = []
        if config["parallel"]:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        train_fold,
                        fold,
                        train_idx,
                        val_idx,
                        hyperparameters=self.genome,
                    )
                    for fold, (train_idx, val_idx) in enumerate(
                        kf.split(features_tensor)
                    )
                ]
                fold_results = [
                    future.result()[constants.VAL_LOSS] for future in futures
                ]
                fold_train_results = [
                    future.result()[constants.TRAIN_LOSS] for future in futures
                ]
                self.state_dicts = [
                    future.result()[constants.MODEL].state_dict() for future in futures
                ]
        else:
            fold_results = []
            fold_train_results = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(features_tensor)):
                result = train_fold(
                    fold, train_idx, val_idx, hyperparameters=self.genome
                )
                fold_results.append(result[constants.VAL_LOSS])
                fold_train_results.append(result[constants.TRAIN_LOSS])
                self.state_dicts.append(result[constants.MODEL].state_dict())
        self.fitness = np.mean(fold_results) - self.get_complexity_penalty()

    def get_complexity_penalty(self):
        # Regularization penalty to discourage complexity with only small performance improvements
        return 0.0

    def predict(self, features_tensor: torch.tensor) -> np.ndarray:
        predictions = []
        for fold, state_dict in enumerate(self.state_dicts):
            model = AnomalyPredictor(
                input_size, self.genome[constants.LAYER_SIZES], device
            )
            model.load_state_dict(state_dict)
            fold_predictions = evaluate_predictions(model, features_tensor)
            predictions.append(fold_predictions)
        # Combine predictions from all folds (e.g., average them)
        return np.mean(predictions, axis=0)

    def __str__(self) -> str:
        layer_config = " -> ".join(
            map(str, [input_size] + self.genome[constants.LAYER_SIZES] + [1])
        )
        hyperparameters = "\n".join(
            f"        {key}: {value}" for key, value in self.genome.items()
        )
        return (
            f"    Layer Configuration: {layer_config}\n"
            f"    Hyperparameters:\n{hyperparameters}\n"
            f"    Average Validation Loss: {self.fitness:.4f}"
        )

    def mutate(self) -> "Individual":
        mut = deepcopy(self.genome)
        if random.random() < mutation_rate:
            i = random.randint(0, len(mut[constants.LAYER_SIZES]) - 1)
            if random.random() < 0.5:
                mut[constants.LAYER_SIZES][i] *= 2  # Double the size of a layer
            else:
                mut[constants.LAYER_SIZES][i] = max(
                    1, mut[constants.LAYER_SIZES][i] // 2
                )  # Halve the size of a layer, minimum 1

        if random.random() < mutation_rate:
            if random.random() < 0.5:
                if random.random() < 0.5:  # Add copy of last layer
                    mut[constants.LAYER_SIZES].append(mut[constants.LAYER_SIZES][-1])
                else:  # Add copy of 1st layer
                    mut[constants.LAYER_SIZES] = [mut[constants.LAYER_SIZES][0]] + mut[
                        constants.LAYER_SIZES
                    ]
            elif mut[constants.LAYER_SIZES]:
                if random.random() < 0.5:  # Remove last layer
                    mut[constants.LAYER_SIZES].pop()
                else:  # Remove 1st layer
                    mut[constants.LAYER_SIZES].pop(0)

        for key in [
            constants.ADAM_LR,
            constants.REGULARIZATION_LAMBDA,
        ]:  # Mutate float hyperparameters
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    mut[key] *= 2
                else:
                    mut[key] = round(
                        max(0.0001, mut[key] / 2), config["anomaly_decimal_places"]
                    )

        for key in [
            constants.EPOCHS,
            constants.PATIENCE,
        ]:  # Mutate integer hyperparameters
            if random.random() < mutation_rate:
                if random.random() < 0.5:
                    mut[key] = round(mut[key] * 1.1)
                else:
                    mut[key] = max(1, round(mut[key] * 0.9))

        return Individual(mut)

    def recombine(self, other: "Individual") -> List["Individual"]:
        if random.random() < recombination_rate:
            rec1, rec2 = {}, {}
            for key in self.genome.keys():
                if random.random() < 0.5:
                    parent1, parent2 = self, other
                else:
                    parent1, parent2 = other, self
                rec1[key] = parent1.genome[key]
                rec2[key] = parent2.genome[key]
            return [Individual(rec1), Individual(rec2)]
        else:
            return [self, other]


def evaluate_population(
    population: List[Individual], kf: KFold, verbosity: int = 1
) -> List[Individual]:
    """
    Description: Evaluates the population of individuals using K-Fold Cross Validation
    """
    for i, individual in enumerate(population):
        individual.evaluate(kf)
        if verbosity == 2:
            print(f"Evaluated individual {i + 1}: \n\n{individual}")
        elif verbosity == 1:
            print(f"Evaluated individual {i + 1} with fitness {individual.fitness:.4f}")
    return population


def sort_population(population: List[Individual]) -> List[Individual]:
    return sorted(population, key=lambda x: x.fitness)


# %%
# Train the neural networks

# Set random seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k_folds = 2
mutation_rate = 0.3
recombination_rate = 0.3
population_size = 10
num_generations = 10

kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)  # K-Fold Cross Validation
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale fitness weights from 0 to 1
default_individual = Individual(genome=None)
default_individual.evaluate(kf)
print(f"Default individual: \n{default_individual}")

population = [
    Individual(
        genome={
            constants.LAYER_SIZES: [random.choice([8, 16, 32, 64])]
            * random.randint(1, 4),
            constants.EPOCHS: random.randint(50, 200),
            constants.PATIENCE: random.randint(5, 20),
            constants.ADAM_LR: random.uniform(0.0001, 0.05),
            constants.REGULARIZATION_LAMBDA: random.uniform(0.0001, 0.2),
        }
    )
    for _ in range(population_size)
]
population = evaluate_population(population, kf)
best_individual = sort_population(population)[0]
print(f"0 generation best individual: \n\n {population[0]}")
print()

for generation in range(num_generations):
    children = []
    weights = scaler.fit_transform(
        np.array([1 / individual.fitness for individual in population]).reshape(-1, 1)
    ).flatten()
    # 1 / fitness is greater for lower fitness (error) values, then normalize to 0-1
    # Recombine into the next generation, randomly basing children on parents, with lower error individuals having a higher chance of being selected
    for i in range(population_size // 2):
        parent1, parent2 = random.choices(population, weights=weights, k=2)
        children += parent1.recombine(parent2)

    # Randomly change each child's genome
    children = [child.mutate() for child in children]

    # Calculate fitness values for each child
    children = evaluate_population(children, kf)

    # Sort population, such that the first set of individuals have the best fitness
    population = sort_population(population + children)
    population = population[
        :population_size
    ]  # Keep only the best half of the population
    if best_individual != population[0]:
        best_individual = population[0]
        print(f"{generation + 1} generation best individual: \n{population[0]}")
    else:
        print(f"{generation + 1} generation same best individual")
    print()

# %%
# Identify best individual
print(f"Best individual after evolution ({k_folds} folds): \n{best_individual}")
print()

# Best individual will preserve its genome but re-evaluate to get new models and a new validation loss
# Compare with re-evaluated default individual with manually set hyperparameters

prediction_k_folds = 2  # (Optionally) retrain with more folds for final evaluation
prediction_kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
best_individual.genome[
    constants.EPOCHS
] *= 1  # (Optionally) retrain with more epochs for final evaluation
best_individual.evaluate(prediction_kf)

print(
    f"Retrained best individual on {prediction_k_folds} folds, x10 epochs: \n{best_individual}"
)
print()

default_individual.genome[constants.EPOCHS] *= 10
default_individual.evaluate(prediction_kf)
print(
    f"Retrained default individual on {prediction_k_folds} folds, x10 epochs: \n{default_individual}"
)
print()

# Evaluate on full dataset
predictions = best_individual.predict(full_features_tensor)

# %%
# Save best results

# Store the predictions as pred_anomaly
pred_df = (
    full_df.with_columns(pl.Series("pred_anomaly", predictions))
    .select("year_bin", "anomaly", "pred_anomaly")
    .with_columns(
        pl.col("anomaly").round(config["anomaly_decimal_places"]),
        pl.col("pred_anomaly").round(config["anomaly_decimal_places"]),
    )
)
pred_df.write_csv("../Outputs/genetic_torch_model_predictions.csv")

default_pred_df = (
    full_df.with_columns(pl.Series("pred_anomaly", predictions))
    .select("year_bin", "anomaly", "pred_anomaly")
    .with_columns(
        pl.col("anomaly").round(config["anomaly_decimal_places"]),
        pl.col("pred_anomaly").round(config["anomaly_decimal_places"]),
    )
)
default_pred_df.write_csv("../Outputs/torch_model_predictions.csv")

end_time = time.time()
print(f"Script finished in {end_time - start_time:.2f} seconds")
print("Saved predictions to csv")

# %%
# Update scoreboard.json with the best individual's fitness
scoreboard_path = "../Outputs/scoreboard.json"

# Load the existing scoreboard
with open(scoreboard_path, "r") as f:
    scoreboard = json.load(f)

# Update the "genetic_torch_model" entry
scoreboard["genetic_torch_model"] = round(
    best_individual.fitness, config["anomaly_decimal_places"]
)

# Save the updated scoreboard
with open(scoreboard_path, "w") as f:
    json.dump(scoreboard, f, indent=4)

print(
    f"Updated {scoreboard_path} with genetic_torch_model fitness: {best_individual.fitness:.5f}"
)

scoreboard["torch_model"] = round(
    default_individual.fitness, config["anomaly_decimal_places"]
)
with open(scoreboard_path, "w") as f:
    json.dump(scoreboard, f, indent=4)
print(
    f"Updated {scoreboard_path} with torch_model fitness: {default_individual.fitness:.5f}"
)

# %%
