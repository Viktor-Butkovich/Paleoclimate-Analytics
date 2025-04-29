# %%
# Imports
import time

start_time = time.time()
print(f"Starting script at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
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
    def __init__(self, input_size, device):
        super(AnomalyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ).to(device)

    def forward(self, x):
        return self.model(x)


def run_loss_train(model, data_loader, criterion, regularization_lambda, optimizer):
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


def evaluate_loss(model, data_loader, criterion, regularization_lambda):
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


def evaluate_predictions(model, features_tensor):
    """
    Description: Returns the predictions of the inputted model on the inputted data
    """
    model.eval()
    with torch.no_grad():
        fold_predictions = model(features_tensor).squeeze().cpu().numpy()
    return fold_predictions


def train_fold(fold, train_idx, val_idx, hyperparameters=None):
    if not hyperparameters:
        hyperparameters = {
            constants.EPOCHS: 2000,
            constants.PATIENCE: 10,
            constants.ADAM_LR: 0.001,
            constants.REGULARIZATION_LAMBDA: 0.05,  # L2 (ridge) regularization parameter
        }
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
    model = AnomalyPredictor(input_size, device)
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
        self.fitness = np.mean(fold_results)

    def predict(self, features_tensor) -> np.ndarray:
        predictions = []
        for fold, state_dict in enumerate(self.state_dicts):
            model = AnomalyPredictor(input_size, device)
            model.load_state_dict(state_dict)
            fold_predictions = evaluate_predictions(model, features_tensor)
            predictions.append(fold_predictions)
        # Combine predictions from all folds (e.g., average them)
        return np.mean(predictions, axis=0)


# %%
# Train the neural networks

# Set random seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# K-Fold Cross Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

default_individual = Individual(genome=None)
default_individual.evaluate(kf)
print(f"Average Validation Loss: {default_individual.fitness:.4f}")

# Evaluate on full dataset
predictions = default_individual.predict(full_features_tensor)

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

end_time = time.time()
print(f"Script finished in {end_time - start_time:.2f} seconds")
print("Saved predictions to csv")

# %%
