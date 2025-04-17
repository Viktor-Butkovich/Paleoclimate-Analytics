# %%
# Imports
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np

# %%
# Load the dataset
present_line = 2025
prediction_line = 200000  # Don't attempt predictions past year 200,000
full_df = (
    pl.read_csv("Outputs/long_term_global_anomaly_view_enriched_training.csv")
    .with_columns(
        pl.when(pl.col("year_bin") > present_line)
        .then(None)
        .otherwise(pl.col("anomaly"))
        .alias("anomaly")
    )
    .filter(pl.col("year_bin") < prediction_line)
)
# %%
# Solar modulation seems to have an inverse effect at -64,000 - possibly some reversal in its effect needs to be modeled
reverse_solar_modulation = False
if reverse_solar_modulation:
    full_df = full_df.with_columns(
        pl.when(pl.col("year_bin") > -200000)
        .then(-pl.col("solar_modulation"))
        .otherwise(pl.col("solar_modulation"))
        .alias("solar_modulation"),
        pl.when(pl.col("year_bin") > -200000)
        .then(-pl.col("delta_solar_modulation"))
        .otherwise(pl.col("delta_solar_modulation"))
        .alias("delta_solar_modulation"),
    )
# %%
# Define the test set range
middle = True
if middle:
    test_start = -500000
    test_end = -300000
else:
    test_start = -100000
    test_end = np.inf

# Split the dataset into training and test sets
test_df = full_df.filter(
    ((pl.col("year_bin") >= test_start) & (pl.col("year_bin") <= test_end))
    | (pl.col("year_bin") > present_line)
)
train_df = full_df.filter(
    (((pl.col("year_bin") < test_start) | (pl.col("year_bin") > test_end)))
    & (pl.col("year_bin") <= present_line)
)

# Prepare the data
features = train_df.drop("anomaly", "year_bin").to_numpy()
print(f"Training on {train_df.drop('anomaly', 'year_bin').columns}")

targets = train_df["anomaly"].to_numpy()

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)


# %%
# Train the neural networks
class AnomalyPredictor(nn.Module):
    def __init__(self, input_size):
        super(AnomalyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

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
        fold_predictions = model(features_tensor).squeeze().numpy()
    return fold_predictions


# K-Fold Cross Validation
k_folds = 5
regularization_lambda = 0.05  # L2 (ridge) regularization parameter
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

input_size = features.shape[1]
fold_results = []

residuals = []

for fold, (train_idx, val_idx) in enumerate(kf.split(features_tensor)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data into training and validation sets
    train_features, val_features = features_tensor[train_idx], features_tensor[val_idx]
    train_targets, val_targets = targets_tensor[train_idx], targets_tensor[val_idx]

    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = AnomalyPredictor(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 2000
    prev_val_loss = float("inf")
    original_patience = 10
    patience = original_patience
    for epoch in range(epochs):
        # Train the model
        run_loss_train(model, train_loader, criterion, regularization_lambda, optimizer)

        # Early stopping: break if validation loss didn't improve
        val_loss = evaluate_loss(model, val_loader, criterion, regularization_lambda)
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
    val_loss = evaluate_loss(model, val_loader, criterion, regularization_lambda)
    train_loss = evaluate_loss(model, train_loader, criterion, regularization_lambda)

    print(f"Training Loss for Fold {fold + 1}: {train_loss:.4f}")
    print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")

    fold_results.append(val_loss)
    torch.save(model.state_dict(), f"models/model_fold_{fold + 1}.pth")

# Average validation loss across folds
avg_val_loss = np.mean(fold_results)
print(f"Average Validation Loss: {avg_val_loss:.4f}")

# %%
# Evaluate on full dataset
full_features = full_df.drop("anomaly", "year_bin").to_numpy()
full_features_tensor = torch.tensor(full_features, dtype=torch.float32)

all_predictions = []
for fold in range(k_folds):
    model.load_state_dict(torch.load(f"models/model_fold_{fold + 1}.pth"))
    fold_predictions = evaluate_predictions(model, full_features_tensor)
    all_predictions.append(fold_predictions)

# Combine predictions from all folds (e.g., average them)
predictions = sum(all_predictions) / len(all_predictions)

# Save the predictions as pred_anomaly
pred_df = full_df.with_columns(pl.Series("pred_anomaly", predictions)).select(
    "year_bin", "anomaly", "pred_anomaly"
)
pred_df.write_csv("Outputs/torch_model_predictions.csv")

print("Saved predictions to csv")

# %%
