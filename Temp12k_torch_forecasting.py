# %%
# Imports
import polars as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np

# %%
# Load the dataset
full_df = pl.read_csv(
    "Outputs/long_term_global_anomaly_view_enriched_training.csv"
).drop_nans()
train_df = full_df.filter(pl.col("year_bin") < -100000)

# Prepare the data
features = train_df.drop("anomaly", "year_bin").to_numpy()
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


# K-Fold Cross Validation
k_folds = 5
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

    # Train the model
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            # Forward pass
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    fold_residuals = []
    with torch.no_grad():
        for val_features, val_targets in val_loader:
            val_predictions = model(val_features)
            val_loss += criterion(val_predictions, val_targets).item()
            fold_residuals.extend((val_targets - val_predictions).squeeze().numpy())

    val_loss /= len(val_loader)
    print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")
    fold_results.append(val_loss)
    residuals.extend(fold_residuals)
    torch.save(model.state_dict(), f"models/model_fold_{fold + 1}.pth")

# Aggregate residuals and compute R-squared
residuals = np.array(residuals)
total_variance = np.var(targets_tensor.numpy())
explained_variance = total_variance - np.var(residuals)
r_squared = explained_variance / total_variance

# Average validation loss across folds
avg_val_loss = np.mean(fold_results)
print(f"Average Validation Loss: {avg_val_loss:.4f}")
print(f"R-squared of the ensemble: {r_squared:.4f}")

# %%
# Evaluate on full dataset
full_features = full_df.drop("anomaly", "year_bin").to_numpy()
full_features_tensor = torch.tensor(full_features, dtype=torch.float32)

all_predictions = []
for fold in range(k_folds):
    model.load_state_dict(torch.load(f"models/model_fold_{fold + 1}.pth"))
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        fold_predictions = model(full_features_tensor).squeeze().numpy()
        all_predictions.append(fold_predictions)

# Combine predictions from all folds (e.g., average them)
predictions = sum(all_predictions) / len(all_predictions)

# Save the predictions as pred_anomaly
pred_df = full_df.with_columns(pl.Series("pred_anomaly", predictions))

# Plot anomaly and pred_anomaly as time series
plt.figure(figsize=(12, 6))
plt.plot(pred_df["year_bin"], pred_df["anomaly"], label="Actual Anomaly", alpha=0.7)
plt.plot(
    pred_df["year_bin"], pred_df["pred_anomaly"], label="Predicted Anomaly", alpha=0.7
)
plt.axvline(x=-100000, color="red", linestyle="--", label="Test Data Cutoff")
plt.xlabel("Year Bin")
plt.ylabel("Anomaly")
plt.title("Actual vs Predicted Anomaly (Test Data)")
plt.legend()
plt.show()

# %%
