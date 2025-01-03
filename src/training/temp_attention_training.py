import torch
import torch.optim as optim
import torch.utils.data as data
from sklearn.utils import shuffle
import pandas as pd
from preprocessing import generate_preprocessed_dataset as pp
from sklearn.model_selection import LeaveOneGroupOut
from config import Config
from models.attention_models import KID_PPG
import time
import numpy as np
from typing import List
import torch.nn as nn

# Constants
N_EPOCHS = 500
BATCH_SIZE = 256
N_CH = 2
CF = Config()


# Function to create temporal pairs
def create_temporal_pairs(X_in, y_in, groups_in, activity_in):
    allXs, allys, allgroups, allactivities = [], [], [], []

    for group in np.unique(groups_in):
        curX_in = X_in[groups_in == group]
        cury_in = y_in[groups_in == group]
        cur_groups_in = groups_in[groups_in == group]
        cur_activity_in = activity_in[groups_in == group]

        cur_X = np.concatenate(
            [curX_in[:-1, ...][..., None], curX_in[1:, ...][..., None]], axis=-1
        )
        cur_y = cury_in[1:]
        cur_groups = cur_groups_in[1:]
        cur_activity = cur_activity_in[1:]

        allXs.append(cur_X)
        allys.append(cur_y)
        allgroups.append(cur_groups)
        allactivities.append(cur_activity)

    X = np.concatenate(allXs, axis=0)
    y = np.concatenate(allys, axis=0)
    groups = np.concatenate(allgroups, axis=0)
    activity = np.concatenate(allactivities, axis=0)

    return X, y, groups, activity


# Create DataLoader for training and validation
def create_dataloaders(X, y, batch_size):
    dataset = data.TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Train the model
def train_model(model, train_loader, val_loader, n_epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model.to(device)
    mse_loss = nn.MSELoss()
    training_loss_list, val_loss_list = [], []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(X_batch[..., 0], X_batch[..., 1])
            loss = mse_loss(y_batch, y_pred)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        training_loss_list.append(total_loss / len(train_loader))
        val_loss = validate_model(model, val_loader, device)
        val_loss_list.append(val_loss)
        if epoch + 1 % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(train_loader)}")
            print(f"Validation Loss: {val_loss}")

    return training_loss_list, val_loss_list, model


# Validate the model
def validate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch[..., 0], X_batch[..., 1])
            loss = mse_loss(y_batch, y_pred)
            total_loss += loss.item()
    val_loss = total_loss / len(val_loader)
    return val_loss


# Example of data split using LeaveOneGroupOut
def run_temp_model_training(n_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The device used for training is: {device}")

    X, y, groups, activity = pp.preprocessing(n_epochs=n_epochs)
    X = X[:, 0, :]
    X, y, groups, activity = create_temporal_pairs(X, y, groups, activity)

    # Init model scoring parameter and function
    _model_score = lambda train_losses, val_losses: max(train_losses) + max(val_losses)
    current_best_score = 0
    best_model = None

    # Shuffle group ids and split into training and validation sets
    group_ids = np.unique(groups)
    group_ids = shuffle(group_ids)
    splits = np.array_split(
        group_ids, 4
    )  # Example: split data into 4 parts for validation
    split_loss = {}

    for i, split in enumerate(splits):
        print(f"Starting training of split: {split}")
        groups_pd = pd.Series(groups)
        test_val_indexes = groups_pd.isin(split)
        train_indexes = ~test_val_indexes

        X_train, X_val_test = X[train_indexes], X[test_val_indexes]
        y_train, y_val_test = y[train_indexes], y[test_val_indexes]

        train_loader = create_dataloaders(X_train, y_train, batch_size=BATCH_SIZE)
        val_loader = create_dataloaders(X_val_test, y_val_test, batch_size=BATCH_SIZE)

        # Build Model
        model = KID_PPG(CF.input_shape, include_attention_weights=False)

        # Train Model
        training_loss_list, val_loss_list, model = train_model(
            model,
            train_loader,
            val_loader,
            n_epochs=N_EPOCHS,
            device=device,
        )

        # Make data structure for training losses and update best model
        if _model_score(training_loss_list, val_loss_list) > current_best_score:
            best_model = model
        split_loss[i] = {
            "split": split,
            "training_loss": training_loss_list,
            "validation_loss_list": val_loss_list,
        }
    return split_loss, best_model
