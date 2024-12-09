import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.adaptive_linear_model import MotionArtifactSeparator
from loss.adaptive_filter_loss import AdaptiveFilterLoss
import numpy as np
import tqdm


def train_adaptive_filter(
    cur_activity_X: np.ndarray,
    adaptive_filter_epochs: int = 500,
    batch_size: int = 32,
    verbose=False,
):
    # Convert the numpy data to torch tensors
    X = cur_activity_X[:, 1:, ...]  # Exclude the first channel for acceleration signal
    y = cur_activity_X[:, :1, ...]  # First channel is the target

    # Convert numpy arrays to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = MotionArtifactSeparator()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = AdaptiveFilterLoss()

    # Training loop
    model.train()
    epoch_loss_list = []

    for epoch in tqdm.tqdm(range(adaptive_filter_epochs)):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()  # Zero the gradients

            inputs = inputs[:, None, :, :]
            targets = targets[:, None, :, :]

            # Forward pass
            y_pred = model(inputs)

            # Compute the loss
            loss = criterion(y_pred, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss for the current batch
            running_loss += loss.item()

        # Average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        epoch_loss_list.append(avg_loss)

        # Optionally, print the loss at intervals or at the end of each epoch
        if verbose:
            if (epoch + 1) % 1000 == 0:
                print(
                    f"Epoch [{epoch + 1}/{adaptive_filter_epochs}], Loss: {avg_loss:.4f}"
                )

    return model, epoch_loss_list
