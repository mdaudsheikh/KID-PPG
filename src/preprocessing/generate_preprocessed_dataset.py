import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm import tqdm
from config import Config
from preprocessing import preprocess_data as pp
from training.adaptive_filter_training import train_adaptive_filter
from utilities import (
    channel_wise_z_score_denormalization,
    channel_wise_z_score_normalization,
)
import os


# Constants
CF = Config()


def filter_activity_data(cur_X, cur_activity, n_epochs):
    """
    Filters the data based on activity changes, normalizes it, applies the adaptive filter model, and then denormalizes it.
    """
    activity_change_indices = get_activity_change_indices(cur_activity)

    filtered_Xs = []
    for start, end in zip(activity_change_indices[:-1], activity_change_indices[1:]):
        cur_activity_X = cur_X[start:end]

        # Normalize the data by channel
        cur_activity_X, ms, stds = normalize_channels(cur_activity_X)

        # Create, train, and apply the adaptive model
        model, _ = train_adaptive_filter(
            cur_activity_X, adaptive_filter_epochs=n_epochs
        )
        X_filtered = apply_adaptive_model(model, cur_activity_X)

        # Denormalize the data
        X_filtered = denormalize_channels(X_filtered, ms, stds)
        filtered_Xs.append(X_filtered)

    return np.concatenate(filtered_Xs, axis=0)


def get_activity_change_indices(activity):
    """
    Returns indices where there is a change in activity (i.e., activity goes from 0 to 1 or vice versa).
    """
    change_indices = np.argwhere(np.abs(np.diff(activity)) > 0).flatten() + 1
    return np.insert(change_indices, [0, len(change_indices)], [0, len(activity)])


def normalize_channels(data):
    """
    Normalizes data by channels (features).
    """
    return channel_wise_z_score_normalization(data)


def denormalize_channels(data, ms, stds):
    """
    Denormalizes data by channels (features).
    """
    return channel_wise_z_score_denormalization(data, ms, stds)


def apply_adaptive_model(model, data):
    """
    Applies the adaptive model to the data and returns the filtered data.
    """
    X_acc = data[:, None, 1:, ...]  # Extract accelerometer data
    X_ppg = data[:, None, :1, ...]  # Extract PPG data

    # Perform inference
    X_filtered = adaptive_model_inference(model, X_acc, X_ppg)
    return X_filtered.detach().numpy()


def adaptive_model_inference(model, X_acc, X_ppg):
    """
    Applies the model to accelerometer data and subtracts the filtered accelerometer data from PPG data.
    """
    X_tensor = torch.tensor(X_acc, dtype=torch.float32)
    X_ppg_tensor = torch.tensor(X_ppg, dtype=torch.float32)

    # Use DataLoader to handle batches
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=len(X_acc), shuffle=True)

    for inputs in dataloader:
        X_acc_mixed = model(inputs[0])  # Apply model to input data
    return X_ppg_tensor - X_acc_mixed


def process_group_data(group, X, y, groups, activity, n_epochs):
    """
    Processes data for a specific group (i.e., filters the data and returns the processed version).
    """
    group_data = X[groups == group]
    group_labels = y[groups == group]
    group_activity = activity[groups == group]

    # Filter data based on activity changes
    filtered_X = filter_activity_data(group_data, group_activity, n_epochs=n_epochs)

    return filtered_X, group_labels, group_activity


def save_processed_data(
    all_data_X, all_data_y, all_data_groups, all_data_activity, file_path
):
    """
    Saves the processed data to a pickle file.
    """
    data = {
        "X": all_data_X,
        "y": all_data_y,
        "groups": all_data_groups,
        "act": all_data_activity,
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print("Data processing and saving complete.")
    return all_data_X, all_data_y, all_data_groups, all_data_activity


def preprocessing(n_epochs):
    """
    Main function to load data, process it for each group, and save the results to a file.
    """
    if os.path.exists(
        CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl"
    ):
        with open(
            CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl", "rb"
        ) as f:
            data = pickle.load(f, encoding="latin1")
        return data["X"], data["y"], data["groups"], data["act"]

    # Load data
    X, y, groups, activity = pp.preprocessing(CF.dataset, CF)
    activity = activity.flatten()

    # Process data for each group
    unique_groups = np.unique(groups)

    all_data_X, all_data_y, all_data_groups, all_data_activity = [], [], [], []

    # for group in unique_groups:
    for group in tqdm(unique_groups):
        print(f"Processing group S{int(group)}")
        filtered_X, group_labels, group_activity = process_group_data(
            group, X, y, groups, activity, n_epochs=n_epochs
        )

        # Collect processed data for the group
        all_data_X.append(filtered_X)
        all_data_y.append(group_labels)
        all_data_groups.append(groups[groups == group])
        all_data_activity.append(group_activity)

    # Combine all groups' data
    all_data_X = np.concatenate(all_data_X, axis=0)
    all_data_y = np.concatenate(all_data_y, axis=0)
    all_data_groups = np.concatenate(all_data_groups, axis=0)
    all_data_activity = np.concatenate(all_data_activity, axis=0)

    # Save the processed data
    return save_processed_data(
        all_data_X,
        all_data_y,
        all_data_groups,
        all_data_activity,
        CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl",
    )


# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import pickle
# from tqdm import tqdm
# from config import Config
# from preprocessing import preprocess_data as pp
# from training.adaptive_filter_training import train_adaptive_filter
# from utilities import (
#     channel_wise_z_score_denormalization,
#     channel_wise_z_score_normalization,
# )
# import os


# # Constants
# CF = Config()

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


# def filter_activity_data(cur_X, cur_activity, n_epochs):
#     """
#     Filters the data based on activity changes, normalizes it, applies the adaptive filter model, and then denormalizes it.
#     """
#     activity_change_indices = get_activity_change_indices(cur_activity)

#     filtered_Xs = []
#     for start, end in zip(activity_change_indices[:-1], activity_change_indices[1:]):
#         cur_activity_X = cur_X[start:end]

#         # Normalize the data by channel
#         cur_activity_X, ms, stds = normalize_channels(cur_activity_X)

#         # Create, train, and apply the adaptive model
#         model, _ = train_adaptive_filter(
#             cur_activity_X, adaptive_filter_epochs=n_epochs
#         )
#         X_filtered = apply_adaptive_model(model, cur_activity_X)

#         # Denormalize the data
#         X_filtered = denormalize_channels(X_filtered, ms, stds)
#         filtered_Xs.append(X_filtered)

#     return np.concatenate(filtered_Xs, axis=0)


# def get_activity_change_indices(activity):
#     """
#     Returns indices where there is a change in activity (i.e., activity goes from 0 to 1 or vice versa).
#     """
#     change_indices = np.argwhere(np.abs(np.diff(activity)) > 0).flatten() + 1
#     return np.insert(change_indices, [0, len(change_indices)], [0, len(activity)])


# def normalize_channels(data):
#     """
#     Normalizes data by channels (features).
#     """
#     return channel_wise_z_score_normalization(data)


# def denormalize_channels(data, ms, stds):
#     """
#     Denormalizes data by channels (features).
#     """
#     return channel_wise_z_score_denormalization(data, ms, stds)


# def apply_adaptive_model(model, data):
#     """
#     Applies the adaptive model to the data and returns the filtered data.
#     """
#     X_acc = data[:, None, 1:, ...]  # Extract accelerometer data
#     X_ppg = data[:, None, :1, ...]  # Extract PPG data

#     # Perform inference
#     X_filtered = adaptive_model_inference(model, X_acc, X_ppg)

#     return X_filtered.detach().cpu().numpy()  # Move to CPU if necessary


# def adaptive_model_inference(model, X_acc, X_ppg):
#     """
#     Applies the model to accelerometer data and subtracts the filtered accelerometer data from PPG data.
#     """
#     # Send data to GPU if available
#     X_tensor = torch.tensor(X_acc, dtype=torch.float32).to(device)
#     X_ppg_tensor = torch.tensor(X_ppg, dtype=torch.float32).to(device)

#     # Use DataLoader to handle batches
#     dataset = TensorDataset(X_tensor)
#     dataloader = DataLoader(dataset, batch_size=len(X_acc), shuffle=True)

#     for inputs in dataloader:
#         # Apply model to input data
#         X_acc_mixed = model(inputs[0])  # Apply model to input data
#     return X_ppg_tensor - X_acc_mixed


# def process_group_data(group, X, y, groups, activity, n_epochs):
#     """
#     Processes data for a specific group (i.e., filters the data and returns the processed version).
#     """
#     group_data = X[groups == group]
#     group_labels = y[groups == group]
#     group_activity = activity[groups == group]

#     # Filter data based on activity changes
#     filtered_X = filter_activity_data(group_data, group_activity, n_epochs=n_epochs)

#     return filtered_X, group_labels, group_activity


# def save_processed_data(
#     all_data_X, all_data_y, all_data_groups, all_data_activity, file_path
# ):
#     """
#     Saves the processed data to a pickle file.
#     """
#     data = {
#         "X": all_data_X,
#         "y": all_data_y,
#         "groups": all_data_groups,
#         "act": all_data_activity,
#     }
#     with open(file_path, "wb") as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

#     print("Data processing and saving complete.")
#     return all_data_X, all_data_y, all_data_groups, all_data_activity


# def preprocessing(n_epochs):
#     """
#     Main function to load data, process it for each group, and save the results to a file.
#     """
#     if os.path.exists(
#         CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl"
#     ):
#         with open(
#             CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl", "rb"
#         ) as f:
#             data = pickle.load(f, encoding="latin1")
#         return data["X"], data["y"], data["groups"], data["act"]

#     # Load data
#     X, y, groups, activity = pp.preprocessing(CF.dataset, CF)
#     activity = activity.flatten()

#     # Process data for each group
#     unique_groups = np.unique(groups)

#     all_data_X, all_data_y, all_data_groups, all_data_activity = [], [], [], []

#     # Process data for each group
#     for group in tqdm(unique_groups):
#         print(f"Processing group S{int(group)}")
#         filtered_X, group_labels, group_activity = process_group_data(
#             group, X, y, groups, activity, n_epochs=n_epochs
#         )

#         # Collect processed data for the group
#         all_data_X.append(filtered_X)
#         all_data_y.append(group_labels)
#         all_data_groups.append(groups[groups == group])
#         all_data_activity.append(group_activity)

#     # Combine all groups' data
#     all_data_X = np.concatenate(all_data_X, axis=0)
#     all_data_y = np.concatenate(all_data_y, axis=0)
#     all_data_groups = np.concatenate(all_data_groups, axis=0)
#     all_data_activity = np.concatenate(all_data_activity, axis=0)

#     # Save the processed data
#     return save_processed_data(
#         all_data_X,
#         all_data_y,
#         all_data_groups,
#         all_data_activity,
#         CF.path_PPG_Dalia + r"\slimmed_dalia_aligned_prefiltered_80000.pkl",
#     )
