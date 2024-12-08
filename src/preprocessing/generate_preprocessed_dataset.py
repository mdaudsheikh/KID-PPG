import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import Config
import pickle
from scipy.io import loadmat
from preprocessing import preprocessing_Dalia_aligned as pp
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.adaptive_linear_model import AdaptiveFilteringModel

# Load configuration
cf = Config(search_type='NAS', root='./data/')

# Constants
N_EPOCHS = 16000
BATCH_SIZE = 256
N_CH = 1
PATIENCE = 150


def channel_wise_z_score_normalization(X):
    """
    Normalize each channel (feature) in X independently (per sample).
    
    The normalization is done by subtracting the mean and dividing by the standard deviation
    for each channel (feature) in the data.
    
    Args:
        X: Input data of shape (samples, channels, time_steps)
        
    Returns:
        X: Normalized input data
        means: Array of shape (samples, channels) containing the means of each channel
        stds: Array of shape (samples, channels) containing the standard deviations of each channel
    """
    # Initialize arrays to store mean and standard deviation for each sample/channel
    means = np.zeros((X.shape[0], X.shape[1]))  # Mean for each channel
    stds = np.zeros((X.shape[0], X.shape[1]))   # Std for each channel
    
    # Iterate over all samples
    for i in range(X.shape[0]):
        sample = X[i, ...]
        
        # Normalize each channel in the current sample
        for j in range(X.shape[1]):  # Iterate over each channel (feature)
            channel_data = sample[j, ...]
            
            # Calculate the mean and standard deviation for the channel
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            
            # Perform Z-score normalization: (data - mean) / std
            sample[j, ...] -= channel_mean
            if channel_std != 0:  # Avoid division by zero
                sample[j, ...] /= channel_std
                
            # Store the mean and std values for later denormalization
            means[i, j] = channel_mean
            stds[i, j] = channel_std
            
        # Save the normalized sample back to X
        X[i, ...] = sample
        
    return X, means, stds


def channel_wise_z_score_denormalization(X, means, stds):
    """
    Reverse the Z-score normalization applied to the data.
    
    This function applies the reverse operation of the normalization:
    (data * std) + mean
    
    Args:
        X: Input data of shape (samples, channels, time_steps) to denormalize
        means: Means for each channel (from normalization)
        stds: Standard deviations for each channel (from normalization)
        
    Returns:
        X: Denormalized input data
    """
    # Iterate over all samples
    for i in range(X.shape[0]):
        sample = X[i, ...]
        
        # Denormalize each channel in the current sample
        for j in range(X.shape[1]):  # Iterate over each channel (feature)
            channel_data = sample[j, ...]
            
            # Reverse Z-score normalization: (data * std) + mean
            if stds[i, j] != 0:  # Avoid multiplication by zero
                channel_data *= stds[i, j]
            
            channel_data += means[i, j]
            sample[j, ...] = channel_data
        
        # Save the denormalized sample back to X
        X[i, ...] = sample
        
    return X


def filter_activity_data(cur_X, cur_activity):
    """
    Filter the data based on changes in activity.
    """
    indexes = np.argwhere(np.abs(np.diff(cur_activity)) > 0).flatten()
    indexes += 1
    indexes = np.insert(indexes, 0, 0)
    indexes = np.insert(indexes, indexes.size, cur_X.shape[0])
    
    filtered_Xs = []
    for i in tqdm(range(indexes.size - 1)):
        cur_activity_X = cur_X[indexes[i]:indexes[i + 1]]
        cur_activity_X, ms, stds = channel_wise_z_score_normalization(cur_activity_X)
        
        # Create and apply the adaptive model
        filtered_X = apply_adaptive_filtering(cur_activity_X)
        
        # Denormalize the data after filtering
        filtered_X = channel_wise_z_score_denormalization(filtered_X, ms, stds)
        filtered_Xs.append(filtered_X)
    
    return np.concatenate(filtered_Xs, axis=0)


def apply_adaptive_filtering(cur_activity_X):
    """
    Apply the adaptive filtering model to the given data.
    """
    # Define and train the model
    model = AdaptiveFilteringModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)
    
    # Assuming the model expects the data in a certain shape
    data_tensor = torch.Tensor(cur_activity_X[..., None])  # Add an extra dimension if necessary
    model_output = model(data_tensor)  # Apply model
    
    return model_output.detach().numpy()  # Get filtered output


def process_group_data(group, X, y, groups, activity):
    """
    Process and filter data for a specific group.
    """
    cur_X = X[groups == group]
    cur_y = y[groups == group]
    cur_activity = activity[groups == group]
    
    # Filter data based on activity changes
    filtered_X = filter_activity_data(cur_X, cur_activity)
    
    return filtered_X, cur_y, groups[groups == group], cur_activity


def save_processed_data(all_data_X, all_data_y, all_data_groups, all_data_activity, file_path):
    """
    Save the processed data to a pickle file.
    """
    data = {
        'X': all_data_X,
        'y': all_data_y,
        'groups': all_data_groups,
        'act': all_data_activity
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    """
    Main function to load data, process it, and save the results.
    """
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    activity = activity.flatten()
    
    # Process data for each group
    unique_groups = np.unique(groups)
    
    all_data_X = []
    all_data_y = []
    all_data_groups = []
    all_data_activity = []
    
    for group in unique_groups:
        print(f"Processing group S{int(group)}")
        filtered_X, cur_y, cur_groups, cur_activity = process_group_data(group, X, y, groups, activity)
        
        # Collect processed data for the group
        all_data_X.append(filtered_X)
        all_data_y.append(cur_y)
        all_data_groups.append(cur_groups)
        all_data_activity.append(cur_activity)
    
    # Combine all groups' data
    all_data_X = np.concatenate(all_data_X, axis=0)
    all_data_y = np.concatenate(all_data_y, axis=0)
    all_data_groups = np.concatenate(all_data_groups, axis=0)
    all_data_activity = np.concatenate(all_data_activity, axis=0)
    
    # Save the processed data
    save_processed_data(all_data_X, all_data_y, all_data_groups, all_data_activity, cf.path_PPG_Dalia + 'slimmed_dalia_aligned_prefiltered_80000.pkl')


if __name__ == "__main__":
    main()
