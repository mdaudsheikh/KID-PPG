import numpy as np


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
    stds = np.zeros((X.shape[0], X.shape[1]))  # Std for each channel

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
