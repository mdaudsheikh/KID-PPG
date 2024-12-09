import pickle
import numpy as np
from skimage.util.shape import view_as_windows
import random
import os
import config as cf

FS_PPG_DALIA = 32
FS_ACTIVITY = 4


def preprocessing(dataset, cf=cf):

    # Initialize dictionaries to store signals, activities, and labels
    S, ppg, acc, activity, ground_truth = {}, {}, {}, {}, {}

    # Randomize session order for reproducibility
    random.seed(20)

    # If the preprocessed data doesn't exist, process the dataset
    if not os.path.exists(cf.path_PPG_Dalia + "slimmed_dalia_aligned.pkl"):
        session_list = random.sample(range(1, 16), 15)

        # Step 1: Process each session data
        for session in session_list:
            paz = session
            # Load session data from pickle file
            session_data = load_session_data(cf, paz)

            # Step 2: Process and store individual signals (PPG and acceleration)
            ppg[paz], acc[paz] = process_signals(session_data)

            # Step 3: Process and store activity and ground truth labels
            activity[paz], ground_truth[paz] = process_labels(session_data)

        # Step 4: Create time windows for signals, activities, and labels
        sig_list, act_list, ground_truth_list, groups = [], [], [], []

        # Step 5: Loop over each session and process their data into windows
        for paz in ground_truth:
            sig_data, act_data = create_windows(ppg[paz], acc[paz], activity[paz], cf)
            sig_list.append(sig_data)
            act_list.append(act_data)
            ground_truth_list.append(ground_truth[paz].reshape(-1, 1))
            groups.append(np.full(sig_data.shape[0], paz))

        # Step 6: Stack the processed data
        X, y, groups, act = stack_data(sig_list, ground_truth_list, groups, act_list)

        # Step 7: Save the processed data to a pickle file
        save_processed_data(cf, X, y, groups, act)

    else:
        # Load preprocessed data if the pickle file already exists
        X, y, groups, act = load_processed_data(cf)

    # Output the shape of the data for debugging
    print(
        f"Training data shape: {X.shape}, Test data shape: {y.shape}, Groups shape: {groups.shape}, Activity shape: {act.shape}"
    )

    return X, y, groups, act


# Step-by-step functions for clarity


def load_session_data(cf, session_id):
    """
    Loads session data from a pickle file.
    """
    with open(
        cf.path_PPG_Dalia + rf"\PPG_FieldStudy\S{session_id}\S{session_id}.pkl", "rb"
    ) as f:
        return pickle.load(f, encoding="latin1")


def process_signals(session_data):
    """
    Processes PPG and acceleration signals, downsamples, and trims the data.
    """
    ppg = session_data["signal"]["wrist"]["BVP"][::2][
        38:
    ]  # Downsample and trim PPG data
    acc = session_data["signal"]["wrist"]["ACC"][:-38]  # Trim acceleration data
    return ppg, acc


def process_labels(session_data):
    """
    Processes activity and ground truth labels, trimming the ground truth data.
    """
    activity = session_data["activity"]
    ground_truth = session_data["label"][:-1]  # Trim ground truth data
    return activity, ground_truth


def create_windows(ppg, acc, activity, cf):
    """
    Creates time windows for PPG, acceleration, and activity data.
    """
    # Step 1: Combine PPG and acceleration signals
    ppg_acc_data = np.concatenate((ppg, acc), axis=1)

    # Step 2: Define window shape and stride for sliding windows
    window_shape = (
        cf.time_window * FS_PPG_DALIA,
        4,
    )  # Example window: (time_window_samples, 4 features)
    stride = FS_PPG_DALIA * 2  # 32 samples with overlap

    # Step 3: Create sliding windows for the concatenated signals
    windows = view_as_windows(ppg_acc_data, window_shape, stride)

    # Step 4: Extract the first slice from the windows. For some reason another axis is in the moddle?
    windows_selected = windows[:, 0, :, :]

    # Step 5: Reorganize the axes for the resulting data.
    # Basically making it long. This is equivelant to throwing
    # the length of each window dim to the last axis ->
    # (4093 * 15 participants, 256 data points in 8 seconds, 4 columns)
    # to becoming (4093 * 15 participants, 4 columns, 256 data points in 8 seconds)
    sig_data = np.moveaxis(windows_selected, 1, 2)  # Reorganize axes (features first)

    # Step 6: Create sliding windows for activity data. This function was getting long
    activity_windows = create_activity_windows(activity, cf)

    return sig_data, activity_windows


def create_activity_windows(activity, cf):
    """
    Creates sliding windows for the activity data and reshapes it for use in model training.

    Parameters:
    - activity: 1D numpy array of activity labels (e.g., walking, running)
    - cf: Configuration object containing time_window and other parameters

    Returns:
    - activity_windows: 2D numpy array containing windowed activity labels
    """

    # Step 1: Define the window shape and stride for sliding windows
    window_shape = (
        cf.time_window * FS_ACTIVITY,
        1,
    )  # 8 * 4 being 32 mean the window size is (32, 1)
    stride = FS_ACTIVITY * 2  # Mean the window will overlap by 24 data points

    # Create the windows
    activity_windows_raw = view_as_windows(activity, window_shape, stride)

    # Annoying but returns 4D so eleminating an unneccessary axis
    activity_windows_selected = activity_windows_raw[:, 0, :, :]

    # moving window size to the last axis
    activity_windows_reshaped = np.moveaxis(activity_windows_selected, 1, 2)

    # Eliminating another axis of 1 so now the activity window should be 4093 windows for S1 recording
    # and a size of the window is 32 as defined above
    # activity_windows_final = activity_windows_reshaped[:, 0, :]
    activity_windows_final = activity_windows_reshaped[:-1, :, 0]

    return activity_windows_final


def stack_data(sig_list, ground_truth_list, groups, act_list):
    """
    Stacks the processed data for features, labels, groups, and activities.
    """
    X = np.vstack(sig_list)  # Stack features (PPG + ACC)
    y = np.vstack(ground_truth_list)  # Stack ground truth labels
    groups = np.hstack(groups)  # Stack group labels
    act = np.vstack(act_list)  # Stack activity data
    return X, y, groups, act


def save_processed_data(cf, X, y, groups, act):
    """
    Saves the processed data to a pickle file.
    """
    data = {"X": X, "y": y, "groups": groups, "act": act}
    with open(cf.path_PPG_Dalia + r"\slimmed_dalia_aligned.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_processed_data(cf):
    """
    Loads the preprocessed data from a pickle file.
    """
    with open(cf.path_PPG_Dalia + r"\slimmed_dalia_aligned.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data["X"], data["y"], data["groups"], data["act"]
