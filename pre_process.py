import numpy as np
from sklearn.model_selection import train_test_split

def normalize_data(arr):
    """
    Normalises a 1D numpy array to have a mean of 0 and a standard deviation of 1.

    Parameters:
    - arr (np.ndarray): Input array

    Returns:
    - norm_arr (np.ndarray): Normalized array
    - mean (float): Mean of original array
    - std (float): Standard deviation of original array
    """
    mean = np.mean(arr) 
    std = np.std(arr)
    norm_arr = (arr - mean) / std
    return norm_arr 

def split_data(X, y, z, predict_size, val_size, seed):
    """
    Splits data into training, validation, and prediction sets.
    
    Parameters:
    - X (np.ndarray): primary input features (images)
    - y (np.ndarray): target (normalised mass)
    - z (np.ndarray): secondary input (normalised redshift)
    - predict_size (float): fraction of data to hold out for prediction
    - val_size (float): fraction of training data to use for validation
    - seed (int):  random seed for reproducibility

    Returns:
    - Arrays of splitted data.
    """
    # Split off prediction set
    X_temp, X_predict, y_temp, y_predict, z_temp, z_predict = train_test_split(
        X, y, z, test_size=predict_size, random_state=seed
    )

    # Split remaining data into train and validation sets
    X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
        X_temp, y_temp, z_temp, test_size=val_size, random_state=seed
    )

    return (X_train, X_val, X_predict,
            y_train, y_val, y_predict,
            z_train, z_val, z_predict)
