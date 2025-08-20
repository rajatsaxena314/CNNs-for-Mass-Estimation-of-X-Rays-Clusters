import pickle
import numpy as np

def load_images(data_directory, image_file, img_names):
    """
    Loads images from a pickle file and returns the image array.

    Parameters:
    - data_directory (str): Path to the directory containing the pickle file.
    - image_file (str): Name of the pickle file. 

    Returns:
    - img (np.ndarray): Numpy array of 3D images.
    """
    file_path = data_directory + image_file

    #Opens the objects in the location and loads the Object from the pickle file.
    with open(file_path, 'rb') as file: #rb --> reading in binary mode; used for non-text data like images.
        data = pickle.load(file)
    
    img = np.array(data[img_names])
    return img