import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

def get_data(image_p, catalog_f):
    '''
    Given the image and catalogue path, this function returns the
    images and labels needed to start directly with model training.
    train_labels: Can be used to filter a particular mass range
    Returns:
    images: [index, energy-band-image]
    labels_z: [log10(mass), redshift, index]
    '''
    is_efedssim=True
    is_efedsobs=False
    images = pd.read_pickle(image_p)
    key_mass = 'm500_wl_final' if is_efedsobs else 'M500c' if is_efedssim else 'HALO_M500c'
    key_redshift = 'z_final' if is_efedsobs else 'z' if is_efedssim else 'redshift_R'
    catalog_df = pd.read_feather(catalog_f) 
    labels = np.log10(catalog_df[key_mass])  #The log of mass is calculated and is given to the user
    redshifts = catalog_df[key_redshift].values
    indices = catalog_df.index.values
    labels_z = np.transpose([labels, redshifts, indices])
    return images, labels_z

def plot_cluster_histograms(log_mass, redshift, bin_num):
    """
    Plots histograms of cluster masses and redshifts.

    Parameters:
    - log_mass (np.ndarray): log-scaled cluster masses
    - redshift(np.ndarray): array-like, cluster redshifts
    - bin_num (int): number of bins

    Returns:
    - Histograms of cluster masses and redshifts
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # First subplot: Histogram of Cluster Masses
    axes[0].hist(log_mass, bins=bin_num, edgecolor='black')
    axes[0].set_xlabel('Mass [$M_\\odot$]')  #M_\\odot represents the solar mass
    axes[0].set_ylabel('Number of Clusters')
    axes[0].set_title('Histogram of Cluster Masses')
    axes[0].grid(True, linestyle='--') 

    # Second subplot: Histogram of Redshift
    axes[1].hist(redshift, bins=bin_num, color='red', edgecolor='black')
    axes[1].set_xlabel('Redshift')
    axes[1].set_ylabel('Number of Clusters')
    axes[1].set_title('Histogram of Redshift')
    axes[1].grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()
