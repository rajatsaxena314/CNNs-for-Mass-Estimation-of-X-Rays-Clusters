import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(train_loss_1, val_loss_1, error, val_error):
    """
    Plots training & validation loss and error over epochs.

    Parameters:
    - history: Keras History object returned by model.fit()
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_1, label='Training Loss')
    plt.plot(val_loss_1, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(error, label='Training')
    plt.plot(val_error, label='Validation')
    plt.title('Error over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_true_vs_predicted(y_true, y_pred, yerr=None):
    """
    Create a scatter plot comparing true vs. predicted values with optional vertical error bars.

    Parameters:
    - y_true (np.ndarray): True target values for the prediction set.
    - y_pred (np.ndarray): Predicted values for the prediction set.
    - yerr (np.ndarray): Vertical error bars for the predicted values.
         
         If None, no error bars are shown. Default is None.

    Notes:
    A dashed red line representing perfect predictions (y = x) is included for reference.
    """
    plt.figure(figsize=(6, 6))
    plt.errorbar(y_true, y_pred, yerr=yerr, fmt='o', 
                 alpha=0.5, ecolor='grey', capsize=3, label=r'$\pm1\sigma$')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             linewidth=2, label='Perfect Prediction (y = x)')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_residuals(true_values, predicted_values):
    """
    Plot residuals (true-predicted) for a set of predictions.

    Parameters:
    - true_values (np.ndarray): Ground truth values.
    - predicted_values (np.ndarray): Model predicted values.
    """
    # Compute residuals
    residuals = np.array(true_values) - np.array(predicted_values)
    
    plt.figure()
    plt.scatter(true_values, residuals, alpha=0.6, edgecolor='grey')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Residual')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.legend()
    plt.show()