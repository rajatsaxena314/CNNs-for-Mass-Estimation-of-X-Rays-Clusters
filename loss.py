import tensorflow as tf

def heteroscedastic_loss(y_true, y_pred):
    """
    Computes the heteroscedastic loss for regression tasks. This loss function utilises 
    the model's predictions for the mean (`mu`) and the log-variance (`log_var`) of a Gaussian 
    distribution for each output. 
    
    Parameters:
    - y_true (tf.Tensor): A tensor of true target values, shape (batch_size,).
    - y_pred (tf.Tensor): A tensor of predicted values, shape (batch_size, 2), where:
            - y_pred[:, 0] is the predicted mean (mu)
            - y_pred[:, 1] is the predicted log-variance (log_var)

    Returns:
    - A scalar tensor representing the mean heteroscedastic loss over the batch.
    """
    mu = y_pred[:, 0]
    log_var = y_pred[:, 1]
    precision = tf.exp(-log_var)  # = 1 / sigma^2

    return tf.reduce_mean(0.5 * log_var + 0.5 * precision * tf.square(y_true[:] - mu))
