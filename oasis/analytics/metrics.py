from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
import numpy as np
import torch
import os
import logging

bentoml_logger = logging.getLogger("bentoml")

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - smape (float): Symmetric Mean Absolute Percentage Error.
    """
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    bentoml_logger.info(f"model: {label} Size : {size/1e3} (KB)")
    os.remove('temp.p')
    return size/1e6# kb (1e3) to MB (1e6)

def metrics(y_test, y_pred):
    metrics = {}
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
    metrics['mae'] = round(mean_absolute_error(y_test, y_pred), 2)
    metrics['smape'] = round(symmetric_mean_absolute_percentage_error(y_test, y_pred), 2)
    metrics['r2'] = r2_score(y_test, y_pred)
    return metrics

def metrics_pytorch(model=None, y_test=0, y_pred=0):
    metrics = {}
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
    metrics['mae'] = round(mean_absolute_error(y_test, y_pred), 2)
    metrics['smape'] = round(symmetric_mean_absolute_percentage_error(y_test, y_pred), 2)
    metrics['r2'] = r2_score(y_test, y_pred)
    metrics['Model Size (MB)'] = print_size_of_model(model,"int8")
    return metrics
