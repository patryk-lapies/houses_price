import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def log_mse(y_true: pd.Series, y_predicted: pd.array) -> float:
    """Function calculates logarithmic mean squared error"""
    return mean_squared_error(np.log(y_true), np.log(y_predicted), squared=False)
