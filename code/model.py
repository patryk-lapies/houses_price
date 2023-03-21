import pandas as pd
import numpy as np
from metrics import log_mse
from xgboost import XGBRFRegressor


class Model:

    def __init__(self):
        self.model = XGBRFRegressor(eta=0.2, max_depth=15, n_estimators=110, random_state=42)

    def fit(self, x_train_data: pd.DataFrame, y_train_data: pd.DataFrame):
        """Training model with given training data"""
        self.model.fit(x_train_data, y_train_data)

    def predict(self, x_test_data: pd.DataFrame) -> np.array:
        """Making predictions by model on given data and returning them"""
        predictions = self.model.predict(x_test_data)
        return predictions

    @staticmethod
    def evaluate(true_values: pd.Series, predictions: np.array) -> float:
        return log_mse(true_values, predictions)
