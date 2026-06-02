import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class ModelTrainer:
    """Trains, evaluates, and saves regression models for particle coordinate prediction."""

    def __init__(self, model_name: str = 'random_forest', random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self._build_model()

    def _build_model(self):
        """Instantiate the model based on model_name."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=150, random_state=self.random_state
            ),
            'linear': MultiOutputRegressor(LinearRegression()),
            'ridge': MultiOutputRegressor(
                Ridge(alpha=0.01, max_iter=1000)
            ),
            'lasso': MultiOutputRegressor(
                Lasso(alpha=0.01, max_iter=1000)
            ),
        }
        if self.model_name not in models:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Choose from: {list(models.keys())}"
            )
        self.model = models[self.model_name]

    def split(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
        """Train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Fit the model."""
        print(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate on test set and return metrics."""
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        metrics = {'r2': round(r2, 4), 'rmse': round(rmse, 4)}
        print(f"R²: {metrics['r2']}  |  RMSE: {metrics['rmse']}")
        return metrics

    def predict(self, X) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def save(self, path: str):
        """Save model to disk as a .pkl file."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load a previously saved model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")