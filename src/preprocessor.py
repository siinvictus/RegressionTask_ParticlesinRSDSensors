import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    """Handles feature/target splitting, scaling, and PCA."""

    # extracted from experiments done in the jupyter notebook 
    # based on: model_rf1.feature_importances_
    MOST_IMPORTANT_FEATURES = [
        'pmax[8]', 'pmax[11]', 'pmax[10]', 'pmax[5]', 'pmax[13]', 'pmax[9]',
        'negpmax[13]', 'negpmax[11]', 'negpmax[3]', 'negpmax[10]', 'pmax[4]',
        'negpmax[8]', 'negpmax[1]', 'pmax[3]', 'negpmax[5]', 'negpmax[6]',
        'negpmax[14]', 'pmax[2]', 'negpmax[4]', 'pmax[1]', 'negpmax[9]',
        'pmax[6]', 'pmax[14]', 'negpmax[2]', 'pmax[15]', 'area[5]', 'area[3]',
        'area[13]', 'area[10]', 'area[11]', 'area[14]', 'area[15]', 'area[8]',
        'area[6]', 'area[4]', 'area[1]', 'area[9]', 'area[2]'
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def split_features_targets(self, df: pd.DataFrame):
        """Split dataframe into features X and targets y (x, y coordinates)."""
        X = df.drop(['x', 'y'], axis=1)
        y = df[['x', 'y']].copy()
        return X, y

    def get_important_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the top features identified by RandomForest importance."""
        available = [f for f in self.MOST_IMPORTANT_FEATURES if f in df.columns]
        return df[available]

    # useful for future exp, not needed for the best model RF
    def fit_transform_pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Fit PCA and return the first n_components."""
        self.pca = PCA(n_components=X.shape[1])
        self.pca.fit(X)
        X_pca = self.pca.transform(X)
        return X_pca[:, :n_components]
    
    # useful for future exp, not needed for the best model RF
    def transform_pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Transform new data with already fitted PCA."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted yet. Call fit_transform_pca first.")
        X_pca = self.pca.transform(X)
        return X_pca[:, :n_components]
    
    # useful for future exp, not needed for the best model RF
    def fit_transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit StandardScaler and transform."""
        return self.scaler.fit_transform(X)
    
    # useful for future exp, not needed for the best model RF
    def transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Transform new data with already fitted scaler."""
        return self.scaler.transform(X)
    
    # useful for future exp, not needed for the best model RF
    def explained_variance_cumulative(self) -> list:
        """Return cumulative explained variance ratio from fitted PCA."""
        if self.pca is None:
            raise RuntimeError("PCA not fitted yet.")
        cum_var = []
        var_exp = 0
        for v in self.pca.explained_variance_ratio_:
            var_exp += v
            cum_var.append(var_exp)
        return cum_var