import pandas as pd


class DataLoader:
    """Loads and cleans the RSD sensor datasets."""

    # identified through the visualizations in the notebook, through the histograms
    OUTLIER_SENSOR_IDS = [0, 7, 12, 15, 16, 17]
    SIGNAL_FEATURES = ['pmax', 'negpmax', 'area', 'tmax', 'rms']

    def __init__(self, dev_path: str, eval_path: str):
        self.dev_path = dev_path
        self.eval_path = eval_path

    def load_development(self) -> pd.DataFrame:
        df = pd.read_csv(self.dev_path)
        print(f"Development set loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_evaluation(self) -> pd.DataFrame:
        df = pd.read_csv(self.eval_path)
        df = df.drop('Id', axis=1)
        print(f"Evaluation set loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def get_outlier_columns(self, df: pd.DataFrame) -> list:
        columns_to_remove = [
            f"{feature}[{sensor_id}]"
            for feature in self.SIGNAL_FEATURES
            for sensor_id in self.OUTLIER_SENSOR_IDS
            if f"{feature}[{sensor_id}]" in df.columns
        ]
        return columns_to_remove

    def remove_outlier_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_remove = self.get_outlier_columns(df)
        df_clean = df.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} outlier columns. "
              f"Remaining: {df_clean.shape[1]} columns")
        return df_clean