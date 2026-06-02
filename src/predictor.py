import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Predictor:
    """Generates predictions on new data and saves output files."""

    def predict_to_csv(self, model, X: pd.DataFrame, output_path: str = 'outputs/output.csv'):
        """
        Run model predictions and save to CSV in the required submission format.
        Output format: Id, Predicted (as 'x|y')
        """
        predictions = model.predict(X)
        df = pd.DataFrame(predictions, columns=['x', 'y'])
        df['Predicted'] = df['x'].astype(str) + '|' + df['y'].astype(str)
        df = df.drop(['x', 'y'], axis=1)
        df.insert(0, 'Id', range(0, len(df)))
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path} ({len(df)} rows)")
        return df

    def plot_hexbin(self, predictions_path: str, output_path: str = 'outputs/hexbin_eval.png'):
        """
        Load predictions CSV and plot a hexbin of predicted x, y coordinates.
        Saves the plot to disk.
        """
        pred_data = pd.read_csv(predictions_path)
        pred_data[['x', 'y']] = (
            pred_data['Predicted']
            .str.replace("'", "")
            .str.split('|', expand=True)
        )
        pred_data[['x', 'y']] = pred_data[['x', 'y']].apply(pd.to_numeric)

        plt.hexbin(pred_data['x'], pred_data['y'], gridsize=50, cmap='inferno')
        plt.colorbar(label='count (x, y) in bin')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Predicted Particle Coordinates')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()
        print(f"Hexbin plot saved to {output_path}")