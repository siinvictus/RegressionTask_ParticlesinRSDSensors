"""
Full pipeline for particle coordinate regression.

"""

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import ModelTrainer
from src.predictor import Predictor


def main():

    # ── 1. LOAD DATA ──────────────────────────────────────────────────────────
    loader = DataLoader(
        dev_path='data/raw/development.csv',
        eval_path='data/raw/evaluation.csv'
    )
    df = loader.load_development()
    df_eval = loader.load_evaluation()

    # ── 2. PREPROCESS ─────────────────────────────────────────────────────────
    preprocessor = Preprocessor()

    # Split features and targets
    X_full, y = preprocessor.split_features_targets(df)

    # Select the top features identified by Random Forest importance in EDA
    X = preprocessor.get_important_features(X_full)
    X_eval = preprocessor.get_important_features(df_eval)

    print(f"Features used: {X.shape[1]} | Targets: {y.shape[1]}")

    # ── 3. TRAIN ──────────────────────────────────────────────────────────────
    trainer = ModelTrainer(model_name='random_forest')
    X_train, X_test, y_train, y_test = trainer.split(X, y)
    trainer.train(X_train, y_train)

    # ── 4. EVALUATE ───────────────────────────────────────────────────────────
    metrics = trainer.evaluate(X_test, y_test)
    print(f"\nFinal metrics: {metrics}")

    # ── 5. SAVE MODEL ─────────────────────────────────────────────────────────
    trainer.save('models/random_forest.pkl')

    # ── 6. PREDICT & SAVE OUTPUT ──────────────────────────────────────────────
    predictor = Predictor()
    predictor.predict_to_csv(trainer.model, X_eval, output_path='outputs/output.csv')
    predictor.plot_hexbin('outputs/output.csv', output_path='outputs/hexbin_eval.png')


if __name__ == '__main__':
    main()