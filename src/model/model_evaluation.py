import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub
import os
from src.logger import logging

# -------------------------------------------------------------------------------------
# DagsHub + MLflow setup (KEEP mlflow=True)
# -------------------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "https://dagshub.com/babatundejulius911/end-to-end-Nlp-Project.mlflow"

dagshub.init(
    repo_owner="babatundejulius911",
    repo_name="end-to-end-Nlp-Project",
    mlflow=True
)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logging.error('Error loading model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        logging.info('Model evaluation metrics calculated')
        return metrics
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str):
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving metrics: %s', e)
        raise


def save_model_info(model_uri: str, file_path: str):
    """Save ONLY model_uri (important for mlflow=True setup)"""
    try:
        model_info = {'model_uri': model_uri}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving model info: %s', e)
        raise


def main():
    mlflow.set_experiment("Dvc-NLP-pipeline")

    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            # -----------------------------
            # Log metrics
            # -----------------------------
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # -----------------------------
            # Log parameters
            # -----------------------------
            if hasattr(clf, 'get_params'):
                for name, value in clf.get_params().items():
                    mlflow.log_param(name, value)

            # -----------------------------
            # Log model (DagsHub style)
            # -----------------------------
            signature = infer_signature(X_test, clf.predict(X_test))

            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                signature=signature,
                input_example=X_test[:5]
            )

            # ✅ CRITICAL: Extract correct URI
            model_uri = model_info.model_uri
            print(f"MODEL URI: {model_uri}")

            # Save correct model info
            save_model_info(model_uri, 'reports/experiment_info.json')

            # Log metrics file
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed pipeline: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()