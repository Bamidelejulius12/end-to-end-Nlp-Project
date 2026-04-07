# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from mlflow.tracking import MlflowClient


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # CI/CD SAFE AUTH (using your CAPSTONE_TEST secret)
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "babatundejulius911"
        repo_name = "end-to-end-Nlp-Project"
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        client = MlflowClient()
        cls.model_name = "my_model"

        # ROBUST MODEL VERSION FETCHING
        # Priority: Staging → Production → Latest
        versions = list(client.search_model_versions(f"name='{cls.model_name}'"))

        if not versions:
            raise Exception(f"No versions found for model '{cls.model_name}'")

        staging_versions = [v for v in versions if v.current_stage == "Staging"]
        production_versions = [v for v in versions if v.current_stage == "Production"]

        if staging_versions:
            selected_version = max(staging_versions, key=lambda v: int(v.version))
            print(f"Using STAGING model version: {selected_version.version}")

        elif production_versions:
            selected_version = max(production_versions, key=lambda v: int(v.version))
            print(f"Using PRODUCTION model version: {selected_version.version}")

        else:
            selected_version = max(versions, key=lambda v: int(v.version))
            print(f"Using LATEST model version: {selected_version.version}")

        cls.model_version = selected_version.version
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"

        print(f"Loading model from: {cls.model_uri}")

        cls.new_model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')
    # TEST

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        # Predict
        prediction = self.new_model.predict(input_df)

        # Validate input shape
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        # Validate output
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        # Features and labels
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict
        y_pred_new = self.new_model.predict(X_holdout)

        # Metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assertions
        self.assertGreaterEqual(accuracy_new, expected_accuracy,
                                f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision,
                                f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall,
                                f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1,
                                f'F1 score should be at least {expected_f1}')


if __name__ == "__main__":
    unittest.main()