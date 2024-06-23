import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from src.base import TextClassifier
from src.classical.feature_generator import TFIDFFeatureGenerator
from src.classical.preprocessor import TextPreprocessor
from src.label_manager import LabelManager


class XGBTextClassifier(TextClassifier):
    def train(self, data, labels: list[str], *args, **kwargs) -> None:
        """Perform model training of XGBoost model using the native API. Check the demo
        code here: https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html
        and official documentation (check .train() method) here:
        https://xgboost.readthedocs.io/en/stable/python/python_api.html. You need to provide to
        the train method the parameters and data into the right matrix format.

        Args:
            data (pd.Series): Text features
            labels (list[str]): Text class
        """
        xgb.set_config(verbosity=2)
        self.label_manager = LabelManager()
        self.preprocessor = Pipeline(
            [("preproc", TextPreprocessor()), ("tfidf", TFIDFFeatureGenerator())]
        )

        # Preprocess the data
        data_processed = self.preprocessor.fit_transform(data)
        labels_encoded = self.label_manager.fit_transform(labels)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            data_processed, labels_encoded, test_size=0.2, random_state=42
        )

        # Convert to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [3, 4, 5],
            'eta': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'objective': ['multi:softmax'],
            'num_class': [len(self.label_manager.classes_)]
        }

        # Hyperparameter tuning using GridSearchCV
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3)
        grid_search.fit(X_train, y_train)

        # Train the best model with early stopping
        best_params = grid_search.best_params_
        self.model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50
        )

        print("Training is done")

    def save(self, path: str):
        """Save the model. You can save it using the native .save_model or pickle format.
        Make sure to save all the necessary preprocessors in order to load them for inference.
        Take a look at .save_model and pickle model saving here:
        -> https://xgboost.readthedocs.io/en/stable/python/python_api.html
        -> https://pythonbasics.org/pickle/
        """

        os.makedirs(Path(path), exist_ok=True)
        
        # Save the model
        self.model.save_model(Path(path) / "xgb_model.json")
        
        # Save the preprocessors and label manager using pickle
        with open(Path(path) / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)
        with open(Path(path) / "label_manager.pkl", "wb") as f:
            pickle.dump(self.label_manager, f)

        print(f"Model artifacts saved to {path}")

    @staticmethod
    def load(path) -> 'XGBTextClassifier':
        """
        Load the XGB model on class attribute self.model. The loading should be consistent according to
        the save method implemented on the class TrainXGBoost, loading pickle of native json file. Take a look at:
        -> https://xgboost.readthedocs.io/en/stable/python/python_api.html (model.load_model method)
        -> https://pythonbasics.org/pickle/
        """
        classifier = XGBTextClassifier()
        
        # Load the model
        classifier.model = xgb.Booster()
        classifier.model.load_model(Path(path) / "xgb_model.json")
        
        # Load the preprocessors and label manager using pickle
        with open(Path(path) / "preprocessor.pkl", "rb") as f:
            classifier.preprocessor = pickle.load(f)
        with open(Path(path) / "label_manager.pkl", "rb") as f:
            classifier.label_manager = pickle.load(f)
        
        return classifier

    def predict(self, samples: Union[pd.Series, np.array]) -> pd.Series:
        """Given a trained model self.model perform inference on feature data. Take into consideration
        that XGB use data on DMatrix format. Check the examples here (check booster.predict method):
        --> https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict
        --> https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html
        Args:
            samples (Union[pd.Series, np.array]): Text feature vectors

        Returns:
            pd.Series: Inference with class label for each doc
        """
        # Preprocess the samples
        samples_processed = self.preprocessor.transform(samples)

        # Convert to DMatrix format
        dtest = xgb.DMatrix(samples_processed)

        # Predict with the model
        predictions = self.model.predict(dtest)

        # Map labels to original texts
        # print(type(predictions))
        predictions_labels = self.label_manager.inverse_transform(predictions.astype(int))

        return pd.Series(predictions_labels)
