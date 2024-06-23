import pickle
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix


def eval_classifier(y_true, y_pred, lables_str):
    classification_metrics = classification_report(y_true, y_pred, output_dict=True)
    df_cr = pd.DataFrame(classification_metrics)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=lables_str)
    df_cm = pd.DataFrame(cm, columns=lables_str, index=lables_str)
    return df_cr, df_cm


class TextClassifier(ABC):
    """Base class for text classifictation models.

    Attributes:
        label_names (list[str]): list of possible labels.

    Methods:

        train(self, model_name: str, train_data: Any, val_data: Any, **kwargs: Any) -> None:
            Trains a model.

        predict(self, test_data: list[list[str]]) -> list[list[str]]:
            Gets predictions for test data.

        compute_metrics(self, eval_preds: Union[list[np.ndarray], list[list[str]]]) -> dict:
            Calculates a set of metrics for previously predicted results.

    Raises:
        NotImplementedError: If the train method is not implemented by the subclass.
    """

    @abstractmethod
    def train(self, data, labels, *args, **kwargs: Any):
        """Trains the provided model."""
        raise NotImplementedError

    def predict(self, samples: list[str]):
        """Gets predictions for test data."""
        raise NotImplementedError

    def compute_metrics(self, y_true, y_pred, lables_str=None):
        """Calculates a set of metrics for previously predicted results."""
        return eval_classifier(y_true, y_pred, lables_str)

    def save(self, path: str):
        raise NotImplementedError

    @staticmethod
    def load(path: str):
        raise NotImplementedError


class BaseFeatureGenerator(BaseEstimator):
    """Base class to perform text feature engineering"""

    def __init__(self):
        """Class initialization

        Args:
            transformation_path (str): path where the transformation artifact will be saved/loaded
        """
        super().__init__()
        self.transformation = None

    @staticmethod
    def load(path) -> "BaseFeatureGenerator":
        """Load transformation objets has been fitted previously"""
        return pickle.load(open(path, "rb"))

    def save(self, path) -> None:
        """Save transformation object in given path"""
        pickle.dump(self, open(path, "wb"))

    @abstractmethod
    def fit(self, data: pd.Series, labels: pd.Series = None) -> None:
        """Setup self.transformation object with given the data

        Args:
            data (pd.Series): data required to setup the feature extraction transformation
        """
        return self

    @abstractmethod
    def transform(self, data: pd.Series) -> pd.Series:
        """Implement a feature transformation on given data

        Args:
            data (pd.Series): data required to run the feature extraction transformation
        """
        return data

    def fit_transform(self, data: pd.Series, labels: pd.Series = None):
        self.fit(data, labels)
        return self.transform(data)
