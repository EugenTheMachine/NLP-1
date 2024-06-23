import pickle

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.base import TextClassifier
from src.classical.feature_generator import TFIDFFeatureGenerator
from src.classical.preprocessor import TextPreprocessor
from src.label_manager import LabelManager


class LogRegTextClassifier(TextClassifier):
    def train(self, data, labels, *args, **kwargs):
        self.label_manager = LabelManager()
        # Change this pipeline with your features.
        # Add proper hyper-parameter tuning
        # Do train-validation split here
        train_data, val_data = train_test_split(data, test_size=.25)
        # Checkout examples on pipelining here: https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
        self.model = Pipeline(
            [
                ("preproc", TextPreprocessor()),
                ("tfidf", TFIDFFeatureGenerator()),
                ("logreg", LogisticRegressionCV(**kwargs)),
            ]
        )
        self.model.fit(data, self.label_manager.fit_transform(labels))

    def predict(self, samples: list[str]):
        return [self.label_manager.classes_[c] for c in self.model.predict(samples)]

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    def load(path):
        return pickle.load(open(path, "rb"))
