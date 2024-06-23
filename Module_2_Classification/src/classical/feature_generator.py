import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.base import BaseFeatureGenerator


class TFIDFFeatureGenerator(BaseFeatureGenerator):
    """TfidfVectorizer wrapper"""

    def __init__(self, *tfidf_args, **tfidf_kwargs):
        self._tfidf = TfidfVectorizer(*tfidf_args, **tfidf_kwargs)

    def fit(self, data, labels):
        self._tfidf.fit(data, labels)
        return self

    def transform(self, data: pd.Series) -> pd.Series:
        return self._tfidf.transform(data)

    def fit_transform(self, data: pd.Series, labels: pd.Series = None):
        return self._tfidf.fit_transform(data, labels)

    @property
    def tfidf(self) -> TfidfVectorizer:
        return self._tfidf


# Implement your own feature extractors here inherited from BaseFeatureGenerator
