import pickle
import re
import string
from abc import ABC, abstractmethod
from typing import Iterable

import contractions
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode
import contractions

from src.classical.stopwords import stop_words_list


class TextProcessor(ABC):
    def __init__(self, X: pd.Series):
        self.X = X.copy()

    @abstractmethod
    def parse(self) -> Iterable[str]:
        """Main method that execute all the transformations on the class following the rigth order

        Returns:
            Iterable[str]: iterable like pd.Series that contains the processed text strings
        """
        pass


class TextCleaning(TextProcessor):
    """
    Class to implement text cleaning transformations
    """

    def __init__(self, X: pd.Series):
        super().__init__(X)
        self.re_sep_digit = re.compile(r"([\d]+)([a-zA-Z]+)")

    def separate_num_from_text(self: TextProcessor) -> TextProcessor:
        """Split numerical characters has been concatenated with text wrongly
        for instance 3months --> 3 months
        """
        self.X = self.X.apply(lambda x: self.re_sep_digit.sub(r"\1 \2", x))
        return self

    def expand_contractions(self: TextProcessor) -> TextProcessor:
        """Expand contractions to reduce amount of unique tokens and standatize words
        for instance text you're -> you are
        Check the library:
        https://github.com/kootenpv/contractions/tree/master
        https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
        """
        # expand contractions in texts in self.X
        # <your code here>
        self.X = self.X.apply(lambda x: contractions.fix(x))
        return self

    def replace_diacritics(self: TextProcessor) -> TextProcessor:
        """Parse non-ASCII characters, and returns a string that can be safely encoded to ASCII.
        For instance  unidecode(u5317, u4EB0) -> 'Bei Jing '
        Diacritics are replaced with nearest characters. If it cannot find compatible ones,
        a space is put in its place by default
        """
        # apply unidecode to the texts
        # <your code here>
        self.X = pd.Series(self.X)
        self.X = self.X.apply(unidecode)
        return self

    def remove_punctuations(self: TextProcessor) -> TextProcessor:
        """Remove not relevant symbols on the text:
        This transformation could be implemented in several ways:
        https://likegeeks.com/python-remove-punctuation/
        """
        # remove punctuation
        # <your code here>
        translator = str.maketrans('', '', string.punctuation)
        self.X = self.X.apply(lambda x: x.translate(translator))
        return self

    def remove_double_spaces(self: TextProcessor) -> TextProcessor:
        """Remove double spaces on text, transforming text like
        'Hello   word' -> 'Hello word'. Check the implementation on:
        https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
        """
        # <your code here>
        self.X = self.X.apply(lambda x: re.sub(' +', ' ', x))
        return self

    def parse(self: TextProcessor) -> pd.Series:
        """Call all the implemented methods to return the clean text
        take a look the implementation on:
        https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
        You can call the methods as follow self.method1().method2().method3() ...
        """
        self = (
            self.replace_diacritics()
            .separate_num_from_text()
            .expand_contractions()
            .remove_double_spaces()
            .remove_punctuations()
        )

        return self.X


class TextNomalization(TextProcessor):
    """Class to that implements text normalization methods"""

    def __init__(self, X: pd.Series):
        super().__init__(X)
        self.download_packages()
        self.stop_words_list = stop_words_list
        # self.stop_words_list = ['hi', 'hello', 'bye']

    @staticmethod
    def download_packages() -> None:
        """Download NLTK packages if not avaible in local machine"""
        try:
            nltk.data.find("tokenizers/punkt")
        except:
            nltk.download("punkt")

    def stemming_word(self: TextProcessor) -> TextProcessor:
        """Implement word stemming to reduce word variations. For example, the words 'programming',
        'programmer', and 'programs' can all be reduced down to the common word stem 'program'.
        Is recommended to execute tokenization first as stemming works at token level
        Check the implementation on:
        https://medium.com/mlearning-ai/nlp-tokenization-stemming-lemmatization-and-part-of-speech-tagging-9088ac068768
        """
        # apply PorterStemmer to the texts stored in self.X
        # <your code here>
        ps = PorterStemmer()
        self.X = self.X.apply(lambda x: ps.stem(x))
        return self

    def stop_word_removal(self: TextProcessor) -> TextProcessor:
        """Given a list of stop words ''stop_word_list', delete them from the text on self.X.
        Is recommended to execute tokenization first as stop word removal works at token level
        Feel free to add in that list really frequent and low frequent words. Also check
        text encoded issues (like uf8f4) and special characters added on the documents (like ##LTLine##).
        Text tokexns with non relevan words like [Hi, I'm, your, assistant] will be transformed on [I, your, assistant]
        Check https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
        """
        # remove stop words from self.X using self.stop_words_list
        # <your code here>
        self.X = self.X.apply(lambda x: " ".join([word for word in str(x).split() if word not in self.stop_words_list]))
        return self

    def to_lower(self: TextProcessor) -> TextProcessor:
        """Transform text variations like capital case or camel case into lower case
        for instance text 'Hello World' -> 'hello world'.
        """
        # lowercase texts in self.X
        # <your code here>
        self.X = self.X.apply(lambda x: x.lower())
        return self

    def tokenize_words(self: TextProcessor) -> TextProcessor:
        """Split text into a list of tokens Implement rules to tokenize string text into word tokens. Is recommended to use specific
        libraries like NLTK instad simple text.split() in order to recognize complex patterns like
        'Good muffins cost $3.88\nin New York' -> ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York']
        Check: https://www.nltk.org/api/nltk.tokenize.html
        """
        # tokenize texts in self.X with nltk
        # <your code here>
        self.X = self.X.apply(lambda x: nltk.word_tokenize(x))
        return self

    def join_tokens(self: TextProcessor) -> TextProcessor:
        """Some methods on this class process the text at token level [word1, word2 .... wordn].
        This method will join these words to get back the string version -> 'word1 word2 ...wordn' as
        Feature extraction methods like tfidf works by default with string texts.
        """
        # join processed texts back into space separated strings
        # <your code here>
        self.X = self.X.apply(lambda x: " ".join(x))
        return self

    def parse(self: TextProcessor) -> pd.Series:
        """Call all the implemented methods to return the normalized text
        You can call the methods as follows: self.method1().method2().method3()
        Take into consideration that tokenization should be the last step. The order matters:
        check https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
        This module should return string text, so is recommended to reverse the tokenization using join tokens
        """
        self = (
            self.to_lower()
            .tokenize_words()
            .join_tokens()
            .stemming_word()
            .stop_word_removal()
        )
        return self.X


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, transformation_path: str = None) -> None:
        super().__init__()
        self.transformation_path = transformation_path

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        """Implements text transformation from TextCleaning and TextNormalization modules"""
        X_ = X.copy()
        text_clean = TextCleaning(X_)
        X_ = text_clean.parse()
        text_normalizer = TextNomalization(X_)
        X_ = text_normalizer.parse()
        return X_

    @classmethod
    def load(cls, transformation_path):
        with open(transformation_path, "rb") as f:
            return pickle.load(f)

    def save(self):
        pickle.dump(self, open(self.transformation_path, "wb"))


if __name__ == "__main__":
    texts = [
        "Hello word this is a text phrases 4$. You're welcome \u5317\u4EB0",
        "Here we have the second sentence to test. I'm 5year old",
    ]
    df = pd.DataFrame({"texts": texts, "text_id": list(range(len(texts)))})
#     print(df.head())

    # Call text clean
    t_c = TextCleaning(df["texts"])
    df["texts_clean"] = t_c.parse()

    # Call text normalizer
    t_n = TextNomalization(df["texts_clean"])
    df["texts_normal"] = t_n.parse()

    # Call feature processor
    tp = TextPreprocessor()
    text_pre_processesed = tp.transform(df["texts"])