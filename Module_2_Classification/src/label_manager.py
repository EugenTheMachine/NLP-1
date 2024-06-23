import pickle
from sklearn.preprocessing import LabelEncoder


class LabelManager(LabelEncoder):
    """LabelEncoder wrapper"""

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path):
        pickle.dump(self, open(path, "wb"))
