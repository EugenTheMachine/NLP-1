import sys

sys.path.insert(0, "..")

import pandas as pd

from src.nn.transformer_classifier import TransformerTextClassifier  # noqa

max_seq_length = 512

data_files = {
    "train": "../dataset_doc/text_classification_train.csv",
    "test": "../dataset_doc/text_classification_test.csv",
}


def test_transformer_training():
    model = TransformerTextClassifier(
        max_seq_length, label_names=["abstract", "paragraph"]
    )

    dataset_df = pd.read_csv(data_files["train"])
    dataset_df = dataset_df[dataset_df["label"].isin(["abstract", "paragraph"])].sample(
        64, random_state=42
    )

    model.train(dataset_df["text"], dataset_df["label"])


def test_transformer_inference():
    model = TransformerTextClassifier(
        max_seq_length, label_names=["abstract", "paragraph"]
    )

    model.predict(["this is a paper about some fancy thing"])
    model.predict(["this is a paper about some fancy thing", "this is just a sentence"])


def test_transformer_metrics():
    model = TransformerTextClassifier(
        max_seq_length, label_names=["abstract", "paragraph"]
    )

    train_df = pd.read_csv(data_files["train"])
    train_df = train_df[train_df["label"].isin(["abstract", "paragraph"])]

    test_df = pd.read_csv(data_files["train"])
    test_df = test_df[test_df["label"].isin(["abstract", "paragraph"])].sample(
        100, random_state=42
    )
    model.train(train_df["text"], train_df["label"], num_epochs=2)

    preds = model.predict(test_df["text"])
    metrics, matrix = model.compute_metrics(preds, test_df["label"])
    assert metrics.loc["precision", "abstract"] > 0.0
    assert metrics.loc["recall", "abstract"] > 0.0
    assert metrics.loc["f1-score", "abstract"] > 0.0
