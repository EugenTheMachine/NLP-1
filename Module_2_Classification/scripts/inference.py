import json
from sys import path
path.append(".")

import pandas as pd

from src.classical.log_reg_classifier import LogRegTextClassifier
from src.classical.xgb_classifier import XGBTextClassifier
from src.nn.transformer_classifier import TransformerTextClassifier
from src.nn.bert_classifier import BertTextClassifier

import argparse


def inference(config: dict) -> None:
    """This method performs model inference and evaluation

    Args:
        config (dict): inference config file that contains the following fields:
        'data_path': path with .csv file that contains the data for inference
        'column_text': name of text column on inference data
        'column_label': name of label column on inference data
        'artifacts_folder': folder where training artifacts were saved
        'model': model to load 'lr' or 'xgb', 'bert' and 'transformer'

    """
    # Load data
    cols = [config["column_text"], config["column_label"]]
    test_df = pd.read_csv(config["data_path"])[cols]
    test_df = test_df[test_df[cols[1]].isin(["paragraph", "abstract"])]

    if config["model"] == "xgb":
        model_path = f'{config["artifacts_folder"]}/xgb'
        model = XGBTextClassifier.load(model_path)

    if config["model"] == "lr":
        model_path = f'{config["artifacts_folder"]}/lr.pkl'
        model = LogRegTextClassifier.load(model_path)

    if config["model"] == "bert":
        model_path = f'{config["artifacts_folder"]}/bert_checkpoint'
        model = BertTextClassifier.load(model_path)

    if config["model"] == "transformer":
        model_path = f'{config["artifacts_folder"]}/transformer_checkpoint'
        model = TransformerTextClassifier.load(model_path)

    predictions = model.predict(list(test_df[cols[0]]))

    report, matrix = model.compute_metrics(predictions, test_df[cols[1]])
    print(report)
    print(matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model of choice")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file in json format",
        default="scripts/config_inference.json",
    )
    args = parser.parse_args()
    with open(args.config) as conf_file:
        config = json.load(conf_file)

    inference(config)
