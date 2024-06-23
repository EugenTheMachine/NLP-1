import json
from sys import path
path.append(".")

import pandas as pd

from src.classical.log_reg_classifier import LogRegTextClassifier
from src.classical.xgb_classifier import XGBTextClassifier
from src.nn.transformer_classifier import TransformerTextClassifier
from src.nn.bert_classifier import BertTextClassifier


import argparse


def train(config: dict):
    """Main method that describes the training pipeline.

    Args:
        config (dict): Train configuration file that include the following keys:
        - 'data_path': File path of .csv file that contains the data for text classification task
        - 'column_text': name of column that contains the texts
        - 'column_label': name of column that contains the text classes
        - 'artifacts_folder': folder where all the training artifacts will be saved
        - 'model': 'lr' to run logistic regression or "xgb" to run Xgboost
        - nested params for each model, e.g. {"xgb": {"params": ...}}
    """
    # Data loader
    cols = [config["column_text"], config["column_label"]]
    train_df = pd.read_csv(config["data_path"])[cols]
    train_df = train_df[train_df[cols[1]].isin(["paragraph", "abstract"])]

    if config["model"] == "xgb":
        out_path = f'{config["artifacts_folder"]}/xgb'
        parameters = config["xgb"]["params"]
        model = XGBTextClassifier()

    if config["model"] == "lr":
        model = LogRegTextClassifier()
        parameters = config["lr"]["params"]
        out_path = f'{config["artifacts_folder"]}/lr.pkl'

    if config["model"] == "bert":
        out_path = f'{config["artifacts_folder"]}/bert_checkpoint'
        parameters = config["bert"]["params"]
        parameters["output_folder"] = out_path
        model = BertTextClassifier(128, ["abstract", "paragraph"])

    if config["model"] == "transformer":
        out_path = f'{config["artifacts_folder"]}/transformer_checkpoint'
        parameters = config["transformer"]["params"]
        model = TransformerTextClassifier(128, ["abstract", "paragraph"])

    model.train(train_df[cols[0]], train_df[cols[1]], **parameters)
    model.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model of choice")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file in json format",
        default="scripts/config_train.json",
    )
    args = parser.parse_args()
    with open(args.config) as conf_file:
        config = json.load(conf_file)

    train(config)
