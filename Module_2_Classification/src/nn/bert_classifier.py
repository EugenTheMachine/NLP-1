import json
import os
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split

from src.base import TextClassifier
from src.nn.data_schema import DocDataset

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer


class BertTextClassifier(TextClassifier):
    """Bert Text classifier.

    Inherits from TextClassifier and utilizes the Bert model for text classification.
    """

    def __init__(
        self,
        max_seq_length: int,
        label_names: list[str],
        model_name="distilbert/distilbert-base-uncased",
    ):
        """Initializes the BertNerClassifier with a BERT model and specified labels."""
        self.label_names = label_names
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        # Mapping labels to numbers
        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # initialize tokenizer and distilbert model using from_pretrained methods
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_names),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def collator(self, x):
        # use torch.stack to collate inputs
        # use label2id mapping to transform them into numbers
        input_ids = torch.stack([xx['input_ids'] for xx in x])
        attention_mask = torch.stack([xx['attention_mask'] for xx in x])
        labels = torch.tensor([xx['label'] for xx in x])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train(
        self, data, labels, output_folder, batch_size=64, num_epochs=1, **kwargs: Any
    ):
        """Run train and validate the model for text classification.

        Args:
            data (Iterable[str]): texts
            labels (Iterable[str]): labels
            batch_size (int): batch size
            num_epochs (int): Tokenized validation data.
            **kwargs (Any): Additional training arguments, e.g., learning_rate, num_train_epochs, weight_decay.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Create a list of dictionaries with text and label for the dataset
        dataset = [{"text": text, "label": self.label2id[label]} for text, label in zip(data, labels)]
        texts_train, texts_val = train_test_split(dataset, test_size=0.2)

        train_dataset = DocDataset(texts_train, self.tokenizer, self.max_seq_length)
        val_dataset = DocDataset(texts_val, self.tokenizer, self.max_seq_length)

        training_args = TrainingArguments(
            output_dir=output_folder,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collator,
        )

        trainer.train()

    def predict(self, samples: list[str]) -> list[dict]:
        """Generates predictions for a set of samples

        Args:
            samples (list[str]): iterable with texts

        Returns:
            list[dict]: list with textual labels
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        self.model.eval()

        inputs = self.tokenizer.batch_encode_plus(samples, max_length=self.max_seq_length, padding='max_length', truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs.to(device)).logits
            output = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred = [self.id2label[val] for val in output]
        return y_pred

    def save(self, path: str):
        """Saves model under path folder

        Args:
            path (str): folder to save the model
        """
        os.makedirs(Path(path), exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        params = {
            'max_seq_length': self.max_seq_length,
            'label_names': self.label_names,
            'model_name': self.model_name
        }
        with open(Path(path) / "model_conf.json", "w") as out:
            json.dump(params, out)

    @staticmethod
    def load(path: str) -> "BertTextClassifier":
        """Loads model from disk

        Args:
            path (str): folder with the model artefacts

        Returns:
            BertTextClassifier: new instance of the model
        """

        with open(Path(path) / "model_conf.json", "r") as params_file:
            params = json.load(params_file)

        model_instance = BertTextClassifier(**params)
        model_instance.model = DistilBertForSequenceClassification.from_pretrained(path)
        model_instance.tokenizer = DistilBertTokenizerFast.from_pretrained(path)

        return model_instance
