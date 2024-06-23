import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from src.base import TextClassifier
from src.nn.data_schema import DocDataset
from src.nn.transformer_blocks import (
    ClassificationHead,
    Encoder,
    EncoderBlock,
    FeedForwardBlock,
    InputEmbeddings,
    MultiHeadAttentionBlock,
    PositionalEncoding,
    Transformer,
)


class TransformerTextClassifier(TextClassifier):
    """Transformer Text classifier.

    Inherits from TextClassifier and utilizes the Transformer model for text classification.

    Attributes:
        label_names (list[str]): list of possible labels.
        tokenizer: Data tokenizer
        model: Transformer model

    Methods:
        train(self, data: DocDataset, batch_size: int, num_epochs: int, num_workers: int, **kwargs: Any) -> None:
            Trains the model for text classification.

        predict(self, examples: list[str]) -> list[dict]:
            Predict label with trained model

        compute_metrics(self, logits: np.ndarray, labels: list[str]) -> dict
            Compute metrics
    """

    def __init__(self, max_seq_length: int, label_names: list[str]):
        """Initializes the BertNerClassifier with a BERT model and specified labels."""
        self.label_names = label_names
        # Mapping labels to numbers
        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.max_seq_length = max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = self.build_transformer(self.tokenizer.vocab_size, max_seq_length)

    def collator(self, x):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "label": torch.tensor([self.label2id[item["label"]] for item in x]),
        }

    def train(
        self, data, labels, batch_size=64, num_epochs=1, num_workers=0, **kwargs: Any
    ):
        """Run train and validate the model for text classification.

        Args:
            data (DocDataset): dataset.
            batch_size (int): batch size.
            num_epochs (int): Tokenized validation data.
            **kwargs (Any): Additional training arguments, e.g., learning_rate, num_train_epochs, weight_decay.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        train_dataset = DocDataset(
            Dataset.from_dict({"text": X_train, "label": y_train}),
            self.tokenizer,
            max_length=self.max_seq_length,
        )
        val_dataset = DocDataset(
            Dataset.from_dict({"text": X_test, "label": y_test}),
            self.tokenizer,
            max_length=self.max_seq_length,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            print("")
            print("======== Epoch {:} / {:} ========".format(epoch + 1, num_epochs))

            # torch.cuda.empty_cache()
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_dataloader, "Training..."):
                # input_ids, attention_mask, labels_batch = [x.to(device) for x in batch]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)

                optimizer.zero_grad()

                output = self.model(input_ids, attention_mask)

                loss = criterion(output.squeeze(), label.float())

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))

            total_eval_loss = 0.0
            self.model.eval()  # Optional when not using Model Specific layer

            print("Running Validation...")

            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    label = batch["label"].to(device)
                    output = self.model(input_ids, attention_mask)
                    loss = criterion(output.squeeze(), label.float())
                    total_eval_loss += loss.item()

            avg_val_loss = total_eval_loss / len(val_dataloader)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    def predict(self, samples: list[str]) -> list[dict]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        self.model.eval()

        tokenized_t = self.tokenizer.batch_encode_plus(
            samples,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        input_ids = tokenized_t["input_ids"].to(device)
        attention_mask = tokenized_t["attention_mask"].to(device)

        with torch.no_grad():
            output = (
                torch.sigmoid(self.model(input_ids, attention_mask))
                .cpu()
                .numpy()
                .reshape(-1)
            )

        threshold = 0.5
        y_pred = (output >= threshold).astype(int)
        y_pred = [self.id2label[val] for val in y_pred]

        return y_pred

    def build_transformer(
        self,
        src_vocab_size: int,
        src_seq_len: int,
        d_model: int = 512,
        N: int = 2,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ):
        """Build a model using needed blocks and classification head
        Args:
            src_vocab_size (int): Vocabulary size
            src_seq_len (int): Sequence length
            d_model (int): Model dimension
            N (int): Number of encoder blocks
            h (int): Number of heads
            dropout (float): Dropout rate
            d_ff (int): dimension of the feed-forward layer
        """
        # Create the embedding layers
        src_embed = InputEmbeddings(d_model, src_vocab_size)

        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(
                d_model, encoder_self_attention_block, feed_forward_block, dropout
            )
            encoder_blocks.append(encoder_block)

        # Create the encoder and decoder
        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        classification = ClassificationHead(d_model)

        # Create the transformer
        transformer = Transformer(encoder, src_embed, src_pos, classification)

        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer

    def save(self, path: str):
        os.makedirs(Path(path), exist_ok=True)
        torch.save(self.model.state_dict(), Path(path) / "model.pt")
        params = {
            "max_seq_length": self.max_seq_length,
            "label_names": self.label_names,
            "id2label": self.id2label,
            "label2id": self.label2id,
        }
        with open(Path(path) / "model_conf.json", "w") as out:
            json.dump(params, out)

    @staticmethod
    def load(path):
        with open(Path(path) / "model_conf.json", "r") as params_file:
            params = json.load(params_file)

        model_instance = TransformerTextClassifier(
            params["max_seq_length"], params["label_names"]
        )

        model_instance.label2id = {k: int(v) for k, v in params["label2id"].items()}
        model_instance.id2label = {int(k): v for k, v, in params["id2label"].items()}

        model_instance.model.load_state_dict(torch.load(Path(path) / "model.pt"))

        return model_instance
