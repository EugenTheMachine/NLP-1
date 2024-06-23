import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset


# Define the PyTorch dataset class
class DocDataset(data.Dataset):
    def __init__(self, doc_dataset, tokenizer, max_length):
        self.doc_dataset = doc_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.doc_dataset)

    def __getitem__(self, idx):
        example = self.doc_dataset[idx]
        text = example["text"]
        label = example["label"]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].unsqueeze(0).int(),
            "label": label,
        }
