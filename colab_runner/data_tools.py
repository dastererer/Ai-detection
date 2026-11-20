import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset



def normalize(data, features=None, preprocessor=None):

    data = data.copy()

    if features is None:
        raise ValueError("Features must be specified")

    if preprocessor is None:
        preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        )

        normalized = preprocessor.fit_transform(data[features]).astype("float32")
    else:
        normalized = preprocessor.transform(data[features]).astype("float32")


    data[features] = normalized

    return data, preprocessor


def collate_batch(batch, tokenizer):
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]

    combined = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


    encoding = tokenizer.pad(combined, return_tensors="pt")

    label = torch.stack([sample["label"] for sample in batch])
    linguistic_features = torch.stack([sample["linguistic_features"] for sample in batch]) if batch[0]["linguistic_features"] is not None else None
    return { "encoding": encoding,
             "label": label,
             "linguistic_features": linguistic_features
             }

class Example(Dataset):
    def __init__(self, encoding, label, linguistic_features):
        self.encoding = encoding.to_dict(orient="records")
        self.label = torch.tensor(label.to_numpy(), dtype=torch.long)
        self.linguistic_features = torch.tensor(linguistic_features.to_numpy(), dtype=torch.float32) if linguistic_features is not None else None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {"input_ids": self.encoding[idx]["input_ids"],
                "attention_mask": self.encoding[idx]["attention_mask"],
                "label": self.label[idx], "linguistic_features": self.linguistic_features[idx]
                if self.linguistic_features is not None else None
                }
