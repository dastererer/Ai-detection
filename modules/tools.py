import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pystylometry.readability.syllables import count_syllables
from functools import lru_cache
from torch.utils.data import Dataset



@lru_cache(maxsize=100_000)
def cached_syllables(word):
    return count_syllables(word)


def token_depth(token):
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth


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


def clean_text(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        val = val[0] if len(val) else ""

    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="ignore")

    return "" if pd.isna(val) else str(val)

def split_data(data, split_ratio=None, seed=42):
    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]
    train_size = int(split_ratio[0] * len(data))
    validate_size = int(split_ratio[1] * len(data))
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    train, val, test = data[:train_size], data[train_size:train_size+validate_size], data[train_size+validate_size:]


    return train, val, test

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


def extract_model_name(model):
    model_name = f"{model}"
    model_name = model_name[model_name.find("/") + 1:]
    return model_name


HISTORY_METRIC_COLUMN_LABELS = {
    "val_accuracy": "Accuracy",
    "val_precision": "Precision",
    "val_recall": "Recall",
    "val_f1": "F1",
}

HISTORY_LOSS_COLUMN_LABELS = {
    "val_loss": "Validation loss",
    "train_loss": "Train loss",
}

SINGLE_METRIC_COLUMN_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

SINGLE_LOSS_COLUMN_LABELS = {
    "loss": "Loss",
}


def _prepare_metrics_dataframe(metrics):
    if isinstance(metrics, list) and all(isinstance(item, dict) for item in metrics):
        return pd.DataFrame(metrics), "line"

    if isinstance(metrics, dict):
        return pd.DataFrame([metrics]), "bar"

    raise ValueError("metrics must be either a list of dictionaries or a dictionary.")


def _plot_metric_group(axis, metrics_df, column_labels, y_label, chart_kind, x_column=None):
    if chart_kind == "line":
        has_x_axis = x_column is not None

        for column_name, label in column_labels.items():
            if has_x_axis:
                axis.plot(metrics_df[x_column], metrics_df[column_name], label=label, marker="o")
            else:
                axis.plot(metrics_df[column_name], label=label, marker="o")

        axis.set_xlabel(x_column if has_x_axis else "")
    else:
        labels = list(column_labels.values())
        values = [metrics_df[column_name].iloc[0] for column_name in column_labels]
        axis.bar(labels, values)

    axis.set_ylabel(y_label)
    axis.legend()


def plot_metrics(metrics, x_column):
    metrics_df, chart_kind = _prepare_metrics_dataframe(metrics)

    if chart_kind == "line":
        metric_column_labels = HISTORY_METRIC_COLUMN_LABELS
        loss_column_labels = HISTORY_LOSS_COLUMN_LABELS
    else:
        metric_column_labels = SINGLE_METRIC_COLUMN_LABELS
        loss_column_labels = SINGLE_LOSS_COLUMN_LABELS

    figure, axes = plt.subplots(1, 2, figsize=(12, 8), layout="constrained")

    _plot_metric_group(
        axes[0],
        metrics_df,
        metric_column_labels,
        "Metric",
        chart_kind,
        x_column,
    )
    _plot_metric_group(
        axes[1],
        metrics_df,
        loss_column_labels,
        "Loss",
        chart_kind,
        x_column,
    )

    plt.show()
    return figure
