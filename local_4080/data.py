from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch


SPLITS = ["train", "validate", "test"]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(path, features):
    data = pd.read_parquet(path).reset_index(drop=True)
    required = {"text", "label", "split", "input_ids", "attention_mask", *features}
    if required - set(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data.empty or data[["text", "label", "split"]].isna().any().any():
        raise ValueError("Empty data or missing text, label, split.")
    if set(data["split"].unique()) != set(SPLITS) or not set(data["label"].unique()) <= {0, 1}:
        raise ValueError("Expected train/validate/test splits and labels 0/1.")
    if not data["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Texts must be strings.")
    data["row_id"] = np.arange(len(data))
    keys = data["text"].str.strip()
    if keys.eq("").any():
        raise ValueError("Empty texts found.")
    keep = pd.Series(True, index=data.index)
    seen = set()
    for split in SPLITS:
        rows = data["split"].eq(split)
        keep.loc[rows] = ~keys.loc[rows].isin(seen)
        seen.update(keys.loc[rows & keep])
    excluded = data.loc[~keep, ["row_id", "split", "label"]].copy()
    excluded["text_sha256"] = keys.loc[~keep].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    data = data.loc[keep].reset_index(drop=True)
    counts = {}
    for split in SPLITS:
        labels = data.loc[data["split"].eq(split), "label"]
        if set(labels.unique()) != {0, 1}:
            raise ValueError(f"Both classes are required in {split}.")
        counts[split] = {str(label): int(count) for label, count in labels.value_counts().items()}
    if np.isinf(data[features].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values found.")
    for ids, mask in zip(data["input_ids"], data["attention_mask"]):
        if len(ids) < 2 or len(ids) != len(mask) or not set(mask) <= {0, 1}:
            raise ValueError("Invalid cached token IDs or attention mask.")
    identity = data[["row_id", "text", "label", "split"] + features]
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(identity, index=False).values.tobytes()).hexdigest()
    report = {
        "sample_fingerprint": fingerprint,
        "class_counts": counts,
        "excluded_rows": excluded.to_dict(orient="records"),
        "exclusion_rule": "Keep exact stripped text in the first split: train, validate, test.",
        "group_check": "Exact texts only; related questions and near duplicates need a separate audit.",
    }
    return data, report



class Samples(torch.utils.data.Dataset):
    def __init__(self, ids, labels, features):
        self.ids = ids
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.labels[index], self.features[index] if self.features is not None else None


def make_datasets(data, tokenizer, columns, max_length):
    from tqdm import tqdm

    processor = None
    if columns:
        processor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        processor.fit(data.loc[data["split"].eq("train"), columns])
    datasets = {}
    vocab_size = len(tokenizer)
    end_tokens = (tokenizer.sep_token_id, tokenizer.eos_token_id)
    for split in SPLITS:
        rows = data.loc[data["split"].eq(split)]
        features = None
        if processor is not None:
            features = torch.from_numpy(processor.transform(rows[columns]).astype("float32"))
        labels = torch.tensor(rows["label"].to_numpy(), dtype=torch.long)
        ids = []
        for tokens, mask in tqdm(zip(rows["input_ids"], rows["attention_mask"]), total=len(rows), desc=f"Prepare {split}", mininterval=2):
            tokens = np.asarray(tokens)[np.asarray(mask, dtype=bool)]
            if len(tokens) < 2 or tokens.min() < 0 or tokens.max() >= vocab_size:
                raise ValueError("Invalid cached token IDs")
            last = tokens[-1]
            tokens = tokens[:max_length].astype(np.int32)
            if last in end_tokens:
                tokens[-1] = last
            ids.append(tokens)
        datasets[split] = Samples(ids, labels, features)
    return datasets, processor


def collate(batch, tokenizer):
    ids, labels, features = zip(*batch)
    encoding = tokenizer.pad({"input_ids": [row.tolist() for row in ids]},
                             return_attention_mask=True, return_tensors="pt")
    return {"encoding": encoding, "label": torch.stack(labels),
            "linguistic_features": torch.stack(features) if features[0] is not None else None}
