from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from colab_runner.data_tools import Example, normalize


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


def make_datasets(data, tokenizer, features, max_length):
    data = data.copy()
    token_ids = []
    masks = []
    for ids, mask in zip(data["input_ids"], data["attention_mask"]):
        ids = [int(value) for value, active in zip(ids, mask) if active]
        if len(ids) < 2 or min(ids) < 0 or max(ids) >= len(tokenizer):
            raise ValueError("Invalid token IDs for this tokenizer.")
        if len(ids) > max_length:
            last = ids[-1]
            ids = ids[:max_length]
            if last in {tokenizer.sep_token_id, tokenizer.eos_token_id}:
                ids[-1] = last
        token_ids.append(ids)
        masks.append([1] * len(ids))
    data["input_ids"] = token_ids
    data["attention_mask"] = masks
    preprocessor = None
    if features is not None:
        _, preprocessor = normalize(data.loc[data["split"].eq("train")], features)
        data, _ = normalize(data, features, preprocessor)
    datasets = {}
    for split in SPLITS:
        part = data.loc[data["split"].eq(split)].reset_index(drop=True)
        datasets[split] = Example(
            part[["input_ids", "attention_mask"]], part["label"],
            part[features] if features is not None else None,
        )
    return datasets, preprocessor
