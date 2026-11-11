import pandas as pd
from pathlib import Path
import numpy as np
from modules.extract_features import extract_linguistic_features


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

def preprocess_hc3(data, features, extract_features, seed=42, save_path=None):
    if save_path is None:
        raise ValueError("Save path must be specified")

    if not features and extract_features:
        raise ValueError("Features must be specified when extract_features is True")

    if features and not extract_features:
        print("Features will not be extracted")
        features = None

    df = data["train"].to_pandas()
    parts = []

    for split_name, part in zip(
        ["train", "validate", "test"],
        split_data(df, seed=seed),
    ):
        human = pd.DataFrame({
            "text": part["human_answers"].apply(clean_text),
            "label": 0,
            "source": part["source"],
            "group_id": part["id"],
        })

        ai = pd.DataFrame({
            "text": part["chatgpt_answers"].apply(clean_text),
            "label": 1,
            "source": part["source"],
            "group_id": part["id"],
        })

        combined = pd.concat([human, ai], ignore_index=True)
        combined["dataset"] = "HC3"
        combined["split"] = split_name
        combined["sample_id"] = [i for i in range(len(combined))]

        parts.append(combined)

    train, val, test = parts

    if extract_features:
        features_train = extract_linguistic_features(train["text"], features = features)
        features_val = extract_linguistic_features(val["text"], features=features)
        features_test = extract_linguistic_features(test["text"], features=features)


        train = pd.concat([train.reset_index(drop=True), features_train], axis=1)
        val = pd.concat([val.reset_index(drop=True), features_val], axis=1)
        test = pd.concat([test.reset_index(drop=True), features_test], axis=1)



    dataset = pd.concat([train, val, test], ignore_index=True)
    dataset = dataset.dropna(axis=0)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(save_path)
