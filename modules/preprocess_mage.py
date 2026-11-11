import pandas as pd
from pathlib import Path
from modules.extract_features import extract_linguistic_features


def preprocess_mage(data, extract_features, features, save_data=False, save_path=None):

    if save_path is None:
        raise ValueError("Save path must be specified")

    if not features and extract_features:
        raise ValueError("Features must be specified when extract_features is True")

    if features and not extract_features:
        print("Features will not be extracted")
        features = None

    train = data["train"].to_pandas()
    val = data["validation"].to_pandas()
    test = data["test"].to_pandas()

    train.rename(columns={"src": "source"}, inplace=True)
    val.rename(columns={"src": "source"}, inplace=True)
    test.rename(columns={"src": "source"}, inplace=True)

    train["split"] = "train"
    val["split"] = "validate"
    test["split"] = "test"

    train["dataset"] = "MAGE"
    val["dataset"] = "MAGE"
    test["dataset"] = "MAGE"

    train["label"] = 1 - train["label"]
    val["label"] = 1 - val["label"]
    test["label"] = 1 - test["label"]

    if extract_features:

        features_train = extract_linguistic_features(train["text"], features=features)
        features_val = extract_linguistic_features(val["text"], features=features)
        features_test = extract_linguistic_features(test["text"], features=features)


        train = pd.concat([train.reset_index(drop=True), features_train], axis=1)
        val = pd.concat([val.reset_index(drop=True), features_val], axis=1)
        test = pd.concat([test.reset_index(drop=True), features_test], axis=1)


    dataset = pd.concat([train, val, test], ignore_index=True)
    dataset["sample_id"] = [i for i in range(len(dataset))]
    dataset = dataset.dropna(axis=0)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(save_path)

