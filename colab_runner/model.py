from pathlib import Path

import joblib
import torch
import json

from transformers import AutoTokenizer, AutoModel
from colab_runner.data_tools import normalize


def define_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

class Detector(torch.nn.Module):
    def __init__(self, model_path, tokenizer, max_token_length, device=None, preprocessor=None, feature_columns=None, results=None, model_options=None):
        super().__init__()
        self.model_options = model_options or {}
        self.model = AutoModel.from_pretrained(model_path, **self.model_options)
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.results = results
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.max_token_length = max_token_length
        if feature_columns is None:
            self.detection_layer = torch.nn.Linear(self.model.config.hidden_size, 2)
            self.feature_columns = feature_columns
        else:
            self.feature_columns = feature_columns
            self.detection_layer = torch.nn.Linear(self.model.config.hidden_size + len(feature_columns), 2)


    def forward(self, encoding, linguistic_features=None):

        if linguistic_features is None and self.feature_columns is not None:
            raise ValueError("Features are expected")
        elif linguistic_features is not None:
            if self.feature_columns is None:
                raise ValueError("No features expected")


        outputs = self.model(**encoding)

        hidden = outputs.last_hidden_state

        cls_embeddings = hidden[:, 0, :]

        if linguistic_features is None:
            logits = self.detection_layer(cls_embeddings)

        else:

            combined_features = torch.cat([cls_embeddings, linguistic_features], dim=1)
            logits = self.detection_layer(combined_features)

        return logits

def save_model(save_path, detector):
    save_path = Path(save_path)
    tokenizer_save_path = save_path / "tokenizer"

    tokenizer_save_path.mkdir(parents=True, exist_ok=True)

    torch.save(detector.state_dict(), save_path / "model.pt")

    config = {
        "features": detector.feature_columns,
        "model_path": detector.model_path,
        "max_token_length": detector.max_token_length,
        "model_options": detector.model_options
    }

    if detector.preprocessor is not None:
        joblib.dump(detector.preprocessor, save_path / "preprocessor.joblib")

    detector.tokenizer.save_pretrained(tokenizer_save_path)

    with open(save_path / "config.json", "w") as f:
        json.dump(config, f)

    return save_path

def load_model(saved_model_path):
    saved_model_path = Path(saved_model_path)

    saved_model = saved_model_path / "model.pt"

    with open(saved_model_path / "config.json", "r") as f:
        config = json.load(f)

    features = config["features"]
    model_path = config["model_path"]
    max_token_length = config["max_token_length"]
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path / "tokenizer")
    try:
        preprocessor = joblib.load(saved_model_path / "preprocessor.joblib")
    except FileNotFoundError:
        if features is not None:
            raise IOError(f"Preprocessor file is missing for a model with linguistic features")
        preprocessor = None

    detector = Detector(
    model_path=model_path,
    feature_columns=features,
    tokenizer=tokenizer,
    preprocessor=preprocessor,
    max_token_length=max_token_length,
    model_options=config.get("model_options", {}))

    state_dict = torch.load(
        saved_model,
        map_location="cpu",
        weights_only=True,
    )

    detector.load_state_dict(state_dict)
    detector.eval()
    return detector
