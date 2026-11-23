from pathlib import Path

import joblib
import torch
import json

from transformers import AutoModel




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
