from pathlib import Path
import json
import random
import numpy as np
import torch


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as file:
        json.dump(data, file, indent=2, allow_nan=False)
    temporary.replace(path)


def get_random_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, detector, optimizer, scaler, state, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "model": detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "state": state,
        "settings": settings,
        "random_state": get_random_state(),
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, detector, optimizer, scaler, settings):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["settings"] != settings:
        different = [k for k in settings if checkpoint["settings"].get(k) != settings[k]]
        raise ValueError(f"Checkpoint settings differ in {different}: {path}")
    detector.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    state = checkpoint["state"]
    state["resume_random_state"] = checkpoint["random_state"]
    return state


def training_results(state, priority):
    return {
        "status": state["status"],
        "priority": priority,
        "best_step": state["best_step"],
        "best_epoch": state["best_epoch"],
        "best_metrics": state["best_metrics"],
        "step": state["step"],
        "epoch": state["epoch"],
        "next_batch": state["next_batch"],
        "epochs": state["epochs"],
        "evaluations": state["evaluations"],
    }
