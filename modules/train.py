from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
from modules.model import save_model
from modules.tools import plot_metrics

def evaluate(detector, loader, criterion):
    detector.eval()
    device = next(detector.parameters()).device
    with torch.no_grad():
        error_sum = 0
        sample_counter = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0

        for i, batch in enumerate(loader):
            labels = batch["label"].to(device)
            logits = detector(
                batch["encoding"].to(device),
                batch["linguistic_features"].to(device) if batch["linguistic_features"] is not None else None,
            )
            loss = criterion(logits, labels)
            error_sum += loss.item() * labels.shape[0]
            sample_counter += labels.shape[0]
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            labels = labels
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()
    metrics = {
        "loss": error_sum / sample_counter,
        "accuracy": correct / sample_counter,
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    }
    return metrics

def train_one_epoch(detector, loader_train, loader_val, optimizer, criterion, scaler, eval_step=None):
    detector.train()
    error_sum = 0
    sample_counter = 0
    device = next(detector.parameters()).device
    step_res =[]
    figure_m = None
    for i, batch in tqdm(enumerate(loader_train), total=len(loader_train), desc="Training: ", smoothing=0.05):

        autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled())

        with autocast:

            optimizer.zero_grad()

            labels = batch["label"].to(device)
            logits = detector(
                batch["encoding"].to(device),
                batch["linguistic_features"].to(device) if batch["linguistic_features"] is not None else None,
            )

            loss = criterion(logits, labels)
        error_sum += loss.item() * labels.shape[0]
        sample_counter += labels.shape[0]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step = i + 1

        if eval_step is not None and step % eval_step == 0:
            evaluation_results = evaluate(
                detector, loader_val, criterion
            )
            detector.train()


            evaluation_results = {
                "step": step,
                "train_loss": error_sum / sample_counter,
                **{
                    f"val_{name}": value
                    for name, value in evaluation_results.items()
                },
            }

            step_res.append(evaluation_results)
            figure_m = plot_metrics(step_res, "step")

    epoch_results = evaluate(detector, loader_val, criterion)
    return error_sum / sample_counter, epoch_results, figure_m

def train(epochs, detector, loader_train, loader_val, optimizer, criterion, fp16=False, priority="accuracy", save_path=None, checkpoints_dir=None, eval_step=1000):
    valid_metrics = {"accuracy", "f1", "loss", "precision", "recall"}

    if priority not in valid_metrics:
        raise ValueError(f"Invalid priority: {priority}")

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if eval_step is not None and eval_step < 1:
        raise ValueError("eval_step must be positive or None.")

    if checkpoints_dir is not None:
        checkpoints_dir = Path(checkpoints_dir)

    results = []
    best_epoch = None
    best_score = None
    best_state_dict = None

    if fp16 and torch.cuda.is_available() and next(detector.parameters()).device.type == "cuda":
        print("Using FP16")
        enable_fp16 = True
    else:
        print("FP16 is not supported on this device or not enabled.")
        enable_fp16 = False

    scaler = torch.amp.GradScaler(enabled=enable_fp16)

    for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training_epoch: ", smoothing=0.05):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, epoch_results, results_figure = train_one_epoch(
            detector=detector,
            loader_train=loader_train,
            loader_val=loader_val,
            optimizer=optimizer,
            criterion=criterion,
            eval_step=eval_step,
            scaler = scaler
        )

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": epoch_results["loss"],
            "val_accuracy": epoch_results["accuracy"],
            "val_precision": epoch_results["precision"],
            "val_recall": epoch_results["recall"],
            "val_f1": epoch_results["f1"],
        })

        plot_metrics(results, "epoch")

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation: {epoch_results}")

        score = epoch_results[priority]

        if best_score is None:
            is_better = True
        elif priority == "loss":
            is_better = score < best_score
        else:
            is_better = score > best_score

        if is_better:
            best_score = score
            best_epoch = epoch

            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in detector.state_dict().items()
            }

        if checkpoints_dir is not None:
            save_model(
                checkpoints_dir / f"epoch_{epoch}.pt",
                detector,
            )

    detector.load_state_dict(best_state_dict)
    detector.eval()

    figure = plot_metrics(results, "epoch")
    detector.performance = figure
    detector.results = results


    if save_path is not None:
        save_model(save_path, detector)

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Validation {priority}: {best_score:.4f}"
    )

    return detector, results
