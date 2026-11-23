from pathlib import Path
import math
import sys
import torch
from tqdm import tqdm
from detector import save_model
from state import save_json, save_checkpoint, load_checkpoint, restore_random_state, training_results


def _check_best_model(metrics, priority, model, best_metrics, best_state_dict, best_step, cur_step):
    if best_metrics is None or best_state_dict is None:
        is_better = True
    elif priority == "loss":
        is_better = metrics[priority] < best_metrics[priority]
    else:
        is_better = metrics[priority] > best_metrics[priority]

    if is_better:
        best_step = cur_step
        best_metrics = metrics.copy()
        best_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    return best_metrics, best_state_dict, best_step





def evaluate(detector, loader, criterion, best_metrics=None, best_state_dict=None, best_step=None, cur_step=0, fp16=False, priority="accuracy", select_best=True):
    detector.eval()
    device = next(detector.parameters()).device
    with torch.no_grad():
        error_sum = 0
        sample_counter = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0

        for batch in tqdm(loader, desc=getattr(loader, "phase", "Validation"), unit="batch", mininterval=2):
            labels = batch["label"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=fp16 and device.type == "cuda"):
                logits = detector(
                    batch["encoding"].to(device),
                    batch["linguistic_features"].to(device) if batch["linguistic_features"] is not None else None,
                )
                loss = criterion(logits, labels)
            error_sum += loss.item() * labels.shape[0]
            sample_counter += labels.shape[0]
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

    if sample_counter == 0:
        raise ValueError("Validation or test loader is empty.")

    metrics = {
        "loss": error_sum / sample_counter,
        "accuracy": correct / sample_counter,
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced non-finite metrics.")
    if select_best:
        best_metrics, best_state_dict, best_step = _check_best_model(metrics, priority, detector, best_metrics, best_state_dict, best_step, cur_step)
    return best_metrics, best_state_dict, best_step, metrics


def initial_state(epoch=1, start_step=0):
    return {
        "epoch": epoch, "next_batch": 0, "step": start_step,
        "loss_sum": 0.0, "sample_counter": 0,
        "best_metrics": None, "best_state_dict": None,
        "best_step": None, "best_epoch": None,
        "meaningful_score": None, "meaningful_step": start_step,
        "epochs": [], "evaluations": [], "status": "training",
    }


def record_evaluation(state, metrics, priority, steps_per_epoch, min_epochs, patience_epochs, min_delta):
    state["evaluations"].append({
        "step": state["step"],
        "epoch": state["epoch"],
        "epoch_progress": state["epoch"] - 1 + state["next_batch"] / steps_per_epoch,
        "train_loss": state["loss_sum"] / state["sample_counter"],
        **{f"val_{name}": value for name, value in metrics.items()},
    })
    if state["best_step"] == state["step"]:
        state["best_epoch"] = state["epoch"]
    score = -metrics[priority] if priority == "loss" else metrics[priority]
    if state["meaningful_score"] is None or (score > state["meaningful_score"] and score - state["meaningful_score"] >= min_delta):
        state["meaningful_score"] = score
        state["meaningful_step"] = state["step"]
    if patience_epochs is not None and state["step"] >= min_epochs * steps_per_epoch:
        if state["step"] - state["meaningful_step"] >= math.ceil(patience_epochs * steps_per_epoch):
            state["status"] = "early_stopped"
    print(state["evaluations"][-1], flush=True)


def train_one_epoch(detector, loader_train, loader_val, optimizer, criterion, scaler, best_metrics, best_state_dict, best_step, eval_step=None, step_results=None, start_step=0, epoch=1, save_path=None, priority="accuracy", state=None, checkpoint_callback=None, checkpoint_steps=250, min_epochs=2, patience_epochs=None, min_delta=0.0075, stop_callback=None):
    if state is None:
        state = initial_state(epoch, start_step)
        state.update(best_metrics=best_metrics, best_state_dict=best_state_dict, best_step=best_step)
        if step_results is not None:
            state["evaluations"] = step_results
    detector.train()
    device = next(detector.parameters()).device
    next_batch = state["next_batch"]
    epoch_results = None

    progress = tqdm(total=len(loader_train), initial=next_batch, desc=f"Epoch {epoch}", unit="step", mininterval=2)
    for i, batch in enumerate(loader_train):
        if i < next_batch:
            continue
        if "resume_random_state" in state:
            restore_random_state(state.pop("resume_random_state"))
        optimizer.zero_grad()
        labels = batch["label"]
        micro_batch = getattr(loader_train, "micro_batch", len(labels))
        batch_loss = 0.0
        for start in range(0, len(labels), micro_batch):
            block = slice(start, start + micro_batch)
            encoding = {key: value[block].to(device) for key, value in batch["encoding"].items()}
            features = batch["linguistic_features"]
            targets = labels[block].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = detector(encoding, features[block].to(device) if features is not None else None)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("Training produced a non-finite loss.")
            scaler.scale(loss * (len(targets) / len(labels))).backward()
            batch_loss += loss.item() * len(targets)
        scaler.step(optimizer)
        scaler.update()
        state["loss_sum"] += batch_loss
        state["sample_counter"] += labels.shape[0]
        state["step"] += 1
        state["next_batch"] = i + 1
        progress.set_postfix(step=state["step"], loss=f"{batch_loss / len(labels):.4f}", refresh=False)
        progress.update()
        at_end = i + 1 == len(loader_train)
        should_evaluate = at_end or (eval_step is not None and state["step"] % eval_step == 0)
        if should_evaluate:
            state["best_metrics"], state["best_state_dict"], state["best_step"], epoch_results = evaluate(
                detector, loader_val, criterion,
                state["best_metrics"], state["best_state_dict"], state["best_step"], state["step"],
                fp16=scaler.is_enabled(), priority=priority,
            )
            record_evaluation(state, epoch_results, priority, len(loader_train), min_epochs, patience_epochs, min_delta)
            detector.train()
        if stop_callback is not None and stop_callback():
            state["status"] = "paused"
        if checkpoint_callback is not None:
            if should_evaluate or state["step"] % checkpoint_steps == 0 or state["status"] != "training":
                checkpoint_callback(state)
        if state["status"] != "training":
            break

    progress.close()
    if "resume_random_state" in state:
        restore_random_state(state.pop("resume_random_state"))
    if epoch_results is None and state["evaluations"]:
        last = state["evaluations"][-1]
        if last["step"] == state["step"]:
            epoch_results = {name[4:]: value for name, value in last.items() if name.startswith("val_")}
    return state["loss_sum"] / state["sample_counter"], epoch_results, state["best_metrics"], state["best_state_dict"], state["best_step"]


def train(epochs, detector, loader_train, loader_val, optimizer, criterion, fp16=False, priority="accuracy", save_path=None, checkpoints_dir=None, eval_step=1000, checkpoint_steps=250, resume=False, seed=42, min_epochs=2, patience_epochs=None, min_delta=0.0075, checkpoint_callback=None, stop_callback=None, experiment_id=None, resume_from=None):
    if priority not in {"accuracy", "f1", "loss", "precision", "recall"}:
        raise ValueError(f"Invalid priority: {priority}")
    if epochs < 1 or checkpoint_steps < 1 or (eval_step is not None and eval_step < 1):
        raise ValueError("Epochs and step intervals must be positive.")
    if min_epochs < 1 or min_delta < 0 or (patience_epochs is not None and patience_epochs <= 0):
        raise ValueError("Invalid early stopping settings.")
    if len(loader_train) == 0 or len(loader_val) == 0:
        raise ValueError("Training and validation loaders must not be empty.")
    if loader_train.num_workers != 0:
        raise ValueError("Use num_workers=0 for reproducible checkpoint continuation.")
    if not isinstance(loader_train.sampler, (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler)):
        raise ValueError("Use a random or sequential sampler for checkpoint continuation.")
    enable_fp16 = fp16 and next(detector.parameters()).device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enable_fp16)
    settings = {
        "epochs": epochs, "steps_per_epoch": len(loader_train), "batch_size": loader_train.batch_size,
        "train_samples": len(loader_train.dataset), "seed": seed, "fp16": enable_fp16,
        "optimizer": type(optimizer).__name__, "optimizer_defaults": optimizer.defaults,
        "priority": priority, "eval_step": eval_step, "min_epochs": min_epochs,
        "patience_epochs": patience_epochs, "min_delta": min_delta, "experiment_id": experiment_id,
    }
    folder = Path(save_path) if save_path is not None else None
    checkpoint_folder = Path(checkpoints_dir) if checkpoints_dir is not None else folder
    checkpoint_path = checkpoint_folder / "resume.pt" if checkpoint_folder is not None else None
    if resume and checkpoint_path is None:
        raise ValueError("Resume requires a save folder.")
    state = initial_state()
    if resume:
        state = load_checkpoint(resume_from or checkpoint_path, detector, optimizer, scaler, settings)
        if state["status"] == "paused":
            state["status"] = "training"

    def persist(current_state):
        if checkpoint_path is not None:
            tqdm.write(f"Saving checkpoint at step {current_state['step']} to {checkpoint_path}", file=sys.stdout)
            save_checkpoint(checkpoint_path, detector, optimizer, scaler, current_state, settings)
        if folder is not None:
            save_json(folder / "results.json", training_results(current_state, priority))
        if checkpoint_callback is not None:
            checkpoint_callback(current_state)

    while state["epoch"] <= epochs and state["status"] == "training":
        epoch = state["epoch"]
        loader_train.generator = torch.Generator().manual_seed(seed + epoch)
        if isinstance(loader_train.sampler, torch.utils.data.RandomSampler):
            loader_train.sampler.generator = torch.Generator().manual_seed(seed + epoch)
        print(f"Epoch {epoch}/{epochs} | next batch: {state['next_batch']} | FP16: {enable_fp16}", flush=True)
        train_loss, metrics, _, _, _ = train_one_epoch(
            detector, loader_train, loader_val, optimizer, criterion, scaler,
            state["best_metrics"], state["best_state_dict"], state["best_step"],
            eval_step=eval_step, epoch=epoch, priority=priority, state=state,
            checkpoint_callback=persist, checkpoint_steps=checkpoint_steps,
            min_epochs=min_epochs, patience_epochs=patience_epochs, min_delta=min_delta,
            stop_callback=stop_callback,
        )
        if state["next_batch"] == len(loader_train):
            state["epochs"].append({
                "epoch": epoch, "step": state["step"], "train_loss": train_loss,
                **{f"val_{name}": value for name, value in metrics.items()},
            })
            state["epoch"] += 1
            state["next_batch"] = 0
            state["loss_sum"] = 0.0
            state["sample_counter"] = 0
        if state["epoch"] > epochs and state["status"] == "training":
            state["status"] = "completed"
        persist(state)

    if state["best_state_dict"] is not None:
        detector.load_state_dict(state["best_state_dict"])
    detector.eval()
    detector.results = training_results(state, priority)
    detector.step_results = state["evaluations"]
    if folder is not None and state["status"] in {"completed", "early_stopped"}:
        save_model(folder / "best_model", detector)
    return detector, state["epochs"]
