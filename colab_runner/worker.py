from pathlib import Path
from functools import partial
import argparse
import gc
import hashlib
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import model_info
from colab_runner.model import Detector
from colab_runner.data_tools import collate_batch
from colab_runner.training import train, evaluate
from colab_runner.checkpoints import save_json
from colab_runner.data import file_hash, prepare_data, make_datasets
from colab_runner.storage import copy_verified, sync_results


ROOT = Path(__file__).resolve().parents[1]


def experiment_name(model, variant):
    return model.replace("/", "--") + "--" + variant


def code_hash():
    digest = hashlib.sha256()
    for folder in [ROOT / "colab_runner"]:
        for path in sorted(folder.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_config(config):
    if not config["drive_project"]:
        raise ValueError("Set drive_project to the mounted Google Drive project folder.")
    for name in ["data_folder", "output_folder"]:
        path = Path(config[name])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{name} must be relative to drive_project.")
    drive = Path(config["drive_project"]).resolve()
    local = Path(config["local_folder"]).resolve()
    if local == drive or local.is_relative_to(drive) or drive.is_relative_to(local):
        raise ValueError("Local work and persistent Drive folders must be separate.")
    if not config["models"] or len(config["models"]) != len(set(config["models"])):
        raise ValueError("Model names must be non-empty and unique.")
    if not set(config["variants"]) <= {"text_only", "with_features"} or not config["variants"]:
        raise ValueError("Choose text_only and/or with_features.")
    if len(config["variants"]) != len(set(config["variants"])):
        raise ValueError("Variants must be unique.")
    if not 2 <= config["max_token_length"] <= 512:
        raise ValueError("This experiment uses a shared context limit of at most 512 tokens.")
    if not 1 <= config["min_epochs"] <= config["epochs"]:
        raise ValueError("Expected 1 <= min_epochs <= epochs.")
    if config["batch_size"] < 1 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer or batch settings.")
    if config["min_delta"] < 0 or config["patience_epochs"] <= 0:
        raise ValueError("Invalid early stopping settings.")
    if config["eval_step"] < 1 or config["checkpoint_steps"] < 1:
        raise ValueError("Intervals must be positive.")
    if config["max_sessions"] < 1 or config["max_total_gpu_hours"] <= 0:
        raise ValueError("Invalid session limits.")
    if not config["features"] or len(config["features"]) != len(set(config["features"])):
        raise ValueError("Features must be non-empty and unique.")
    if Path(config["run_name"]).name != config["run_name"] or config["run_name"] in {".", "..", ""}:
        raise ValueError("run_name must be a folder name.")


def audit(config):
    validate_config(config)
    drive = Path(config["drive_project"])
    folder = drive / config["output_folder"] / config["run_name"]
    manifest_path = folder / "config.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != config or manifest["code_hash"] != code_hash():
            raise ValueError("Run settings or code changed. Choose a new run_name.")
        print("Existing run manifest found. Workers will verify each data file.")
        return manifest
    manifest = {"settings": config, "code_hash": code_hash(), "models": {}, "f1_average": "binary", "positive_label": 1}
    fingerprint = None
    for model in config["models"]:
        path = drive / config["data_folder"] / f"{model.rsplit('/', 1)[-1]}_{config['dataset']}.parquet"
        data, report = prepare_data(path, config["features"])
        if fingerprint is not None and fingerprint != report["sample_fingerprint"]:
            raise ValueError(f"Samples, split order or features differ for {model}.")
        fingerprint = report["sample_fingerprint"]
        revision = model_info(model).sha
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        sample = data.sample(n=min(16, len(data)), random_state=config["seed"])
        expected = tokenizer(sample["text"].tolist(), truncation=True, max_length=config["max_token_length"], padding=False)["input_ids"]
        for (_, row), tokens in zip(sample.iterrows(), expected):
            cached = [int(value) for value, active in zip(row["input_ids"], row["attention_mask"]) if active]
            if len(cached) > config["max_token_length"]:
                last = cached[-1]
                cached = cached[:config["max_token_length"]]
                if last in {tokenizer.sep_token_id, tokenizer.eos_token_id}:
                    cached[-1] = last
            if cached != tokens:
                raise ValueError(f"Cached tokens disagree with the tokenizer for {model}.")
        manifest["models"][model] = {"path": str(path), "sha256": file_hash(path), "revision": revision}
        if "data_audit" not in manifest:
            manifest["data_audit"] = report
        print(f"Audited {model}: {len(data)} rows", flush=True)
        del data
    save_json(manifest_path, manifest)
    return manifest


def run_model(config, model, deadline=None):
    validate_config(config)
    if model not in config["models"]:
        raise ValueError("Model is not listed in the experiment config.")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Select a GPU runtime before training.")
    drive_run = Path(config["drive_project"]) / config["output_folder"] / config["run_name"]
    manifest = json.loads((drive_run / "config.json").read_text())
    if manifest["settings"] != config or manifest["code_hash"] != code_hash():
        raise ValueError("Config or source code differs from the audited run.")
    info = manifest["models"][model]
    local_run = Path(config["local_folder"]) / config["run_name"]
    local_data = local_run / "data" / Path(info["path"]).name
    if not local_data.exists() or file_hash(local_data) != info["sha256"]:
        copy_verified(info["path"], local_data)
    if file_hash(local_data) != info["sha256"]:
        raise ValueError("Prepared dataset changed after the audit.")
    data, report = prepare_data(local_data, config["features"])
    if report["sample_fingerprint"] != manifest["data_audit"]["sample_fingerprint"]:
        raise ValueError("Prepared samples differ from the run manifest.")
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name} | VRAM: {properties.total_memory / 1024 ** 3:.1f} GiB", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=info["revision"])
    tokenizer.model_max_length = config["max_token_length"]
    options = {"revision": info["revision"]}
    if model == "chandar-lab/NeoBERT":
        from xformers.ops import SwiGLU
        options["trust_remote_code"] = True
    if model == "google/bigbird-roberta-base":
        options["attention_type"] = "original_full"
    summaries = []
    for variant in config["variants"]:
        if deadline is not None and time.time() >= deadline:
            break
        name = experiment_name(model, variant)
        local_folder, drive_folder = local_run / name, drive_run / name
        signature = hashlib.sha256(json.dumps({"config": config, "model": model, "variant": variant, "data": info, "code": manifest["code_hash"]}, sort_keys=True).encode()).hexdigest()
        previous = drive_folder / "results.json"
        if previous.exists():
            saved_config = json.loads((drive_folder / "config.json").read_text())
            if saved_config["experiment_id"] != signature:
                raise ValueError("Existing experiment differs. Choose a new run_name.")
            results = json.loads(previous.read_text())
            if results.get("status") == "completed" and "test" in results:
                if not (drive_folder / "best_model" / "model.pt").is_file():
                    raise FileNotFoundError("Completed run is missing its best model.")
                summaries.append({"model": model, "variant": variant, **results})
                continue
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        features = config["features"] if variant == "with_features" else None
        datasets, preprocessor = make_datasets(data, tokenizer, features, config["max_token_length"])
        loaders = {}
        for split, dataset in datasets.items():
            loaders[split] = DataLoader(
                dataset, batch_size=config["batch_size"], shuffle=split == "train", num_workers=0,
                collate_fn=partial(collate_batch, tokenizer=tokenizer),
                generator=torch.Generator().manual_seed(config["seed"]),
            )
        detector = Detector(model, tokenizer, config["max_token_length"], device="cuda", preprocessor=preprocessor, feature_columns=features, model_options=options).to("cuda")
        optimizer = torch.optim.AdamW(detector.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        criterion = torch.nn.CrossEntropyLoss()
        save_json(local_folder / "config.json", {
            "experiment_id": signature, "model": model, "variant": variant, "settings": config,
            "revision": info["revision"], "model_options": options, "data_sha256": info["sha256"],
            "gpu": properties.name, "vram_bytes": properties.total_memory,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
        })
        if (drive_folder / "resume.pt").exists():
            copy_verified(drive_folder / "resume.pt", local_folder / "resume.pt")
        elif (local_folder / "resume.pt").exists():
            raise ValueError("Local-only checkpoint found. Recover it before reusing this run folder.")

        def sync(state):
            sync_results(local_folder, drive_folder)

        started = time.time()
        detector, _ = train(
            config["epochs"], detector, loaders["train"], loaders["validate"], optimizer, criterion,
            fp16=config["fp16"], priority=config["priority"], save_path=local_folder,
            eval_step=config["eval_step"], checkpoint_steps=config["checkpoint_steps"],
            resume=(local_folder / "resume.pt").exists(), seed=config["seed"],
            min_epochs=config["min_epochs"], patience_epochs=config["patience_epochs"], min_delta=config["min_delta"],
            checkpoint_callback=sync, stop_callback=lambda: deadline is not None and time.time() >= deadline,
            experiment_id=signature,
        )
        results = detector.results
        if results["status"] in {"completed", "early_stopped"}:
            _, _, _, test = evaluate(detector, loaders["test"], criterion, fp16=config["fp16"], select_best=False)
            results["stop_reason"] = "max_epochs" if results["status"] == "completed" else "early_stopping"
            results["status"] = "completed"
            results["test"] = test
        results["last_invocation_seconds"] = time.time() - started
        save_json(local_folder / "results.json", results)
        sync_results(local_folder, drive_folder, include_model=True)
        summaries.append({"model": model, "variant": variant, **results})
        if results["status"] == "completed":
            (local_folder / "resume.pt").unlink(missing_ok=True)
            (local_folder / "best_model" / "model.pt").unlink(missing_ok=True)
        del detector, optimizer, datasets, loaders, preprocessor, dataset
        gc.collect()
        torch.cuda.empty_cache()
        if results["status"] == "paused":
            break
    local_data.unlink(missing_ok=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--receipt")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.audit:
        audit(config)
    elif args.model:
        deadline = time.time() + args.hours * 3600 if args.hours is not None else None
        if args.deadline is not None:
            deadline = args.deadline
        summaries = run_model(config, args.model, deadline)
        if args.receipt:
            save_json(args.receipt, {
                "model": args.model, "synced": True,
                "all_variants_completed": len(summaries) == len(config["variants"]) and all(result["status"] == "completed" for result in summaries),
                "variants": [{name: result.get(name) for name in ["variant", "status", "stop_reason", "best_step", "best_epoch", "best_metrics", "test"]} for result in summaries],
            })
    else:
        parser.error("Choose --audit or --model. No GPU is allocated by this script.")


if __name__ == "__main__":
    main()
