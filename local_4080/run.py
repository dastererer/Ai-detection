"""Continue the five remaining MAGE experiments on one local CUDA GPU."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QUEUE = [
    ("chandar-lab/NeoBERT", "with_features"),
    ("google/bigbird-roberta-base", "text_only"),
    ("google/bigbird-roberta-base", "with_features"),
    ("microsoft/deberta-v3-base", "text_only"),
    ("microsoft/deberta-v3-base", "with_features"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locate(requested, options, description):
    candidates = [requested] if requested else options
    matches = [path.resolve() for path in candidates if path.exists()]
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        paths = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Expected one {description}. Found {len(matches)}. Checked: {paths}")
    return matches[0]


def folders(args):
    roots = [ROOT / "ai-detection", ROOT / "ai_detection", ROOT]
    candidates = [base / suffix / "config.json" for base in roots for suffix in
                  ("runs/mage-colab-01", "mage-colab-01", "")]
    if args.archive is None:
        existing = []
        for p in candidates:
            if p.is_file():
                content = read_json(p)
                if "models" in content and "settings" in content:
                    existing.append(p)
        if existing:
            candidates = existing[:1]
    manifest = locate(args.archive / "config.json" if args.archive else None, candidates, "Colab run manifest (--archive)")
    return manifest.parent, roots


def inspect_experiment(folder, signature):
    config = folder / "config.json"
    progress = folder / "results.json"
    checkpoint = folder / "resume.pt"
    if config.exists() and read_json(config)["experiment_id"] != signature:
        raise ValueError(f"Experiment signature mismatch: {folder}")
    if progress.exists():
        if not config.is_file():
            raise ValueError(f"Missing experiment config: {folder}")
        result = read_json(progress)
        if result.get("status") == "completed" and "test" in result:
            if not (folder / "best_model/model.pt").is_file():
                raise ValueError(f"Missing best model: {folder}")
            return "completed"
        if result.get("step", 0) and not checkpoint.is_file():
            raise ValueError(f"Missing resume.pt for step {result['step']}: {folder}")
    if checkpoint.exists():
        return "resume"
    return "new"


def plan(args):
    archive, roots = folders(args)
    manifest = read_json(archive / "config.json")
    config = manifest["settings"]
    output = HERE / "runs" / archive.name
    if output.resolve() == archive.resolve():
        raise ValueError("Local output must be separate from the imported run.")
    jobs = []
    for model, variant in QUEUE:
        if args.model and model != args.model:
            continue
        name = model.replace("/", "--") + "--" + variant
        info = manifest["models"][model]
        signature = hashlib.sha256(json.dumps({"config": config, "model": model, "variant": variant,
            "data": info, "code": manifest["code_hash"]}, sort_keys=True).encode()).hexdigest()
        folder = output / name
        state = inspect_experiment(folder, signature)
        origin = folder
        if state == "new":
            origin = archive / name
            state = inspect_experiment(origin, signature)
        path = None
        if state != "completed":
            file = Path(info["path"]).name
            options = [base / config["data_folder"] / file for base in roots] + [archive / file, ROOT / "ai-detection" / file]
            available = [item for item in options if item.is_file()]
            path = locate(args.data_dir / file if args.data_dir else None, available[:1] or options, f"{file} (--data-dir)")
        print(f"{state.upper():9} {name}" + (f"  <- {origin / 'resume.pt'}" if state == "resume" else ""))
        if state != "completed":
            jobs.append((model, variant, signature, folder, origin / "resume.pt" if state == "resume" else None, path))
    return manifest, output, jobs


def experiment(job, manifest, data, micro_batch):
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from data import collate, make_datasets
    from detector import Detector
    from state import save_json
    from training import train, evaluate

    model, variant, signature, folder, checkpoint, path = job
    config = manifest["settings"]
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    info = manifest["models"][model]
    options = {"revision": info["revision"]}
    if model == "chandar-lab/NeoBERT":
        from xformers.ops import SwiGLU
        options["trust_remote_code"] = True
    if model == "google/bigbird-roberta-base":
        options["attention_type"] = "original_full"
    features = config["features"] if variant == "with_features" else None
    print(f"Loading tokenizer and preparing splits for {model} ({variant})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=info["revision"])
    tokenizer.model_max_length = config["max_token_length"]
    datasets, preprocessor = make_datasets(data, tokenizer, features, config["max_token_length"])
    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(dataset, batch_size=config["batch_size"] if split == "train" else micro_batch,
            shuffle=split == "train", generator=torch.Generator().manual_seed(seed),
            collate_fn=partial(collate, tokenizer=tokenizer))
        loaders[split].phase = split.capitalize()
    loaders["train"].micro_batch = micro_batch
    print(f"Loading {model} on CUDA | microbatch {micro_batch} | effective batch {config['batch_size']}", flush=True)
    detector = Detector(model, tokenizer, config["max_token_length"], device="cuda",
                        preprocessor=preprocessor, feature_columns=features, model_options=options).to("cuda")
    optimizer = torch.optim.AdamW(detector.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()
    save_json(folder / "config.json", {"experiment_id": signature, "model": model, "variant": variant,
        "settings": config, "revision": info["revision"], "model_options": options, "data_sha256": info["sha256"],
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "micro_batch": micro_batch, "effective_batch": config["batch_size"], "source_file": str(path)})
    start = time.monotonic()
    if checkpoint:
        print(f"Restoring {checkpoint}", flush=True)
    train(config["epochs"], detector, loaders["train"], loaders["validate"], optimizer, criterion,
          fp16=config["fp16"], priority=config["priority"], save_path=folder, eval_step=config["eval_step"],
          checkpoint_steps=config["checkpoint_steps"], resume=checkpoint is not None, resume_from=checkpoint,
          seed=seed, min_epochs=config["min_epochs"], patience_epochs=config["patience_epochs"],
          min_delta=config["min_delta"], experiment_id=signature)
    result = detector.results
    print(f"Testing best weights from step {result['best_step']}", flush=True)
    result["test"] = evaluate(detector, loaders["test"], criterion, fp16=config["fp16"], select_best=False)[-1]
    result["stop_reason"] = "early_stopping" if result["status"] == "early_stopped" else "max_epochs"
    result["status"] = "completed"
    result["last_invocation_seconds"] = time.monotonic() - start
    save_json(folder / "results.json", result)
    print(f"Finished {model} ({variant}) | best step {result['best_step']} | test F1 {result['test']['f1']:.4f}", flush=True)


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Imported mage-colab-01 directory containing config.json")
    parser.add_argument("--data-dir", type=Path, help="Directory containing tokenized MAGE parquet files")
    parser.add_argument("--model", choices=list(dict.fromkeys(model for model, _ in QUEUE)))
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--plan", action="store_true", help="List completed, resumable and new jobs without GPU")
    args = parser.parse_args()
    if args.micro_batch < 1:
        parser.error("--micro-batch must be positive")
    if args.plan:
        plan(args)
        return

    import gc
    import torch
    from filelock import FileLock
    from data import file_hash, prepare_data

    archive, _ = folders(args)
    output = HERE / "runs" / archive.name
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    with FileLock(output / "queue.lock", timeout=0), (output / "train.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            manifest, _, jobs = plan(args)
            if not jobs:
                print("Queue already complete.")
                return
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Check nvidia-smi and install a CUDA build of PyTorch.")
            print(f"GPU: {torch.cuda.get_device_name(0)} | log: {log.name}", flush=True)
            last_path = None
            data = None
            for index, job in enumerate(jobs, 1):
                model, variant, _, _, _, path = job
                try:
                    print(f"[{index}/{len(jobs)}] {model} | {variant}", flush=True)
                    if path != last_path:
                        data = None
                        last_path = None
                        print(f"Verifying and reading {path}", flush=True)
                        if file_hash(path) != manifest["models"][model]["sha256"]:
                            raise ValueError(f"Parquet checksum differs from Colab: {path}")
                        data, report = prepare_data(path, manifest["settings"]["features"])
                        if report["sample_fingerprint"] != manifest["data_audit"]["sample_fingerprint"]:
                            raise ValueError("Sample fingerprint differs from Colab.")
                        last_path = path
                    experiment(job, manifest, data, args.micro_batch)
                except Exception:
                    failures.append((model, variant))
                    traceback.print_exc(file=sys.stdout)
                gc.collect()
                torch.cuda.empty_cache()
            if failures:
                raise SystemExit(f"Failed jobs: {failures}. See {log.name}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
