# Colab experiments

This runner is self-contained in `colab_runner`. Its training, model and data helpers are separate copies adapted for Colab. The original project modules and notebooks do not depend on this folder and can be used without it. The runner reads existing prepared Parquet datasets. No plots are generated.

## Settings

Edit `colab_runner/config.json`. `drive_project` is set to `/content/drive/MyDrive/ai_detection`, matching the supplied Google Drive folder hierarchy. With Drive mounted at `/content/drive`, the prepared datasets are in `/content/drive/MyDrive/ai_detection/data/processed/tokenized`. These paths still need a live mount/read check in Colab. The data and output folders are relative to that project folder. Local work stays on the runtime disk.

The default comparison has seven encoders and two variants per encoder. DeBERTa is **v3-base**, as in the existing project configuration. All variants use 512 tokens, batch size 32, seed 42, AdamW, and the same prepared examples. BigBird uses full attention for this short-context experiment. NeoBERT uses its published custom model code, pinned to a commit during the audit, and requires a compatible xformers installation.

Validation runs every 2,000 batches and at the end of each epoch. A step means a processed training batch, not an individual example. With FP16, a batch with overflow can skip the optimizer update while still advancing the batch counter.

F1 means **binary F1 for label 1 (AI)**, matching the existing evaluation code. It is not macro-F1. Every strictly better validation F1 preserves the best weights. Early stopping separately requires an improvement of at least 0.0075, or 0.75 percentage points, to reset its patience. Small improvements accumulate against the last meaningful improvement. Stopping is checked at validation points after at least two epochs, with a patience window of 0.75 epoch and a maximum of four epochs. The test split is evaluated only after restoring the selected weights.

## Prepare without a GPU

Use a CPU Colab session to mount Drive and audit the cached data before paying for GPU training. The official Colab CLI must already be installed and authenticated on the computer running the launcher. Its first sign-in and Drive consent can require interaction.

From the project root:

```bash
python -m colab_runner.launch
python -m colab_runner.launch --prepare YOUR_CPU_SESSION
```

The first command only prints the plan. The second uploads the Python source, installs the training dependencies, mounts Drive, and audits all seven cached datasets. It leaves the preparation session open.

The audit checks split labels, missing data, both classes in each split, identical samples and features across tokenizers, and a small sample of cached token IDs against each pinned tokenizer. Exact text overlaps are excluded from later splits in the order train, validation, test. The original Parquet files are never changed. Excluded row numbers and text hashes are included in the run manifest. Near duplicates and related questions still require a separate dataset audit.

The dataset filenames must be `<model-name>_MAGE.parquet`, for example `bert-base-uncased_MAGE.parquet`. The current defaults match the previous runner's convention. Model revisions and dataset hashes are recorded once. Changing the code or run configuration requires a new run name.

## Run on Colab

After preparation:

```bash
python -m colab_runner.launch --run
```

The launcher requests up to seven A100 sessions, mounts Drive, and prepares each runtime. A capacity refusal after some successful allocations uses the sessions already obtained. Other allocation or setup errors stop preparation. Authentication is not bypassed; complete the first interactive setup before leaving the machine unattended.

Alternatively, supply GPU session names that you have already opened:

```bash
python -m colab_runner.launch --run --sessions gpu-one gpu-two
```

These sessions become dedicated to this runner and will be stopped after their queue is finished. Do not supply sessions that are doing unrelated work. A single coordinator assigns each model once; its two variants run sequentially on the same GPU. A free GPU takes the next model. Do not start a second coordinator for the same run on another computer.

The coordinator must remain running, with its network connection available. Colab can still reclaim a runtime. CLI access does not guarantee an overnight session or seven simultaneous GPUs.

`max_total_gpu_hours` defaults to 40. The launcher divides this allowance between the allocated sessions, counting setup time from allocation. Workers pause at batch boundaries when their deadline is reached, save, and release their session after confirmed copying. Validation, final testing, and copying can extend the deadline. This is an approximate time budget, **not a billing cap**: 40 hours only corresponds to 280 units if the actual rate is 7 units per hour. Confirm the rate and GPU memory on the actual runtime. Unused time is not automatically reassigned.

If a worker disconnects or a Drive copy cannot be confirmed, its session is left running for recovery and listed in the controller summary. It may continue consuming credits. Recover or stop it before rerunning the coordinator. Successful workers free their GPUs automatically. The runner does not silently retry a job that might still be running remotely.

You can also invoke a worker directly inside an already configured Colab runtime:

```bash
python -m colab_runner.worker --model google-bert/bert-base-uncased --hours 6
```

Direct worker mode does not allocate or stop the runtime. Automatic stopping belongs to the launcher.

## Saved files

Each run has a `config.json` containing the settings, model revisions, code hash, data hashes and exclusion report. Its `summary.json` combines reported model results. The local controller summary additionally records terminal connection and shutdown failures.

Each model/variant folder contains:

- `config.json`: model, variant, experiment fingerprint, GPU, software versions and settings.
- `results.json`: all intermediate evaluations, completed epochs, best step/epoch, best validation metrics, status, stop reason and final test metrics.
- `resume.pt`: the latest training weights, optimizer, FP16 scaler, random state, batch position, histories and best weights.
- `best_model/`: the selected weights, tokenizer, preprocessor when needed, and model configuration.

Checkpoints are written locally every 1,000 batches and at every evaluation. A temporary file is copied to Drive and checked by hashing before replacing the previous copy. Final results are published after the model files have been copied. This checks the mounted filesystem; it cannot guarantee Google's backend durability after a service failure. A hard crash can lose work since the last successful checkpoint.

A restart uses the same run name and configuration. Completed variants are skipped. Interrupted variants continue from the last saved batch, with the same shuffle order and random state. The resume files use PyTorch serialization with Python random state; only load checkpoints created by this project that you trust. The saved best model can be loaded with `colab_runner.model.load_model`, with network access or a cached copy of its pinned base model. NeoBERT also needs its xformers dependency when loading.

Inside this folder only, the training API returns `(detector, epoch_results)`. `detector.results` now holds the unified result dictionary, and `detector.step_results` remains the evaluation history. `save_path` is now an experiment folder: selected model files are inside its `best_model` subfolder. Optional `checkpoints_dir` stores one rotating `resume.pt`, rather than an export per epoch. Old run artifacts are not migrated or removed. The existing project training and Modal runner keep their original behavior and file formats.

## Verification

```bash
python -m unittest discover -s colab_runner/tests -v
```

Tests cover interrupted continuation, validation-based selection, early stopping, data filtering, train-only preprocessing, copy failure, and queue cleanup. They run on CPU without downloading encoder weights. Actual CUDA throughput, seven-model compatibility, Drive authorization, session quotas and the Colab connection still need a short live check before an overnight run.

Official references: [Colab CLI](https://github.com/googlecolab/google-colab-cli), [session management](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [execution timeouts](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md), [Colab limits](https://research.google.com/colaboratory/faq.html).
