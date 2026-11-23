# Overnight MAGE training on one RTX 4080 Laptop GPU

Use 64-bit Python 3.11 or 3.12 and an NVIDIA driver supporting CUDA 12.8. Keep the laptop on AC power with system sleep disabled during training. Screen lock is fine. Launch from the repository root.

```text
ai-detection/                         repository
  local_4080/
  ai-detection/                       copied project folder
    data/processed/tokenized/
      NeoBERT_MAGE.parquet
      bigbird-roberta-base_MAGE.parquet
      deberta-v3-base_MAGE.parquet
    runs/mage-colab-01/
      config.json                     original Colab run manifest
      chandar-lab--NeoBERT--with_features/    if a checkpoint exists
      ...                                     other imported experiments
```

Windows PowerShell:

```powershell
py -3.12 local_4080/start.py --plan
py -3.12 local_4080/start.py
```

Linux or WSL:

```bash
python3.12 local_4080/start.py --plan
python3.12 local_4080/start.py
```

The plan uses only Python's standard library. First launch creates a private environment, downloads the pinned CUDA wheels and training dependencies, and loads the model revisions recorded in the manifest from Hugging Face. Subsequent runs reuse the environment. Keep enough free disk for the model cache, the best weights and the temporary and final copies of multi-GB optimizer checkpoints.

The launcher also recognizes the existing ai_detection layout or data and runs directly at the repository root. Use explicit directories if there are multiple copies:

```bash
python local_4080/start.py --archive "D:/MAGE/runs/mage-colab-01" --data-dir "D:/MAGE/data/processed/tokenized"
```

The queue starts with NeoBERT with features, then both BigBird variants, then both DeBERTa variants. NeoBERT text_only was already completed and is excluded. To select one model use e.g. `--model microsoft/deberta-v3-base`. Finished variants are skipped. For each pending variant the launcher prefers its local checkpoint, otherwise it imports a matching Colab resume.pt. Use trusted checkpoint files. SHA-256, split/sample fingerprint, experiment ID, epoch count and optimizer settings are checked before continuation.

Inputs remain in the imported folders. New results go to `local_4080/runs/mage-colab-01/<model>--<variant>/`. Run the same command after interruption to resume from the last atomic checkpoint. The terminal shows the active variant, epoch, optimizer step, loss, ETA, validation and test progress. The append-only train.log records queue events, metrics and tracebacks. If an individual experiment fails, the remaining variants are attempted and the process exits with a nonzero status listing the failures.

The training code uses the Colab runner's dataset filtering, feature scaling, optimizer, early stopping, metrics and checkpoint format. It splits each original batch of 32 into microbatches of 2, accumulates gradients weighted by the actual number of samples, and runs evaluation in microbatches. The validation and checkpoint intervals count logical optimizer steps. The original sequence limit and FP16 setting come from the manifest. Different CUDA kernels, dropout and microbatching can change numerical results. If GPU memory is still insufficient, rerun with `--micro-batch 1`. This changes neither checkpoint step numbering nor the effective training batch size.
