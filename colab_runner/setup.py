from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neobert", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        constraints = Path(directory) / "constraints.txt"
        constraints.write_text(f"torch=={torch.__version__}\n")
        command = [sys.executable, "-m", "pip", "install", "-c", str(constraints)]
        subprocess.run(command + ["-r", str(Path(__file__).with_name("requirements.txt"))], check=True)
        if args.neobert:
            if torch.version.cuda is None:
                raise RuntimeError("NeoBERT needs a CUDA runtime with compatible xformers.")
            index = "https://download.pytorch.org/whl/cu" + torch.version.cuda.replace(".", "")
            subprocess.run(command + ["--only-binary=:all:", "--extra-index-url", index, "xformers"], check=True)
    print("Setup complete. The installed PyTorch version was preserved.", flush=True)


if __name__ == "__main__":
    main()
