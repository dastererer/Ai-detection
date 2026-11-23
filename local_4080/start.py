"""Prepare a private CUDA environment and run the local experiment queue."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


def main():
    here = Path(__file__).resolve().parent
    args = sys.argv[1:]
    if "--plan" in args or "--help" in args or "-h" in args:
        return subprocess.call([sys.executable, str(here / "run.py"), *args])
    if sys.platform not in {"win32", "linux"} or sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise SystemExit("Use Python 3.11 or 3.12 on the Windows/Linux NVIDIA laptop.")
    check = subprocess.run([sys.executable, str(here / "run.py"), *args, "--plan"])
    if check.returncode:
        return check.returncode
    env = here / ".venv"
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = here / "requirements.txt"
    cuda_packages = ["torch==2.9.1", "xformers==0.0.33.post2"]
    fingerprint = hashlib.sha256(requirements.read_bytes() + repr(cuda_packages).encode()).hexdigest()
    marker = env / "requirements.sha256"
    if not python.is_file():
        print(f"Creating training environment in {env}", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    if not marker.is_file() or marker.read_text() != fingerprint:
        print("Installing CUDA PyTorch and training dependencies (first start)", flush=True)
        install = [str(python), "-m", "pip", "install"]
        subprocess.run(install + ["--index-url", "https://download.pytorch.org/whl/cu128", *cuda_packages], check=True)
        subprocess.run(install + ["-r", str(requirements)], check=True)
        marker.write_text(fingerprint)
    return subprocess.call([str(python), "-u", str(here / "run.py"), *args])


if __name__ == "__main__":
    sys.exit(main())
