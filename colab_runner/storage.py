from pathlib import Path
import shutil
from colab_runner.data import file_hash


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".uploading")
    shutil.copyfile(source, temporary)
    if file_hash(source) != file_hash(temporary):
        raise IOError(f"Copy verification failed: {destination}")
    temporary.replace(destination)


def sync_results(local_folder, drive_folder, include_model=False):
    local_folder, drive_folder = Path(local_folder), Path(drive_folder)
    names = ["config.json", "resume.pt"]
    if include_model and (local_folder / "best_model").is_dir():
        names += [str(path.relative_to(local_folder)) for path in sorted((local_folder / "best_model").rglob("*")) if path.is_file()]
    names.append("results.json")
    for name in names:
        source = local_folder / name
        if source.is_file():
            copy_verified(source, drive_folder / name)
