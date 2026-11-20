from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import queue
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def colab(arguments, code=None, capture=False):
    return subprocess.run(
        ["colab", *arguments], input=code, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )


def execute(session, code):
    colab(["exec", "-s", session, "--timeout", "0"], code=code)


def bundle_code(path, config):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in [ROOT / "colab_runner"]:
            for source in sorted(folder.glob("*.py")):
                archive.write(source, source.relative_to(ROOT))
        archive.write(ROOT / "colab_runner" / "requirements.txt", "colab_runner/requirements.txt")
        archive.writestr("colab_runner/config.json", json.dumps(config))


def prepare_session(session, bundle, remote_root):
    colab(["drivemount", "-s", session, "/content/drive"])
    remote_bundle = "content/" + bundle.name
    colab(["upload", "-s", session, str(bundle), remote_bundle])
    execute(session, "\n".join([
        "import zipfile, subprocess, sys",
        f"zipfile.ZipFile({'/' + remote_bundle!r}).extractall({remote_root!r})",
        f"subprocess.run([sys.executable, '-m', 'colab_runner.setup'], cwd={remote_root!r}, check=True)",
    ]))


def publish_summary(session, config, summary):
    path = str(Path(config["drive_project"]) / config["output_folder"] / config["run_name"] / "summary.json")
    execute(session, "\n".join([
        "from pathlib import Path",
        f"summary_path = Path({path!r})",
        "temporary = summary_path.with_name('summary.json.tmp')",
        f"temporary.write_text({json.dumps(summary, indent=2, allow_nan=False)!r})",
        "temporary.replace(summary_path)",
    ]))


def run_queue(config, sessions, remote_root, folder):
    jobs = queue.Queue()
    for model in config["models"]:
        jobs.put(model)
    summary = {"models": [], "errors": [], "sessions_left_running": [], "unstarted_models": [], "unfinished_models": []}
    lock = threading.Lock()
    hours = config["max_total_gpu_hours"] / len(sessions)

    def worker(session, allocated_at):
        safe_to_stop = True
        model = None
        deadline = allocated_at + hours * 3600
        try:
            while time.time() < deadline:
                try:
                    model = jobs.get_nowait()
                except queue.Empty:
                    break
                receipt = "receipt-" + uuid.uuid4().hex + ".json"
                if model == "chandar-lab/NeoBERT":
                    execute(session, f"import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'colab_runner.setup', '--neobert'], cwd={remote_root!r}, check=True)")
                safe_to_stop = False
                command = ["-m", "colab_runner.worker", "--model", model, "--deadline", str(deadline), "--receipt", "/content/" + receipt]
                execute(session, f"import subprocess, sys\nsubprocess.run([sys.executable] + {command!r}, cwd={remote_root!r}, check=True)")
                local_receipt = folder / receipt
                colab(["download", "-s", session, "content/" + receipt, str(local_receipt)])
                result = json.loads(local_receipt.read_text())
                if result["model"] != model or not result["synced"]:
                    raise RuntimeError("Missing confirmation that results reached Google Drive.")
                safe_to_stop = True
                with lock:
                    summary["models"].append(result)
                    if not result["all_variants_completed"]:
                        summary["unfinished_models"].append(model)
                    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
                    publish_summary(session, config, summary)
                local_receipt.unlink()
        except Exception as error:
            with lock:
                summary["errors"].append({"session": session, "model": model, "error": str(error)})
            print(f"Session {session} failed: {error}", flush=True)
        finally:
            if safe_to_stop:
                try:
                    colab(["stop", "-s", session])
                except Exception as error:
                    with lock:
                        summary["sessions_left_running"].append(session)
                        summary["errors"].append({"session": session, "error": str(error)})
            else:
                with lock:
                    summary["sessions_left_running"].append(session)
                print(f"Sync was not confirmed. Session {session} was left running for recovery; it may still consume credits.", flush=True)

    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = [executor.submit(worker, session, started) for session, started in sessions]
        for future in futures:
            future.result()
    while not jobs.empty():
        summary["unstarted_models"].append(jobs.get_nowait())
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--prepare", metavar="CPU_SESSION")
    parser.add_argument("--sessions", nargs="+")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not args.run and not args.prepare:
        print(json.dumps(config, indent=2))
        print("Plan only. No sessions allocated. Set drive_project, prepare an audited run, then use --run.")
        return
    if not config["drive_project"]:
        parser.error("Set drive_project before preparing or launching.")
    if config["max_sessions"] < 1 or config["max_total_gpu_hours"] <= 0:
        parser.error("Session limits must be positive.")
    if Path(config["run_name"]).name != config["run_name"] or config["run_name"] in {".", "..", ""}:
        parser.error("Invalid run_name.")
    if args.sessions and len(args.sessions) != len(set(args.sessions)):
        parser.error("Session names must be unique.")
    folder = ROOT / "runs" / (config["run_name"] + "-controller")
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "controller.lock", "w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remote_root = "/content/ai_detection_code_" + uuid.uuid4().hex[:10]
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / ("ai-detection-" + uuid.uuid4().hex + ".zip")
            bundle_code(bundle, config)
            if args.prepare:
                prepare_session(args.prepare, bundle, remote_root)
                execute(args.prepare, f"import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'colab_runner.worker', '--audit'], cwd={remote_root!r}, check=True)")
                print("Data audit saved on Drive. The preparation session is still open.")
                return
            sessions = []
            try:
                targets = args.sessions or ["ai-detection-" + uuid.uuid4().hex[:10] for _ in range(min(config["max_sessions"], len(config["models"])))]
                for session in targets:
                    started = time.time()
                    if not args.sessions:
                        try:
                            result = colab(["new", "-s", session, "--gpu", config["gpu"], "--high-mem"], capture=True)
                            print(result.stdout, flush=True)
                        except subprocess.CalledProcessError as error:
                            output = error.stdout or ""
                            if sessions and ("Allocation refused" in output or "capacity limit" in output):
                                print("Using the GPU sessions already allocated.", flush=True)
                                break
                            raise
                    sessions.append((session, started))
                    prepare_session(session, bundle, remote_root)
                    manifest = str(Path(config["drive_project"]) / config["output_folder"] / config["run_name"] / "config.json")
                    execute(session, f"from pathlib import Path\nassert Path({manifest!r}).is_file(), 'Run the CPU data audit first.'")
            except BaseException:
                for session, _ in sessions:
                    colab(["stop", "-s", session])
                raise
            result = run_queue(config, sessions, remote_root, folder)
            print(json.dumps(result, indent=2))
            if result["errors"] or result["unstarted_models"] or result["unfinished_models"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
