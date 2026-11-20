from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import json
from colab_runner.launch import run_queue, bundle_code
from colab_runner.recovery import recover
import zipfile


class RecoveryTests(unittest.TestCase):
    def test_legacy_worker_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle.zip"
            bundle_code(path, {})
            with zipfile.ZipFile(path) as archive:
                original = Path(__file__).resolve().parents[1] / "recovery/worker_launch.txt"
                self.assertEqual(archive.read("colab_runner/launch.py"), original.read_bytes())

    def test_two_confirmations(self):
        command = Mock(return_value=CompletedProcess([], 0, "[colab] No active sessions found on server."))
        self.assertEqual(recover(command, "gpu", "r", Path("r"), time.time()+10, sleep=lambda _: None), "gone")
        self.assertEqual(command.call_count, 2)

    def test_unknown_session_not_replaced(self):
        command = Mock(return_value=CompletedProcess([], 0, "[?] endpoint | Hardware: A100"))
        self.assertEqual(recover(command, "gpu", "r", Path("r"), time.time()+10), "unknown")

    def test_active_with_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "r"
            path.write_text('{"synced": true}')
            command = Mock(return_value=CompletedProcess([], 0, "[gpu] endpoint | Hardware: A100"))
            self.assertEqual(recover(command, "gpu", "r", path, time.time()+10), "finished")

    def test_lost_worker_retries_same_model_then_finishes_queue(self):
        config = {"models": ["model-a", "model-b"], "gpu": "A100", "max_total_gpu_hours": 1}
        attempted = []
        def execute(session, code):
            model = next(m for m in config["models"] if repr(m) in code)
            attempted.append((session, model))
            if session == "old":
                raise CalledProcessError(1, "exec")
        def command(args, **kwargs):
            if args[0] == "download":
                Path(args[-1]).write_text(json.dumps({"model": attempted[-1][1], "synced": True, "all_variants_completed": True}))
        with tempfile.TemporaryDirectory() as temp, patch("colab_runner.launch.colab", side_effect=command), patch("colab_runner.launch.execute", side_effect=execute), patch("colab_runner.launch.prepare_session"), patch("colab_runner.launch.publish_summary"), patch("colab_runner.recovery.recover", return_value="gone"), patch("colab_runner.recovery.replace_session", return_value="new"):
            result = run_queue(config, [("old", time.time())], "/content/code", Path(temp), bundle=Path("bundle"))
        self.assertEqual(attempted, [("old","model-a"), ("new","model-a"), ("new","model-b")])
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["models"]), 2)
