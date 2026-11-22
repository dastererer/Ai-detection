from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import torch
from colab_runner.data import prepare_data, make_datasets
from colab_runner.storage import copy_verified
from colab_runner.launch import bundle_code, execute_worker, run_queue


class Tokenizer:
    sep_token_id = 2
    eos_token_id = None

    def __len__(self):
        return 100


class DataTests(unittest.TestCase):
    def test_overlap_removed_and_scaler_uses_train_only(self):
        rows = []
        for split, offset in [("train", 0), ("validate", 100), ("test", 200)]:
            for label in [0, 1]:
                rows.append({"text": f"{split}-{label}", "label": label, "split": split, "input_ids": [1, 4, 5, 6, 2, 0], "attention_mask": [1, 1, 1, 1, 1, 0], "feature": offset + label * 2})
        rows.append({**rows[0], "split": "test"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.parquet"
            pd.DataFrame(rows).to_parquet(path)
            data, report = prepare_data(path, ["feature"])
        self.assertEqual(len(data), 6)
        self.assertEqual(len(report["excluded_rows"]), 1)
        self.assertEqual(report["excluded_rows"][0]["row_id"], 6)
        datasets, processor = make_datasets(data, Tokenizer(), ["feature"], 4)
        self.assertEqual(processor.named_steps["standardscaler"].mean_.tolist(), [1.0])
        self.assertEqual(datasets["train"][0]["input_ids"], [1, 4, 5, 2])
        self.assertEqual(datasets["train"][0]["attention_mask"], [1, 1, 1, 1])
        self.assertEqual(datasets["train"][0]["linguistic_features"].tolist(), [-1.0])
        self.assertEqual(datasets["test"][0]["linguistic_features"].tolist(), [199.0])
        datasets, processor = make_datasets(data, Tokenizer(), None, 4)
        self.assertIsNone(processor)
        self.assertIsNone(datasets["train"][0]["linguistic_features"])

    def test_failed_copy_does_not_replace_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "target"
            source.write_text("new")
            target.write_text("old")
            with patch("colab_runner.storage.file_hash", side_effect=["one", "two"]):
                with self.assertRaises(IOError):
                    copy_verified(source, target)
            self.assertEqual(target.read_text(), "old")
            copy_verified(source, target)
            self.assertEqual(target.read_text(), "new")

    def test_worker_error_reaches_kernel_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(output):
            with patch("colab_runner.launch.execute", side_effect=lambda session, code: exec(code, {})):
                with self.assertRaisesRegex(RuntimeError, "status 1: ValueError: worker diagnostic"):
                    execute_worker("gpu", ["-c", "print('started'); raise ValueError('worker diagnostic')"], folder)
        self.assertIn("started\nTraceback", output.getvalue())
        self.assertIn("ValueError: worker diagnostic", output.getvalue())

    def test_queue_assigns_each_model_once_and_stops_successful_sessions(self):
        import time
        import threading
        config = {"models": ["model-a", "model-b", "model-c"], "max_total_gpu_hours": 1, "drive_project": "/content/drive/project", "output_folder": "runs", "run_name": "test"}
        receipts = {}
        assigned = []
        stopped = []
        lock = threading.Lock()

        def worker(session, command, remote_root):
            model = command[command.index("--model") + 1]
            name = Path(command[command.index("--receipt") + 1]).name
            with lock:
                assigned.append(model)
                receipts[name] = {"model": model, "synced": True, "all_variants_completed": True, "variants": []}

        def command(arguments, **kwargs):
            if arguments[0] == "download":
                Path(arguments[-1]).write_text(json.dumps(receipts[Path(arguments[-2]).name]))
            elif arguments[0] == "stop":
                stopped.append(arguments[-1])

        with tempfile.TemporaryDirectory() as folder:
            with patch("colab_runner.launch.execute_worker", side_effect=worker), patch("colab_runner.launch.colab", side_effect=command):
                result = run_queue(config, [("gpu-a", time.time()), ("gpu-b", time.time())], "/content/code", Path(folder))
        self.assertCountEqual(assigned, config["models"])
        self.assertCountEqual(stopped, ["gpu-a", "gpu-b"])
        self.assertEqual(result["sessions_left_running"], [])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["unstarted_models"], [])

    def test_queue_failure_preserves_session_for_recovery(self):
        config = {"models": ["model-a"], "max_total_gpu_hours": 1}
        with tempfile.TemporaryDirectory() as folder:
            with patch("colab_runner.launch.execute", side_effect=RuntimeError("Disconnected")), patch("colab_runner.launch.colab") as command:
                import time
                result = run_queue(config, [("session-a", time.time())], "/content/code", Path(folder))
            self.assertEqual(result["sessions_left_running"], ["session-a"])
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
