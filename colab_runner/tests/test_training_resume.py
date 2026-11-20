from pathlib import Path
import contextlib
import copy
import io
import json
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
from colab_runner.training import train, initial_state, record_evaluation


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))

    def forward(self, encoding, features=None):
        return self.layers(encoding)


def make_loader(shuffle):
    samples = [{"encoding": torch.tensor([i / 12, (i % 3) / 3, 1.0]), "label": torch.tensor(i % 2), "linguistic_features": None} for i in range(12)]

    def collate(batch):
        return {
            "encoding": torch.stack([item["encoding"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "linguistic_features": None,
        }

    return DataLoader(samples, batch_size=3, shuffle=shuffle, collate_fn=collate, generator=torch.Generator().manual_seed(123))


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


class TrainingTests(unittest.TestCase):
    def run_training(self, path, pause_after=None, resume=False):
        set_seed()
        model = TinyDetector()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        seen = [0]

        def should_stop():
            seen[0] += 1
            return pause_after is not None and seen[0] == pause_after

        with patch("colab_runner.training.save_model"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model, epochs = train(
                3, model, make_loader(True), make_loader(False), optimizer, torch.nn.CrossEntropyLoss(),
                priority="f1", save_path=path, eval_step=2, checkpoint_steps=1,
                resume=resume, stop_callback=should_stop, experiment_id="tiny-test",
            )
        return model, epochs, torch.load(Path(path) / "resume.pt", weights_only=False)

    def test_resume_matches_uninterrupted_with_dropout(self):
        torch.set_num_threads(1)
        for pause_after in [1, 3, 4, 5, 8]:
            with self.subTest(pause_after=pause_after), tempfile.TemporaryDirectory() as root:
                full, full_epochs, full_checkpoint = self.run_training(Path(root) / "full")
                self.run_training(Path(root) / "resumed", pause_after=pause_after)
                resumed, resumed_epochs, resumed_checkpoint = self.run_training(Path(root) / "resumed", resume=True)
                self.assertEqual(full_epochs, resumed_epochs)
                self.assertEqual(full.results, resumed.results)
                for name, tensor in full.state_dict().items():
                    self.assertTrue(torch.equal(tensor, resumed.state_dict()[name]), name)
                for name, tensor in full_checkpoint["model"].items():
                    self.assertTrue(torch.equal(tensor, resumed_checkpoint["model"][name]), name)

    def test_small_gains_do_not_reset_patience_but_best_is_retained(self):
        state = initial_state()
        state.update(sample_counter=1, loss_sum=0.2)
        with contextlib.redirect_stdout(io.StringIO()):
            for step, score in [(5, 0.8), (10, 0.802), (15, 0.804), (20, 0.806)]:
                state.update(step=step, epoch=(step - 1) // 10 + 1, next_batch=(step - 1) % 10 + 1, best_step=step)
                record_evaluation(state, {"f1": score}, "f1", 10, 2, 0.75, 0.0075)
        self.assertEqual(state["meaningful_step"], 5)
        self.assertEqual(state["best_step"], 20)
        self.assertEqual(state["best_epoch"], 2)
        self.assertEqual(state["status"], "early_stopped")

    def test_best_step_survives_early_stopping(self):
        from colab_runner.training import _check_best_model
        scores = iter([0.9, 0.8, 0.85, 0.88])
        selected = {}

        def evaluation(model, loader, criterion, best_metrics, best_state, best_step, step, **kwargs):
            score = next(scores)
            metrics = {"loss": 1 - score, "accuracy": score, "precision": score, "recall": score, "f1": score}
            if step == 2:
                selected.update({name: tensor.clone() for name, tensor in model.state_dict().items()})
            best_metrics, best_state, best_step = _check_best_model(metrics, "f1", model, best_metrics, best_state, best_step, step)
            return best_metrics, best_state, best_step, metrics

        with tempfile.TemporaryDirectory() as folder:
            set_seed()
            model = TinyDetector()
            with patch("colab_runner.training.evaluate", side_effect=evaluation), patch("colab_runner.training.save_model"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model, history = train(4, model, make_loader(True), make_loader(False), torch.optim.AdamW(model.parameters(), lr=0.01), torch.nn.CrossEntropyLoss(), priority="f1", save_path=folder, eval_step=2, patience_epochs=0.75, min_epochs=2)
            self.assertEqual(model.results["status"], "early_stopped")
            self.assertEqual(model.results["best_step"], 2)
            self.assertEqual(model.results["best_epoch"], 1)
            self.assertEqual(len(history), 2)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, selected[name]))
            self.assertEqual(json.loads((Path(folder) / "results.json").read_text())["best_step"], 2)
            self.assertEqual(list(Path(folder).glob("*.png")), [])

    def test_configuration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_training(root, pause_after=3)
            model = TinyDetector()
            with self.assertRaisesRegex(ValueError, "settings differ"):
                train(4, model, make_loader(True), make_loader(False), torch.optim.AdamW(model.parameters()), torch.nn.CrossEntropyLoss(), save_path=root, resume=True)


if __name__ == "__main__":
    unittest.main()
