import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import run_overnight_paired_pipeline as runner
from scripts.run_overnight_paired_pipeline import build_paths, command_to_text, replace_batch_size


class OvernightPairedPipelineTests(unittest.TestCase):
    def test_build_paths_keeps_outputs_under_run_directory(self):
        paths = build_paths("20260707_2211")

        self.assertEqual(paths.run_dir, Path("outputs") / "overnight_paired_20260707_2211")
        self.assertEqual(paths.dirty_data_dir, Path("data") / "photo_action_click_all_dirty80_paired_10k_20260707_2211")
        self.assertEqual(paths.clean_data_dir, Path("data") / "photo_action_click_all_clean80_paired_10k_20260707_2211")
        self.assertEqual(paths.dirty_checkpoint, paths.run_dir / "checkpoints" / "action_resnet18_frozen_dirty80_paired_20260707_2211.pt")
        self.assertEqual(paths.clean_metrics, paths.run_dir / "clean_eval" / "test" / "metrics.json")

    def test_replace_batch_size_changes_existing_value(self):
        command = ["python", "scripts/train_action_sequence.py", "--batch-size", "32", "--epochs", "15"]

        self.assertEqual(
            replace_batch_size(command, "16"),
            ["python", "scripts/train_action_sequence.py", "--batch-size", "16", "--epochs", "15"],
        )

    def test_command_to_text_quotes_paths_with_spaces(self):
        text = command_to_text(["python", "script.py", "--path", "D:/a dir/file.txt"])

        self.assertIn('"D:/a dir/file.txt"', text)

    def test_focus_class_summary_maps_class_keys_to_metric_object_names(self):
        self.assertTrue(hasattr(runner, "build_focus_class_summary"))
        dirty = {"per_class": {"裙子": {"cell_exact_match": 0.2, "cell_recall": 0.4}}}
        clean = {"per_class": {"裙子": {"cell_exact_match": 0.5, "cell_recall": 0.7}}}

        result = runner.build_focus_class_summary(
            dirty,
            clean,
            class_name_map={"dress": "裙子"},
            class_keys=["dress"],
        )

        self.assertEqual(result["dress"]["object_name"], "裙子")
        self.assertEqual(result["dress"]["dirty_cell_exact_match"], 0.2)
        self.assertEqual(result["dress"]["clean_cell_exact_match"], 0.5)
        self.assertAlmostEqual(result["dress"]["delta_cell_exact_match"], 0.3)

    def test_expected_manifest_samples_uses_requested_split_sizes(self):
        self.assertTrue(hasattr(runner, "expected_manifest_samples"))
        args = SimpleNamespace(num_train=12, num_val=3, num_test=5)

        self.assertEqual(runner.expected_manifest_samples(args), 20)


if __name__ == "__main__":
    unittest.main()
