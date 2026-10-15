"""CPU-only tests for restoring bundled VLM results."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from metric_scale.processing.precomputed_vlm import (
    ARCHIVE_RELATIVE_PATH,
    restore_precomputed_vlm_results,
)


class PrecomputedVLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.output = self.project / "outputs"
        self.archive = self.project / ARCHIVE_RELATIVE_PATH
        self.archive.parent.mkdir(parents=True)

    def make_archive(self, extra=None):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for dataset, sequence in (("nuScenes", "scene1"), ("nuScenes", "scene2"), ("KITTI", "scene1")):
                archive.writestr(
                    f"{dataset}/{sequence}/vlm_outputs/results/001_camera1.json",
                    json.dumps({"dataset": dataset, "sequence": sequence}),
                )
            if extra:
                archive.writestr(extra, "{}")

    def result(self, dataset="nuScenes", sequence="scene1"):
        return self.output / dataset / sequence / "vlm_outputs/results/001_camera1.json"

    def test_dataset_filter_and_repeat(self):
        self.make_archive()
        self.assertEqual(restore_precomputed_vlm_results(self.project, self.output, "nuScenes"), 2)
        self.assertTrue(self.result(sequence="scene2").is_file())
        self.assertFalse(self.result(dataset="KITTI").exists())
        self.assertEqual(restore_precomputed_vlm_results(self.project, self.output, "nuScenes"), 0)

    def test_sequence_filter(self):
        self.make_archive()
        self.assertEqual(restore_precomputed_vlm_results(self.project, self.output, "nuScenes", ["scene1"]), 1)
        self.assertEqual(json.loads(self.result().read_text())["sequence"], "scene1")
        self.assertFalse(self.result(sequence="scene2").exists())

    def test_existing_cache_is_preserved(self):
        self.make_archive()
        existing = self.result()
        existing.parent.mkdir(parents=True)
        existing.write_text("user-owned cache")
        self.assertEqual(restore_precomputed_vlm_results(self.project, self.output, "nuScenes"), 1)
        self.assertEqual(existing.read_text(), "user-owned cache")

    def test_missing_archive(self):
        self.assertEqual(restore_precomputed_vlm_results(self.project, self.output, "nuScenes"), 0)
        self.assertFalse(self.output.exists())

    def test_unsafe_paths_are_rejected_before_restoring(self):
        for path in ("../escaped.json", "/absolute.json", "nuScenes/../escaped.json", "C:/escaped.json", "..\\escaped.json"):
            with self.subTest(path=path):
                self.make_archive(path)
                with self.assertRaises(ValueError):
                    restore_precomputed_vlm_results(self.project, self.output, "nuScenes")
                self.assertFalse(self.output.exists())

    def test_symlink_parent_cannot_escape_output_root(self):
        self.make_archive()
        outside = self.project / "outside"
        outside.mkdir()
        self.output.mkdir()
        (self.output / "nuScenes").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            restore_precomputed_vlm_results(self.project, self.output, "nuScenes")
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
