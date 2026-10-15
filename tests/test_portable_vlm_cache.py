"""CPU-only cache and relocation checks; no model imports or API credentials."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_methods(path, name, methods, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class PortableVLMCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        namespace = {"os": os}
        cls = load_methods("metric_scale/config/config.py", "Config",
                           {"resolve_vlm_image_path", "reuse_vlm_results", "resume_existing_results"}, namespace)
        self.config = cls()
        self.runtime = {}
        self.config._get_section = lambda name: self.runtime
        self.config.project_root = str(self.root)
        self.config.dataset = "nuscenes"
        self.config.data_path = "data/metric_bench/nuScenes"
        self.image = self.root / self.config.data_path / "scene1" / "001_camera1.jpg"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(b"image bytes")
        self.portable_path = self.image.relative_to(self.root).as_posix()

    def test_default_only_reuses_vlm(self):
        self.assertTrue(self.config.reuse_vlm_results)
        self.assertFalse(self.config.resume_existing_results)

    def test_project_relative_path_is_independent_of_cwd(self):
        self.assertEqual(self.config.resolve_vlm_image_path(self.portable_path, "scene1"), str(self.image))

    def test_custom_data_root_and_legacy_path_relocation(self):
        self.assertEqual(self.config.resolve_vlm_image_path(
            "/old/server/data/metric_bench/nuScenes/scene1/001_camera1.jpg", "scene1"), str(self.image))
        self.config.data_path = str(self.root / "custom_root")
        relocated = Path(self.config.data_path) / "scene1" / self.image.name
        relocated.parent.mkdir(parents=True)
        relocated.write_bytes(b"custom image")
        self.image.unlink()
        self.assertEqual(self.config.resolve_vlm_image_path(self.portable_path, "scene1"), str(relocated))

    def test_wrong_dataset_and_sequence_are_rejected(self):
        for path in (self.portable_path.replace("nuScenes", "KITTI"),
                     self.portable_path.replace("scene1", "scene2")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.config.resolve_vlm_image_path(path, "scene1")

    def test_missing_or_unidentified_path_does_not_guess(self):
        for path in ("001_camera1.jpg", "/old/KITTI/scene1/001_camera1.jpg",
                     self.portable_path.replace("001_", "002_")):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                self.config.resolve_vlm_image_path(path, "scene1")

    def load_processor(self):
        self.config.enable_visualization = False
        self.config.get_path = lambda key, seq: str(self.root / key / seq)
        namespace = {"os": os, "config": self.config, "_device_prefix": lambda _: "cpu",
                     "file_utils": SimpleNamespace(ensure_directory=lambda path: os.makedirs(path, exist_ok=True))}
        processor = load_methods("metric_scale/processing/processing.py", "VLMProcessor",
                                 {"process_frames", "_vlm_result_path"}, namespace)
        processor._is_api_vlm_mode = Mock(side_effect=RuntimeError("inference requested"))
        results = Path(self.config.get_path("vlm_results", "scene1"))
        results.mkdir(parents=True)
        (results / "001_camera1.json").write_text(json.dumps({"image_path": self.portable_path}))
        return processor

    def test_full_cache_hit_never_initializes_vlm(self):
        processor = self.load_processor()
        processor.process_frames(str(self.image.parent), "scene1", ["001"])
        processor._is_api_vlm_mode.assert_not_called()

    def test_resume_still_reuses_vlm_when_reuse_flag_disabled(self):
        processor = self.load_processor()
        self.runtime.update(reuse_vlm_results=False, resume_existing_results=True)
        processor.process_frames(str(self.image.parent), "scene1", ["001"])
        processor._is_api_vlm_mode.assert_not_called()

    def test_missing_frame_reaches_inference(self):
        processor = self.load_processor()
        with self.assertRaisesRegex(RuntimeError, "inference requested"):
            processor.process_frames(str(self.image.parent), "scene1", ["001", "002"])

    def test_explicit_disable_reaches_inference(self):
        processor = self.load_processor()
        self.runtime["reuse_vlm_results"] = False
        with self.assertRaisesRegex(RuntimeError, "inference requested"):
            processor.process_frames(str(self.image.parent), "scene1", ["001"])

    def test_p3_copies_project_relative_image(self):
        self.config.get_path = lambda key, seq: str(self.root / key / seq)
        vlm_dir = self.root / "vlm"
        vlm_dir.mkdir()
        (vlm_dir / "001_camera1.json").write_text(json.dumps({"image_path": self.portable_path}))
        namespace = {"os": os, "config": self.config,
                     "file_utils": SimpleNamespace(
                         ensure_directory=lambda path: os.makedirs(path, exist_ok=True),
                         read_json=lambda path: json.loads(Path(path).read_text()))}
        preparer = load_methods("metric_scale/processing/processing.py", "P3ImagePreparer", {"prepare_images"}, namespace)
        preparer.prepare_images(str(vlm_dir), "scene1")
        self.assertEqual((self.root / "p3_images/scene1/001_camera1.jpg").read_bytes(), self.image.read_bytes())

    def test_sam_opens_resolved_image(self):
        self.config.enable_visualization = False
        self.config.get_path = lambda key, seq: str(self.root / key / seq)
        vlm_dir = self.root / "vlm"
        vlm_dir.mkdir()
        (vlm_dir / "001_camera1.json").write_text(json.dumps({"image_path": self.portable_path}))
        open_image = Mock(side_effect=RuntimeError("image opened"))
        namespace = {"os": os, "config": self.config,
                     "Image": SimpleNamespace(open=open_image), "model_manager": Mock(),
                     "file_utils": SimpleNamespace(
                         ensure_directory=lambda path: os.makedirs(path, exist_ok=True),
                         read_json=lambda path: json.loads(Path(path).read_text()))}
        sam = load_methods("metric_scale/processing/processing.py", "SAMProcessor", {"process_segmentation"}, namespace)
        with self.assertRaisesRegex(RuntimeError, "image opened"):
            sam.process_segmentation(str(vlm_dir), "scene1", ["001"])
        open_image.assert_called_once_with(str(self.image))

    def test_p3_ignores_unselected_cached_frame_without_image(self):
        self.config.get_path = lambda key, seq: str(self.root / key / seq)
        vlm_dir = Path(self.config.get_path("vlm_results", "scene1"))
        vlm_dir.mkdir(parents=True)
        for frame_id in ("001", "002"):
            (vlm_dir / f"{frame_id}_camera1.json").write_text(json.dumps({
                "image_path": self.portable_path.replace("001_", f"{frame_id}_")}))
        namespace = {"os": os, "config": self.config, "_device_prefix": lambda _: "cpu",
                     "timer": Mock(), "DepthProcessor": Mock(),
                     "file_utils": SimpleNamespace(
                         ensure_directory=lambda path: os.makedirs(path, exist_ok=True),
                         read_json=lambda path: json.loads(Path(path).read_text()))}
        load_methods("metric_scale/processing/processing.py", "P3ImagePreparer", {"prepare_images"}, namespace)
        pipeline_cls = load_methods("metric_scale/processing/processing.py", "ProcessingPipeline",
                                    {"_prepare_p3_for_sequence"}, namespace)
        pipeline = pipeline_cls()
        pipeline._get_sequence_frame_ids = Mock(return_value=["001"])
        pipeline._prepare_p3_for_sequence("scene1", sequence_path=str(self.image.parent))
        pipeline._get_sequence_frame_ids.assert_called_once_with(str(self.image.parent))
        self.assertEqual((self.root / "p3_images/scene1/001_camera1.jpg").read_bytes(), self.image.read_bytes())
        self.assertFalse((self.root / "p3_images/scene1/002_camera1.jpg").exists())

    def test_pipeline_restores_current_sequences_before_vlm(self):
        self.config.dataset_name = "nuScenes"
        self.config.get_path = lambda key: str(self.root / "outputs")
        events = []
        restore = Mock(side_effect=lambda *args: events.append("restore") or 1)
        namespace = {"config": self.config, "restore_precomputed_vlm_results": restore,
                     "model_manager": Mock()}
        pipeline_cls = load_methods("metric_scale/processing/processing.py", "ProcessingPipeline", {"run"}, namespace)
        pipeline = pipeline_cls()
        pipeline.sequences = [(str(self.image.parent), "scene1")]
        pipeline.prepare_sequences = lambda: events.append("prepare")
        pipeline.process_vlm = lambda: events.append("vlm")
        pipeline._should_parallelize_sequences = lambda: False
        pipeline._process_sequence_task = lambda *args: events.append("depth")
        pipeline.run()
        self.assertEqual(events, ["prepare", "restore", "vlm", "depth"])
        restore.assert_called_once_with(str(self.root), str(self.root / "outputs"), "nuScenes", ["scene1"])


if __name__ == "__main__":
    unittest.main()
