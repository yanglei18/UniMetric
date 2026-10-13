"""CPU-only preprocessing regression tests; no model weights or inference."""
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as TF


ROOT = Path(__file__).resolve().parents[1]


def load_methods(filename, class_name, names, namespace, bases=()):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), filename, "exec"), namespace)
    return namespace[class_name]


class BalancedPreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import math
        ns = dict(np=np, Image=Image, torch=torch, math=math)
        load_methods("metric_scale/models/vggt_long_depth_estimator.py", "VGGTLongDepthEstimator",
                     {"_load_chunk_images_resized", "_next_multiple"}, ns)
        cls.estimator = load_methods("metric_scale/models/vggt_omega_long_depth_estimator.py",
                                    "VGGTOmegaLongDepthEstimator",
                                    {"_compute_target_size", "_load_chunk_images_resized"}, ns,
                                    ("VGGTLongDepthEstimator",))

    def make_estimator(self, mode="balanced"):
        obj = self.estimator()
        obj.preprocess_mode, obj.image_resolution, obj.patch_size = mode, 512, 16
        obj.to_tensor = TF.ToTensor()
        return obj

    def test_sizes_and_legacy_limits(self):
        obj = self.make_estimator()
        self.assertEqual(obj._compute_target_size(800, 1280, 1, 16), (400, 640))
        self.assertEqual(obj._compute_target_size(1280, 800, None), (640, 400))
        self.assertEqual(self.make_estimator("resize_only")._compute_target_size(800, 1280, 255000, 512), (320, 512))
        for h, w in [(1, 100), (100, 1), (0, 800)]:
            with self.assertRaises(ValueError):
                obj._compute_target_size(h, w, None)

    def test_pixels_and_metadata(self):
        obj = self.make_estimator()
        with tempfile.TemporaryDirectory(prefix="omega_preprocess_test_") as directory:
            path = str(Path(directory) / "rgba.png")
            array = np.random.default_rng(42).integers(0, 256, (800, 1280, 4), dtype=np.uint8)
            Image.fromarray(array).save(path)
            tensor, metas = obj._load_chunk_images_resized([path], (400, 640))
            rgb = Image.alpha_composite(Image.new("RGBA", (1280, 800), (255, 255, 255, 255)),
                                       Image.fromarray(array)).convert("RGB")
            expected = TF.ToTensor()(rgb.resize((640, 400), Image.Resampling.BICUBIC))
            self.assertTrue(torch.equal(tensor[0], expected))
            self.assertEqual((metas[0]["orig_h"], metas[0]["orig_w"]), (800, 1280))
            self.assertEqual((metas[0]["infer_h"], metas[0]["infer_w"]), (400, 640))
            with self.assertRaises(ValueError):
                obj._load_chunk_images_resized([path], (320, 512))
            legacy, _ = self.make_estimator("resize_only")._load_chunk_images_resized([path], (320, 512))
            expected_legacy = TF.ToTensor()(Image.fromarray(array).convert("RGB").resize((512, 320), Image.Resampling.LANCZOS))
            self.assertTrue(torch.equal(legacy[0], expected_legacy))
            official_path = os.environ.get("OMEGA_OFFICIAL_LOADER")
            if official_path:
                spec = importlib.util.spec_from_file_location("omega_official_loader", official_path)
                official = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(official)
                self.assertTrue(torch.equal(tensor, official.load_and_preprocess_images([path], mode="balanced")))
                print("Official balanced loader: exact tensor equality verified")


if __name__ == "__main__":
    unittest.main()
