"""Import-boundary tests with stubbed numerical libraries; no GPU or weights."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SETUP = """
import importlib
import importlib.abc
import sys
from types import ModuleType
from unittest.mock import MagicMock

# Stub unrelated dependencies so this checks the real adapter modules even on
# machines without the scientific stack. Never stub VGGT in the import test.
for name in ('cv2', 'numpy', 'torch', 'PIL', 'torchvision', 'yaml',
             'metric_scale.config.config', 'metric_scale.config.path_config',
             'metric_scale.utils.utils'):
    sys.modules[name] = MagicMock()
sys.modules['numpy'].ndarray = type('ndarray', (), {})
sys.modules['torch'].Tensor = type('Tensor', (), {})
sys.modules['torch'].device = type('device', (), {})
sys.modules['torch'].nn.Module = type('Module', (), {})

class BlockVGGT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vggt' or fullname.startswith('vggt.'):
            raise ModuleNotFoundError('VGGT intentionally unavailable', name=fullname)

sys.meta_path.insert(0, BlockVGGT())
from metric_scale.models.vggt_long_depth_estimator import VGGTLongDepthEstimator
"""


@unittest.skipIf(sys.version_info < (3, 10), 'Adapters require Python 3.10+ annotations')
class GFMDependencyIsolationTests(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run(
            [sys.executable, '-c', SETUP + textwrap.dedent(code)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_helpers_and_adapters_import_without_vggt(self):
        self.run_isolated("""
            for module_name, class_name in (
                ('dvlt', 'DVLTLongDepthEstimator'),
                ('vggt_omega', 'VGGTOmegaLongDepthEstimator'),
                ('fourrc', 'FourRCLongDepthEstimator'),
                ('stream3r', 'STream3RLongDepthEstimator'),
            ):
                module = importlib.import_module(
                    'metric_scale.models.' + module_name + '_long_depth_estimator')
                cls = getattr(module, class_name)
                assert issubclass(cls, VGGTLongDepthEstimator)
                obj = cls.__new__(cls)
                obj.chunk_size, obj.overlap = 60, 20
                assert obj._build_chunk_indices(101) == [(0, 60), (40, 100), (80, 101)]
                assert obj._frame_prefix(0, ['frame']) == 'frame_camera1'
            assert not any(n == 'vggt' or n.startswith('vggt.') for n in sys.modules)
        """)

    def test_vggt_constructor_requires_its_backend(self):
        self.run_isolated("""
            try:
                VGGTLongDepthEstimator(device='cpu')
            except ModuleNotFoundError as exc:
                assert exc.name == 'vggt'
            else:
                raise AssertionError('VGGT constructor did not import its backend')
        """)

    def test_vggt_constructor_loads_requested_checkpoint(self):
        self.run_isolated("""
            backend = ModuleType('vggt.models.vggt')
            backend.VGGT = MagicMock()
            sys.modules['vggt.models.vggt'] = backend
            obj = VGGTLongDepthEstimator(pretrained_model='test/checkpoint', device='cpu')
            backend.VGGT.from_pretrained.assert_called_once_with('test/checkpoint')
            backend.VGGT.from_pretrained.return_value.to.assert_called_once_with('cpu')
            assert obj.model is backend.VGGT.from_pretrained.return_value.to.return_value.eval.return_value
        """)

    def test_vggt_inference_uses_lazy_pose_decoder(self):
        self.run_isolated("""
            class Decoded(Exception):
                pass
            backend = ModuleType('vggt.utils.pose_enc')
            backend.pose_encoding_to_extri_intri = MagicMock(side_effect=Decoded)
            sys.modules['vggt.utils.pose_enc'] = backend
            obj = VGGTLongDepthEstimator.__new__(VGGTLongDepthEstimator)
            obj.device, obj.device_type, obj.dtype = 'cpu', 'cpu', 'float32'
            images = MagicMock()
            obj._load_chunk_images_resized = MagicMock(return_value=(images, []))
            obj.model = MagicMock()
            obj.model.aggregator.return_value = ('tokens', 0)
            obj.model.camera_head.return_value = ['pose']
            try:
                obj._run_chunk_inference(0, [], 0, None, (28, 28))
            except Decoded:
                pass
            else:
                raise AssertionError('VGGT inference did not invoke its decoder')
            backend.pose_encoding_to_extri_intri.assert_called_once_with(
                'pose', images.to.return_value.__getitem__.return_value.shape[-2:])
        """)


if __name__ == '__main__':
    unittest.main()
