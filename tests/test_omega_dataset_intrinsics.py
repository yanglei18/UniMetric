"""CPU synthetic geometry tests. Never load model weights or run a real model."""
import filecmp
import gc
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as TF
import yaml

from test_omega_balanced_preprocess import ROOT, load_methods


class DatasetIntrinsicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = dict(np=np, os=os, re=re, yaml=yaml, filecmp=filecmp, Image=Image,
                  torch=torch, gc=gc, math=math, cv2=cv2, json=json)
        ns['ensure_dir'] = lambda path: os.makedirs(path, exist_ok=True)
        cls.config = SimpleNamespace(depth_output_resolution='native')
        ns['config'] = cls.config
        exec(compile((ROOT/'metric_scale/models/output_metadata.py').read_text(), 'metadata', 'exec'), ns)
        load_methods('metric_scale/models/vggt_long_depth_estimator.py', 'VGGTLongDepthEstimator',
                     {'_rescale_intrinsics', '_depth_to_pointmap', '_resize_frame_output', '_frame_prefix',
                      '_sample_valid_world_points', '_estimate_weighted_sim3', '_estimate_adjacent_chunk_sim3'}, ns)
        cls.estimator = load_methods('metric_scale/models/vggt_omega_long_depth_estimator.py',
                                    'VGGTOmegaLongDepthEstimator',
                                    {'_prepare_dataset_intrinsics', '_compute_target_size', '_load_chunk_images_resized',
                                     '_run_chunk_inference', '_w2c_to_c2w', '_depth_to_world_points',
                                     '_scalar_int', '_load_frame_prediction', '_apply_sim3_and_export_frame', 'save_results'},
                                    ns, ('VGGTLongDepthEstimator',))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='omega_dataset_k_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.obj = self.estimator()
        self.obj.geometry_intrinsics_source = 'dataset'
        self.obj.preprocess_mode = 'balanced'
        self.obj.image_resolution = 32
        self.obj.patch_size = 16
        self.obj.to_tensor = TF.ToTensor()
        self.obj.device = 'cpu'
        self.obj.delete_temp_files = False
        self.obj.max_points_per_frame = 5000
        self.images = []
        self.k = np.array([[100., 0, 35.], [0, 110., 22.], [0, 0, 1.]])
        for fid in ['000001', '000002']:
            p = self.root/(fid+'_camera1.png')
            Image.new('RGB', (80, 48), (12, 23, 45)).save(p)
            self.images.append(str(p))
            (self.root/(fid+'.yaml')).write_text(yaml.safe_dump({'camera1': {'intrinsic': self.k.tolist()}}))

    def prepare(self):
        self.obj._prepare_dataset_intrinsics(self.images, [(48,80)]*2, (32,48), str(self.root), ['000001','000002'])

    def test_preflight_and_scaling(self):
        self.prepare()
        expected = np.diag([48/80, 32/48, 1]) @ self.k
        np.testing.assert_allclose(self.obj._dataset_intrinsics[self.images[0]]['intrinsic'], expected, rtol=1e-6)
        with self.assertRaises(ValueError):
            self.obj._prepare_dataset_intrinsics(self.images, [(48,80)]*2, (32,48), None)
        with self.assertRaises(ValueError):
            self.obj._prepare_dataset_intrinsics(self.images, [(48,80)]*2, (32,48), str(self.root), ['000002','000001'])
        (self.root/'000001.yaml').write_text('camera1: {intrinsic: [[0, 0, 0], [0, 1, 0], [0, 0, 1]]}')
        with self.assertRaises(ValueError): self.prepare()
        self.obj.geometry_intrinsics_source = 'predicted'
        self.obj._prepare_dataset_intrinsics(self.images, [], (32,48), None)
        self.assertEqual(self.obj._dataset_intrinsics, {})

    def test_changed_rgb_and_missing_calibration_rejected(self):
        copied = self.root/'copied'; copied.mkdir()
        p = copied/'000001_camera1.png'
        Image.new('RGB', (80,48), (99,99,99)).save(p)
        with self.assertRaises(ValueError):
            self.obj._prepare_dataset_intrinsics([str(p)], [(48,80)], (32,48), str(self.root))
        (self.root/'000001.yaml').unlink()
        with self.assertRaises(FileNotFoundError): self.prepare()

    def fake_outputs(self, images):
        n, _, h, w = images.shape
        depth = torch.arange(h*w, dtype=torch.float32).reshape(1,1,h,w,1)/(h*w)+2
        return {'images': images, 'depth': depth.repeat(1,n,1,1,1),
                'depth_conf': torch.ones(1,n,h,w,1), 'pose_enc': n}

    def fake_cameras(self, n, shape):
        ext = torch.eye(4)[:3].reshape(1,1,3,4).repeat(1,n,1,1)
        ext[:,:,0,3] = .1
        k = torch.tensor([[120.,0,24],[0,120.,16],[0,0,1]]).reshape(1,1,3,3).repeat(1,n,1,1)
        return ext, k

    def test_chunk_alignment_export_and_metadata(self):
        self.prepare()
        self.obj.model = self.fake_outputs
        self.obj.encoding_to_camera = self.fake_cameras
        records = self.obj._run_chunk_inference(0, self.images, 0, str(self.root/'tmp'), (32,48))
        pred = self.obj._load_frame_prediction(records[0]['pred_path'])
        self.assertEqual(pred['geometry_intrinsics_source'], 'dataset')
        expected_camera = self.obj._depth_to_pointmap(pred['depth'], pred['intrinsic'])
        expected_world = expected_camera.copy(); expected_world[:,:,0] -= .1
        np.testing.assert_allclose(pred['world_points'], expected_world, atol=1e-6)
        self.assertNotEqual(float(pred['intrinsic'][0,0]), float(pred['network_intrinsic'][0,0]))
        # Identity overlap uses the reprojected world points saved by the chunk path.
        sim = self.obj._estimate_adjacent_chunk_sim3(records, records, [0,1])
        self.assertAlmostEqual(sim['s'], 1., places=5)
        np.testing.assert_allclose(sim['R'], np.eye(3), atol=1e-5)
        rotation = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])
        sim = {'s': 2., 'R': rotation, 't': np.array([1.,2.,3.])}
        exported = self.obj._apply_sim3_and_export_frame(pred, sim)
        w2c = exported['extrinsics']
        world = (exported['pointmap']-w2c[:3,3])@w2c[:3,:3]
        np.testing.assert_allclose(world, 2*pred['world_points']@rotation.T+sim['t'], atol=1e-5)
        manifest = {'frame_records':records, 'chunk_transforms':[sim]}
        for mode, shape in [('native',(32,48)),('original',(48,80))]:
            self.config.depth_output_resolution = mode
            out = self.root/mode
            self.obj.save_results(manifest, str(out), frame_ids=['000001','000002'])
            files = list((out/'camera_params').glob('*.json'))
            meta = json.loads(files[0].read_text())
            self.assertEqual(meta['geometry_intrinsics_source'], 'dataset')
            np.testing.assert_allclose(meta['intrinsics_original'], self.k, rtol=1e-6)
            self.assertEqual(tuple(meta['output_resolution']), shape)
            self.assertIn('network_intrinsics_inference', meta)

    def test_predicted_path_preserves_network_geometry_single_frame(self):
        self.obj.geometry_intrinsics_source = 'predicted'
        self.obj._dataset_intrinsics = {}
        self.obj.model = self.fake_outputs
        self.obj.encoding_to_camera = self.fake_cameras
        records = self.obj._run_chunk_inference(0, self.images[:1], 0, str(self.root/'tmp'), (32,48))
        pred = self.obj._load_frame_prediction(records[0]['pred_path'])
        np.testing.assert_array_equal(pred['intrinsic'], pred['network_intrinsic'])
        self.assertEqual(pred['depth'].shape, (32,48))
        self.assertEqual(pred['geometry_intrinsics_source'], 'predicted')


if __name__ == '__main__':
    unittest.main()
