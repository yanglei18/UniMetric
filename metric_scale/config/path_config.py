#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp

from metric_scale.config.config import config, ensure_dir


PROJECT_ROOT = config.project_root


def normalize_dataset_name(dataset_name):
    if not dataset_name:
        return None
    normalized = osp.basename(osp.normpath(str(dataset_name)))
    return normalized or None


def _resolve_path(path_name, sequence_name=None, dataset_name=None):
    template = config._get_section("paths").get(path_name)
    if template is None:
        return None

    extra_context = {}
    if sequence_name is not None:
        extra_context["sequence_name"] = sequence_name
    if dataset_name is not None:
        extra_context["dataset_name"] = normalize_dataset_name(dataset_name)
    return config._resolve_template(template, extra_context=extra_context)


def get_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("output", sequence_name, dataset_name)


def get_vlm_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vlm_output", sequence_name, dataset_name)


def get_vlm_results_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vlm_results", sequence_name, dataset_name)


def get_vlm_visualization_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vlm_visualization", sequence_name, dataset_name)


def get_sam_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("sam_output", sequence_name, dataset_name)


def get_sam_visualization_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("sam_visualization", sequence_name, dataset_name)


def get_sam_aligned_anchor_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("sam_aligned_anchor", sequence_name, dataset_name)


def get_da3_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("da3_output", sequence_name, dataset_name)


def get_pi3_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("pi3_output", sequence_name, dataset_name)


def get_vggt_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vggt_output", sequence_name, dataset_name)


def get_vggt_omega_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vggt_omega_output", sequence_name, dataset_name)


def get_dvlt_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("dvlt_output", sequence_name, dataset_name)


def get_fourrc_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("fourrc_output", sequence_name, dataset_name)


def get_stream3r_output_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("stream3r_output", sequence_name, dataset_name)


def get_vggt_relative_scale_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vggt_relative_scale", sequence_name, dataset_name)


def get_vggt_absolute_scale_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vggt_absolute_scale", sequence_name, dataset_name)


def get_vggt_depth_visualization_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("vggt_depth_visualization", sequence_name, dataset_name)


def get_da3_relative_scale_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("da3_relative_scale", sequence_name, dataset_name)


def get_da3_absolute_scale_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("da3_absolute_scale", sequence_name, dataset_name)


def get_da3_depth_visualization_dir(sequence_name=None, dataset_name=None):
    return _resolve_path("da3_depth_visualization", sequence_name, dataset_name)


def get_p3_images_dir(sequence_name=None):
    return _resolve_path("p3_images", sequence_name=sequence_name)


def get_frame_image_path(data_dir, camera_id, frame_id):
    return osp.join(data_dir, f"{frame_id}_camera{camera_id}.jpg")


def get_frame_lidar_path(data_dir, frame_id):
    return osp.join(data_dir, f"{frame_id}.pcd")


def get_frame_yaml_path(data_dir, frame_id):
    return osp.join(data_dir, f"{frame_id}.yaml")


def get_pretrained_model_dir():
    return _resolve_path("pretrained_model_dir")


def get_qwen_model_path():
    return _resolve_path("qwen_model")


def get_sam_config_path():
    return _resolve_path("sam_config")


def get_sam_checkpoint_path():
    return _resolve_path("sam_checkpoint")
