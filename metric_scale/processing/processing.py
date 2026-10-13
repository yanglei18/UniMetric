import os
import multiprocessing as mp
import queue
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
import cv2
from PIL import Image
import torch
from metric_scale.processing.frame_loader import FrameLoader
from metric_scale.config.config import config
from metric_scale.models.models import model_manager
from metric_scale.models.mllm_agent import ModelConfig, get_model
from metric_scale.models.vlm_semantic_scale_anchor import VLMSemanticScaleAnchor
from metric_scale.visualization.visualization import DepthVisualizer
from metric_scale.processing.cuda_isolated_worker import (
    run_depth_tasks_on_gpu,
    run_sam_tasks_on_gpu,
    run_sequence_pipeline_on_gpu,
)
from metric_scale.utils.utils import timer, file_utils, data_utils, sequence_utils


def _device_prefix(device=None):
    return f"[{device or 'main'}]"


def _terminate_processes(processes, label):
    if not processes:
        return

    print(f"\nStopping {label} worker process(es)...")
    for process in processes:
        if process.is_alive():
            process.terminate()

    for process in processes:
        process.join(timeout=5.0)

    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()

    for process in processes:
        if process.is_alive():
            process.join(timeout=1.0)


def _run_depth_tasks_on_gpu(gpu_id, task_payloads, result_queue):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        for task in task_payloads:
            task_start = time.time()
            DepthProcessor.run_depth_estimation(
                task["sequence_name"],
                task["frame_ids"],
                start_index=task["start_index"],
                model_type=task["model_type"],
                device=device,
                sequence_path=task.get("sequence_path"),
            )
            model_manager.release_model(f"depth_{task['model_type']}_{device}")
            result_queue.put(
                {
                    "status": "ok",
                    "gpu_id": gpu_id,
                    "sequence_name": task["sequence_name"],
                    "subsequence_index": task["subsequence_index"],
                    "subsequence_total": task["subsequence_total"],
                    "duration": time.time() - task_start,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        model_manager.release_all()


def _run_sam_tasks_on_gpu(gpu_id, task_payloads, result_queue):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        for task in task_payloads:
            task_start = time.time()
            SAMProcessor.process_segmentation(
                task["vlm_results_dir"],
                task["sequence_name"],
                frame_ids=task["frame_ids"],
                device=device,
            )
            model_manager.release_model(f"sam_tracker_{device}")
            result_queue.put(
                {
                    "status": "ok",
                    "gpu_id": gpu_id,
                    "sequence_name": task["sequence_name"],
                    "subsequence_index": task["subsequence_index"],
                    "subsequence_total": task["subsequence_total"],
                    "duration": time.time() - task_start,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        model_manager.release_all()


def _run_sequence_pipeline_on_gpu(gpu_id, task_payloads, result_queue):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        pipeline = ProcessingPipeline()
        device_suffix = "" if device in (None, "cpu") else f"_{device}"

        # Run SAM/P3 for every assigned sequence first, then free SAM before Pi3X.
        # Sharing both models on a 24GB GPU at DROID resolution (672x378, chunk 40+) OOMs.
        for task in task_payloads:
            if pipeline._existing_sequence_metric_path(task["sequence_name"]):
                continue
            pipeline._process_sam_for_sequence(
                task["sequence_path"],
                task["sequence_name"],
                device=device,
            )
            pipeline._prepare_p3_for_sequence(task["sequence_name"], device=device)

        model_manager.release_model(f"sam_tracker{device_suffix}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for task in task_payloads:
            task_start = time.time()
            if pipeline._existing_sequence_metric_path(task["sequence_name"]):
                pipeline._finalize_sequence_metrics_and_cleanup(
                    task["sequence_path"],
                    task["sequence_name"],
                    device=device,
                )
            else:
                pipeline._process_depth_for_sequence(
                    task["sequence_path"],
                    task["sequence_name"],
                    device=device,
                )
                pipeline._process_scale_for_sequence(
                    task["sequence_path"],
                    task["sequence_name"],
                    device=device,
                )
                pipeline._finalize_sequence_metrics_and_cleanup(
                    task["sequence_path"],
                    task["sequence_name"],
                    device=device,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result_queue.put(
                {
                    "status": "ok",
                    "gpu_id": gpu_id,
                    "sequence_name": task["sequence_name"],
                    "duration": time.time() - task_start,
                }
            )
    except Exception as exc:
        result_queue.put(
            {
                "status": "error",
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        model_manager.release_all()


_vlm_thread_local = threading.local()

class FrameProcessor:
    """帧处理器"""

    @staticmethod
    def load_frame(sequence_path, camera_id, frame_id):
        """加载帧数据"""
        frame_data = FrameLoader(sequence_path, camera_id=camera_id, frame_id=frame_id)
        frame_data.process_data()
        return frame_data

class VLMProcessor:
    """VLM处理器"""

    @staticmethod
    def _is_api_vlm_mode():
        model_type = config.vision_model_type.lower().replace("_", "-")
        if model_type in ("qwen-api", "qwenapi"):
            return True
        return "qwen" not in model_type

    @staticmethod
    def _get_thread_local_scale_anchor():
        scale_anchor = getattr(_vlm_thread_local, "scale_anchor", None)
        if scale_anchor is None:
            vqa_model = ModelConfig(
                model_path=config.vision_model_path,
                model_type=config.vision_model_type,
                torch_dtype=config.torch_dtype,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                num_beams=config.num_beams,
                reasoning_effort=config.reasoning_effort,
                thinking_type=config.thinking_type,
                thinking_budget_tokens=config.thinking_budget_tokens,
                max_video_frames=config.max_video_frames,
                video_fps=config.video_fps,
                model_version=config.model_version,
            )
            vision_model = get_model(vqa_model, api_key=config.api_key)
            scale_anchor = VLMSemanticScaleAnchor(vision_model)
            _vlm_thread_local.scale_anchor = scale_anchor
        return scale_anchor

    @staticmethod
    def _vlm_result_path(results_dir, frame_id):
        return os.path.join(results_dir, f"{frame_id}_camera1.json")

    @staticmethod
    def _mine_height_anchors(scale_anchor, image_path):
        source = config.height_anchor_source
        dataset_name = config.dataset
        if source == "static":
            dynamic_height_anchors = []
            static_height_anchors = scale_anchor.mine_static_height_anchors(
                image_path, dataset_name=dataset_name
            )
        elif source == "dynamic":
            dynamic_height_anchors = scale_anchor.mine_dynamic_height_anchors(
                image_path, dataset_name=dataset_name
            )
            static_height_anchors = []
        else:
            dynamic_height_anchors = scale_anchor.mine_dynamic_height_anchors(
                image_path, dataset_name=dataset_name
            )
            static_height_anchors = scale_anchor.mine_static_height_anchors(
                image_path, dataset_name=dataset_name
            )
        semantic_planes = scale_anchor.mine_scene_semantic_planes(
            image_path, dataset_name=dataset_name
        )
        return dynamic_height_anchors, static_height_anchors, semantic_planes

    @staticmethod
    def _process_single_frame(sequence_path, sequence_name, frame_id, visualization_dir, results_dir, device_label=None):
        frame_data = FrameProcessor.load_frame(sequence_path, camera_id=1, frame_id=frame_id)
        scale_anchor = VLMProcessor._get_thread_local_scale_anchor()

        dynamic_height_anchors, static_height_anchors, semantic_planes = VLMProcessor._mine_height_anchors(
            scale_anchor, frame_data.image_path
        )

        object_count = 0
        for anchor in dynamic_height_anchors:
            anchor["anchor_group"] = "dynamic"
            anchor["object_id"] = object_count
            object_count += 1
        for anchor in static_height_anchors:
            anchor["anchor_group"] = "static"
            anchor["object_id"] = object_count
            object_count += 1
        for plane in semantic_planes:
            plane["object_id"] = object_count
            object_count += 1

        result = {
            "image_path": frame_data.image_path,
            "dynamic_height_anchors": dynamic_height_anchors,
            "static_height_anchors": static_height_anchors,
            "semantic_planes": semantic_planes,
        }
        image_name = os.path.basename(result["image_path"])
        save_path = os.path.join(results_dir, image_name.replace(".jpg", ".json"))
        file_utils.write_json(result, save_path)

        if config.enable_visualization:
            scale_anchor.visualize(
                frame_data.image_path,
                visualization_dir,
                dynamic_height_anchors=dynamic_height_anchors,
                static_height_anchors=static_height_anchors,
                scene_planes=semantic_planes,
            )

        return frame_id, device_label
    
    @staticmethod
    def process_frames(sequence_path, sequence_name, frame_ids, device_label=None):
        """处理帧以提取语义锚点和平面"""
        output_dir = config.get_path('vlm_output', sequence_name)
        results_dir = config.get_path('vlm_results', sequence_name)
        visualization_dir = config.get_path('vlm_visualization', sequence_name)
        total_frames = len(frame_ids)
        log_prefix = _device_prefix(device_label)
        
        # 确保目录存在
        file_utils.ensure_directory(output_dir)
        file_utils.ensure_directory(results_dir)
        if config.enable_visualization:
            file_utils.ensure_directory(visualization_dir)

        if total_frames == 0:
            print("No frames provided for VLM processing.")
            return

        pending_frame_ids = []
        skipped_existing = 0
        for frame_id in frame_ids:
            existing = VLMProcessor._vlm_result_path(results_dir, frame_id)
            if config.resume_existing_results and os.path.isfile(existing) and os.path.getsize(existing) > 2:
                skipped_existing += 1
                continue
            pending_frame_ids.append(frame_id)
        if skipped_existing:
            print(
                f"{log_prefix} Skip {skipped_existing} existing VLM JSON file(s) "
                f"for sequence {sequence_name}."
            )
        if not pending_frame_ids:
            print(f"{log_prefix} VLM already complete for {sequence_name} ({total_frames} frames).")
            return
        frame_ids = pending_frame_ids
        total_frames = len(frame_ids)

        print(f"{log_prefix} VLM processing {total_frames} frame(s) for sequence {sequence_name}.")

        if VLMProcessor._is_api_vlm_mode() and config.parallel_vlm_api and len(frame_ids) > 1:
            max_workers = max(1, min(config.max_vlm_workers, len(frame_ids)))
            print(f"{log_prefix} Running API-based VLM with {max_workers} worker threads.")
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for frame_id in frame_ids:
                    futures.append(
                        executor.submit(
                            VLMProcessor._process_single_frame,
                            sequence_path,
                            sequence_name,
                            frame_id,
                            visualization_dir,
                            results_dir,
                            device_label,
                        )
                    )
                completed_count = 0
                for future in as_completed(futures):
                    completed_frame_id, completed_device = future.result()
                    completed_count += 1
                    print(
                        f"  {_device_prefix(completed_device)} VLM progress "
                        f"{completed_count}/{total_frames}: completed frame {completed_frame_id}"
                    )
            return

        scale_anchor = model_manager.initialize_scale_anchor()

        for frame_index, frame_id in enumerate(frame_ids, start=1):
            print(f"  {log_prefix} VLM progress {frame_index}/{total_frames}: processing frame {frame_id}")
            frame_data = FrameProcessor.load_frame(sequence_path, camera_id=1, frame_id=frame_id)

            dynamic_height_anchors, static_height_anchors, semantic_planes = VLMProcessor._mine_height_anchors(
                scale_anchor, frame_data.image_path
            )

            # 为锚点和平面添加object_id
            object_count = 0
            for anchor in dynamic_height_anchors:
                anchor["anchor_group"] = "dynamic"
                anchor["object_id"] = object_count
                object_count += 1
            for anchor in static_height_anchors:
                anchor["anchor_group"] = "static"
                anchor["object_id"] = object_count
                object_count += 1
            for plane in semantic_planes:
                plane["object_id"] = object_count
                object_count += 1

            # 保存结果
            result = {
                "image_path": frame_data.image_path,
                "dynamic_height_anchors": dynamic_height_anchors,
                "static_height_anchors": static_height_anchors,
                "semantic_planes": semantic_planes,
            }
            image_name = os.path.basename(result["image_path"])
            save_path = os.path.join(results_dir, image_name.replace(".jpg", ".json"))
            file_utils.write_json(result, save_path)

            # 可视化
            if config.enable_visualization:
                scale_anchor.visualize(
                    frame_data.image_path,
                    visualization_dir,
                    dynamic_height_anchors=dynamic_height_anchors,
                    static_height_anchors=static_height_anchors,
                    scene_planes=semantic_planes,
                )

            print(f"  {log_prefix} VLM progress {frame_index}/{total_frames}: finished frame {frame_id}")

class SAMProcessor:
    """SAM处理器"""

    @staticmethod
    def _normalize_bbox_prompt(bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if not np.isfinite([x1, y1, x2, y2]).all():
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _bbox_intersects(first, second):
        return (
            min(first[2], second[2]) > max(first[0], second[0])
            and min(first[3], second[3]) > max(first[1], second[1])
        )

    @staticmethod
    def _is_floor_plane(plane):
        return str(plane.get("plane_type", "")).strip().lower() == "floor"

    @staticmethod
    def _prepare_anchor_prompt_entries(
        dynamic_anchors,
        static_anchors,
        semantic_planes,
        require_floor_intersection,
    ):
        anchor_entries = []
        for anchor_group, anchors in (
            ("dynamic", dynamic_anchors),
            ("static", static_anchors),
        ):
            for anchor in anchors:
                bbox = SAMProcessor._normalize_bbox_prompt(anchor.get("object_bbox_2d"))
                if bbox is None:
                    continue
                anchor_entries.append((anchor_group, anchor, bbox))

        if not require_floor_intersection:
            return anchor_entries

        floor_bboxes = []
        for plane in semantic_planes:
            if not SAMProcessor._is_floor_plane(plane):
                continue
            bbox = SAMProcessor._normalize_bbox_prompt(plane.get("plane_bbox_2d"))
            if bbox is not None:
                floor_bboxes.append(bbox)

        filtered_entries = []
        for anchor_group, anchor, bbox in anchor_entries:
            intersects_floor = any(
                SAMProcessor._bbox_intersects(bbox, floor_bbox)
                for floor_bbox in floor_bboxes
            )
            if intersects_floor:
                filtered_entries.append((anchor_group, anchor, bbox))
        return filtered_entries

    @staticmethod
    def process_segmentation(vlm_results_dir, sequence_name, frame_ids=None, device=None):
        """基于VLM结果处理SAM分割"""
        output_dir = config.get_path('sam_output', sequence_name)
        visualization_dir = config.get_path('sam_visualization', sequence_name)
        
        # 确保目录存在
        file_utils.ensure_directory(output_dir)
        if config.enable_visualization:
            file_utils.ensure_directory(visualization_dir)
        
        # 加载VLM结果
        vlm_results = []
        selected_frame_ids = None if frame_ids is None else set(frame_ids)
        for json_file in os.listdir(vlm_results_dir):
            if json_file.endswith(".json"):
                frame_key = os.path.splitext(json_file)[0]
                normalized_frame_id = frame_key
                if "_camera" in frame_key:
                    normalized_frame_id = frame_key.rsplit("_camera", 1)[0]
                if selected_frame_ids is not None and normalized_frame_id not in selected_frame_ids:
                    continue
                json_path = os.path.join(vlm_results_dir, json_file)
                vlm_results.append(file_utils.read_json(json_path))
        
        tracker = model_manager.initialize_sam_tracker(device=device)
        
        # 处理每个结果
        for result in vlm_results:
            image = Image.open(result["image_path"])
            image_name = os.path.basename(result["image_path"]).split('.')[0]
            
            # 准备提示
            bbox_prompts, object_classes, object_ids, confidences, reasons, visualization_labels = [], [], [], [], [], []
            semantic_planes = result["semantic_planes"]
            anchor_entries = SAMProcessor._prepare_anchor_prompt_entries(
                result.get("dynamic_height_anchors", []),
                result.get("static_height_anchors", []),
                semantic_planes,
                config.use_sam_anchor_bbox_intersection_filter,
            )
            
            for anchor_group, anchor, bbox in anchor_entries:
                bbox_prompts.append(bbox)
                object_classes.append(anchor["object_class"])
                object_ids.append(anchor["object_id"])
                confidences.append(anchor["confidence"])
                reasons.append(anchor["reason"])
                label_prefix = "D" if anchor_group == "dynamic" else "S"
                visualization_labels.append(
                    f"[{label_prefix}] {anchor['object_class']}#{anchor['object_id']}"
                )
            
            for plane in semantic_planes:
                bbox = SAMProcessor._normalize_bbox_prompt(plane.get("plane_bbox_2d"))
                if bbox is None:
                    continue
                bbox_prompts.append(bbox)
                object_classes.append(plane["plane_type"])
                object_ids.append(plane["object_id"])
                confidences.append(plane["confidence"])
                reasons.append(plane["reason"])
                visualization_labels.append(f"[P] {plane['plane_type']}#{plane['object_id']}")

            if not bbox_prompts:
                print(f"{_device_prefix(device)} [SAM] Skip frame {image_name}: no valid bbox prompts from VLM results.")
                continue
            
            # 执行分割
            image_np = np.array(image.convert("RGB"))
            masks_batch = tracker.segment(
                image_list=[image_np],
                image_boxes_list=[np.asarray(bbox_prompts, dtype=np.float32)],
                labels_batch=[visualization_labels],
                output_dir=os.path.join(visualization_dir, image_name + ".jpg") if config.enable_visualization else None
            )
            
            # 保存分割结果
            output_subdir = os.path.join(output_dir, image_name)
            file_utils.ensure_directory(output_subdir)
            
            # SAM2 squeezes the per-box channel axis when a frame has a single
            # prompt, so normalize to [n_boxes, 1, H, W] before indexing.
            masks_batch = np.asarray(masks_batch[0])
            if masks_batch.ndim == 3:
                masks_batch = masks_batch[:, None]
            for idx in range(masks_batch.shape[0]):
                binary_mask = (masks_batch[idx, 0] * 255).astype(np.uint8)
                mask_path = os.path.join(output_subdir, f"{object_classes[idx]}_mask_{object_ids[idx]}.png")
                cv2.imwrite(mask_path, binary_mask)

class DepthProcessor:
    """深度处理器"""

    @staticmethod
    def _iter_p3_depth_maps(relative_scale_dir):
        depth_dir = os.path.join(relative_scale_dir, "depth")
        if os.path.isdir(depth_dir):
            for filename in file_utils.natural_sorted(os.listdir(depth_dir)):
                if not filename.endswith(".npy"):
                    continue
                frame_prefix = os.path.splitext(filename)[0]
                depth_map = np.load(os.path.join(depth_dir, filename))
                if isinstance(depth_map, np.ndarray) and depth_map.ndim >= 2:
                    yield frame_prefix, depth_map
            return

        pointmap_dir = os.path.join(relative_scale_dir, "pointmap")
        if not os.path.isdir(pointmap_dir):
            return

        for filename in file_utils.natural_sorted(os.listdir(pointmap_dir)):
            if not filename.endswith(".npy"):
                continue
            frame_prefix = os.path.splitext(filename)[0]
            pointmap = np.load(os.path.join(pointmap_dir, filename))
            if isinstance(pointmap, np.ndarray) and pointmap.ndim >= 3 and pointmap.shape[-1] >= 3:
                yield frame_prefix, pointmap[..., 2]

    @staticmethod
    def save_p3_depth_visualizations(relative_scale_dir, output_dir):
        """Save depth-map visualizations for Pi3/Pi3Long outputs."""
        if not config.enable_visualization:
            return 0

        file_utils.ensure_directory(output_dir)
        saved_count = 0
        for frame_prefix, depth_map in DepthProcessor._iter_p3_depth_maps(relative_scale_dir):
            output_path = os.path.join(output_dir, f"depth_visualization_{frame_prefix}.png")
            DepthVisualizer.visualize_depth_map(depth_map, output_path)
            saved_count += 1
        return saved_count

    @staticmethod
    def refresh_existing_depth_visualizations(sequence_name, model_type=None):
        """Refresh saved depth-map visualizations from already-exported depth outputs."""
        model_type = model_type or config.depth_model
        if model_type != "pi3_long":
            return 0

        output_dir = config.get_path('pi3_output', sequence_name)
        relative_scale_dir = os.path.join(output_dir, "relative_scale")
        depth_visualization_dir = os.path.join(output_dir, "depth_visualizations")
        return DepthProcessor.save_p3_depth_visualizations(relative_scale_dir, depth_visualization_dir)

    @staticmethod
    def _get_depth_output_dirs(sequence_name, model_type):
        """Resolve depth output directories without initializing the depth model."""
        if model_type == "pi3_long":
            output_dir = config.get_path('pi3_output', sequence_name)
        elif model_type == "scal3r_long":
            output_dir = config.get_path('scal3r_output', sequence_name)
        elif model_type == "lingbot_map_long":
            output_dir = config.get_path('lingbot_map_output', sequence_name)
        elif model_type == "dvlt_long":
            output_dir = config.get_path('dvlt_output', sequence_name)
        elif model_type == "4rc_long":
            output_dir = config.get_path('fourrc_output', sequence_name)
        elif model_type == "stream3r_long":
            output_dir = config.get_path('stream3r_output', sequence_name)
        elif model_type == "vggt_omega_long":
            output_dir = config.get_path('vggt_omega_output', sequence_name)
        elif model_type == "vggt_long":
            output_dir = config.get_path('vggt_output', sequence_name)
        elif model_type == "da3_long":
            output_dir = config.get_path('da3_output', sequence_name)
        else:
            raise ValueError(f"Unsupported depth model: {model_type}")

        depth_visualization_dir = os.path.join(output_dir, "depth_visualizations")
        relative_scale_dir = os.path.join(output_dir, "relative_scale")
        return output_dir, depth_visualization_dir, relative_scale_dir

    @staticmethod
    def run_depth_estimation(sequence_name, frame_ids, start_index=0, model_type=None, device=None, sequence_path=None):
        """运行深度估计"""
        model_type = model_type or config.depth_model
        input_dir = config.get_path('p3_images', sequence_name)
        output_dir, depth_visualization_dir, relative_scale_dir = DepthProcessor._get_depth_output_dirs(
            sequence_name,
            model_type,
        )

        if os.path.isdir(relative_scale_dir):
            print(
                f"{_device_prefix(device)} Existing depth outputs found; "
                f"rerunning depth estimation for {sequence_name} ({model_type}) and overwriting files."
            )

        # 确保目录存在
        file_utils.ensure_directory(output_dir)
        if config.enable_visualization:
            file_utils.ensure_directory(depth_visualization_dir)
        file_utils.ensure_directory(relative_scale_dir)
        input_pixel_limit = config.depth_input_pixel_limit
        input_max_long_edge = config.depth_input_max_long_edge
        depth_estimator = model_manager.initialize_depth_estimator(model_type, device=device)

        # 运行深度估计
        if model_type == "pi3_long":
            results = depth_estimator.process(
                input_dir,
                interval=1,
                pixel_limit=input_pixel_limit,
                max_long_edge=input_max_long_edge,
                start_frame=start_index,
                frame_count=len(frame_ids),
            )
            if results is not None:
                depth_estimator.save_results(
                    results,
                    relative_scale_dir,
                    start_frame=start_index,
                    frame_ids=frame_ids,
                    frame_count=len(frame_ids),
                    save_depth=True,
                )
                DepthProcessor.save_p3_depth_visualizations(relative_scale_dir, depth_visualization_dir)
        elif model_type in {"vggt_long", "vggt_omega_long", "dvlt_long", "4rc_long", "stream3r_long"}:
            geometry_kwargs = (
                {"calibration_dir": sequence_path, "expected_frame_ids": frame_ids}
                if model_type == "vggt_omega_long" else {}
            )
            results = depth_estimator.process(
                input_dir,
                interval=1,
                pixel_limit=input_pixel_limit,
                max_long_edge=input_max_long_edge,
                start_frame=start_index,
                frame_count=len(frame_ids),
                **geometry_kwargs,
            )
            if results is not None:
                if config.enable_visualization:
                    depth_estimator.save_visualizations(
                        results,
                        depth_visualization_dir,
                        start_frame=start_index,
                        frame_ids=frame_ids,
                    )
                depth_estimator.save_results(
                    results,
                    relative_scale_dir,
                    start_frame=start_index,
                    frame_ids=frame_ids,
                )
        elif model_type == "da3_long":
            results = depth_estimator.process(
                input_dir,
                interval=1,
                pixel_limit=input_pixel_limit,
                max_long_edge=input_max_long_edge,
                start_frame=start_index,
                frame_count=len(frame_ids),
            )
            if results is not None:
                if config.enable_visualization:
                    depth_estimator.save_visualizations(
                        results,
                        depth_visualization_dir,
                        start_frame=start_index,
                        frame_ids=frame_ids,
                    )
                depth_estimator.save_results(
                    results,
                    relative_scale_dir,
                    start_frame=start_index,
                    frame_ids=frame_ids,
                )
        elif model_type == "scal3r_long":
            results = depth_estimator.process(
                input_dir,
                interval=1,
                pixel_limit=input_pixel_limit,
                max_long_edge=input_max_long_edge,
                start_frame=start_index,
                frame_count=len(frame_ids),
            )
            if results is not None:
                if config.enable_visualization:
                    depth_estimator.save_visualizations(
                        results,
                        depth_visualization_dir,
                        start_frame=start_index,
                        frame_ids=frame_ids,
                    )
                depth_estimator.save_results(
                    results,
                    relative_scale_dir,
                    start_frame=start_index,
                    frame_ids=frame_ids,
                    save_depth=True,
                )
        elif model_type == "lingbot_map_long":
            results = depth_estimator.process(
                input_dir,
                interval=1,
                pixel_limit=input_pixel_limit,
                max_long_edge=input_max_long_edge,
                start_frame=start_index,
                frame_count=len(frame_ids),
            )
            if results is not None:
                if config.enable_visualization:
                    depth_estimator.save_visualizations(
                        results,
                        depth_visualization_dir,
                        start_frame=start_index,
                        frame_ids=frame_ids,
                    )
                depth_estimator.save_results(
                    results,
                    relative_scale_dir,
                    start_frame=start_index,
                    frame_ids=frame_ids,
                    save_depth=True,
                )
        else:
            raise ValueError(f"Unsupported depth model: {model_type}")

class SemanticPlaneScaleProcessor:
    """语义平面尺度处理器"""

    _consistent_scale_history = []

    @staticmethod
    def _get_scale_debug_dir(depth_model, sequence_name):
        base_dirs = SemanticPlaneScaleProcessor._get_scale_output_dirs(depth_model, sequence_name)
        relative_output_dir = base_dirs["relative"]
        return os.path.join(os.path.dirname(relative_output_dir), "semantic_scale_debug")

    @staticmethod
    def _save_scale_debug_json(debug_dir, filename, payload):
        if not config.save_scale_intermediate_results:
            return
        file_utils.ensure_directory(debug_dir)
        file_utils.write_json(payload, os.path.join(debug_dir, filename))

    @staticmethod
    def cleanup_scale_debug_outputs(depth_model, sequence_name):
        """删除语义尺度求解阶段的调试中间结果。"""
        import shutil

        debug_dir = SemanticPlaneScaleProcessor._get_scale_debug_dir(depth_model, sequence_name)
        if not os.path.isdir(debug_dir):
            return False

        shutil.rmtree(debug_dir)
        return True

    @staticmethod
    def _get_scale_output_dirs(depth_model, sequence_name):
        if depth_model == "pi3_long":
            base_output_dir = config.get_path('pi3_output', sequence_name)
        elif depth_model == "scal3r_long":
            base_output_dir = config.get_path('scal3r_output', sequence_name)
        elif depth_model == "lingbot_map_long":
            base_output_dir = config.get_path('lingbot_map_output', sequence_name)
        elif depth_model == "dvlt_long":
            base_output_dir = config.get_path('dvlt_output', sequence_name)
        elif depth_model == "4rc_long":
            base_output_dir = config.get_path('fourrc_output', sequence_name)
        elif depth_model == "stream3r_long":
            base_output_dir = config.get_path('stream3r_output', sequence_name)
        elif depth_model == "vggt_omega_long":
            base_output_dir = config.get_path('vggt_omega_output', sequence_name)
        elif depth_model == "vggt_long":
            base_output_dir = config.get_path('vggt_output', sequence_name)
        elif depth_model == "da3_long":
            base_output_dir = config.get_path('da3_output', sequence_name)
        else:
            raise ValueError(f"Unsupported depth model: {depth_model}")

        return {
            "relative": os.path.join(base_output_dir, "relative_scale"),
            "consistent": os.path.join(base_output_dir, "consistent_scale"),
            "optimized_legacy": os.path.join(base_output_dir, "optimized_scale_legacy"),
            "optimized_original": os.path.join(base_output_dir, "optimized_scale_original"),
            "optimized_weighted_bi": os.path.join(base_output_dir, "optimized_scale_weighted_bi"),
            "optimized_weighted_ifpa": os.path.join(base_output_dir, "optimized_scale_weighted_ifpa"),
            "optimized": os.path.join(base_output_dir, "optimized_scale"),
            "selected": os.path.join(base_output_dir, "selected_scale"),
            "absolute": os.path.join(base_output_dir, "absolute_scale"),
        }

    @staticmethod
    def _atomic_save_npy(path, array):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as file:
            np.save(file, array)
        os.replace(tmp_path, path)

    @staticmethod
    def _save_scaled_outputs(output_dir, frame_prefix, pointmap, extrinsics, camera_params):
        file_utils.ensure_directory(output_dir)
        file_utils.ensure_directory(os.path.join(output_dir, "camera_params"))
        file_utils.ensure_directory(os.path.join(output_dir, "pointmap"))

        saved_camera_params = dict(camera_params)
        saved_camera_params["extrinsics"] = extrinsics.tolist()
        file_utils.write_json(saved_camera_params, os.path.join(output_dir, "camera_params", frame_prefix + ".json"))
        SemanticPlaneScaleProcessor._atomic_save_npy(
            os.path.join(output_dir, "pointmap", frame_prefix + ".npy"),
            pointmap,
        )

    @staticmethod
    def _reset_scale_output_dirs(output_dirs):
        for output_dir in output_dirs:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            file_utils.ensure_directory(output_dir)
            file_utils.ensure_directory(os.path.join(output_dir, "camera_params"))
            file_utils.ensure_directory(os.path.join(output_dir, "pointmap"))

    @staticmethod
    def _is_valid_scale(value):
        try:
            return value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mean_valid_scales(scales):
        valid_scales = [
            float(scale)
            for scale in scales
            if SemanticPlaneScaleProcessor._is_valid_scale(scale)
        ]
        if not valid_scales:
            return None, 0
        return float(np.mean(valid_scales)), len(valid_scales)

    @staticmethod
    def _load_consistent_scale_history_from_debug(debug_dir, current_filename=None):
        if not os.path.isdir(debug_dir):
            return None, 0

        history_scales = []
        for filename in file_utils.natural_sorted(os.listdir(debug_dir)):
            if not filename.endswith(".json"):
                continue
            if current_filename is not None and filename == current_filename:
                continue

            try:
                payload = file_utils.read_json(os.path.join(debug_dir, filename))
            except (OSError, ValueError, TypeError):
                continue

            if int(payload.get("consistent_count") or 0) <= 0:
                continue
            scale = payload.get("consistent_scale")
            if SemanticPlaneScaleProcessor._is_valid_scale(scale):
                history_scales.append(float(scale))

        return SemanticPlaneScaleProcessor._mean_valid_scales(history_scales)

    @staticmethod
    def _record_consistent_scale_history(consistent_scale, consistent_count):
        if int(consistent_count or 0) <= 0:
            return
        if not SemanticPlaneScaleProcessor._is_valid_scale(consistent_scale):
            return
        SemanticPlaneScaleProcessor._consistent_scale_history.append(float(consistent_scale))

    @staticmethod
    def _resolve_consistent_scale_history_mean(debug_dir, current_filename=None):
        debug_mean, debug_count = (
            SemanticPlaneScaleProcessor._load_consistent_scale_history_from_debug(
                debug_dir, current_filename=current_filename
            )
        )
        if debug_mean is not None:
            return debug_mean, debug_count, "sequence_debug_history_mean"

        runtime_mean, runtime_count = SemanticPlaneScaleProcessor._mean_valid_scales(
            SemanticPlaneScaleProcessor._consistent_scale_history
        )
        if runtime_mean is not None:
            return runtime_mean, runtime_count, "runtime_history_mean"

        return 1.0, 0, "default_1.0_no_history"

    @staticmethod
    def _apply_consistent_scale_history_fallback(
        consistent_scale,
        optimized_scale,
        consistent_count,
        debug_dir,
        current_filename=None,
    ):
        _ = debug_dir, current_filename
        fallback_info = {
            "applied": False,
            "source": "estimated",
            "history_mean": None,
            "history_count": 0,
            "consistent_scale_before_fallback": consistent_scale,
            "optimized_scale_before_fallback": optimized_scale,
        }
        if int(consistent_count or 0) > 0:
            SemanticPlaneScaleProcessor._record_consistent_scale_history(
                consistent_scale, consistent_count
            )
            return consistent_scale, optimized_scale, fallback_info

        fallback_info.update(
            {
                "applied": False,
                "source": "no_intervals",
                "history_mean": None,
                "history_count": 0,
            }
        )
        return consistent_scale, optimized_scale, fallback_info

    @staticmethod
    def _metric_variant_to_optimized_key(metric_variant):
        return {
            "optimized_legacy": "optimized_scale_legacy",
            "optimized_original": "optimized_scale_original",
            "optimized_weighted_bi": "optimized_scale_weighted_bi",
            "optimized_weighted_ifpa": "optimized_scale_weighted_ifpa",
        }.get(metric_variant)

    @staticmethod
    def _resolve_metric_variant_scale(metric_variant, consistent_scale, default_optimized_scale, variant_scales):
        if metric_variant == "relative":
            return 1.0
        if metric_variant == "consistent":
            return consistent_scale

        optimized_key = SemanticPlaneScaleProcessor._metric_variant_to_optimized_key(metric_variant)
        if optimized_key is not None:
            return variant_scales.get(optimized_key, default_optimized_scale)

        return consistent_scale

    @staticmethod
    def _select_scale(
        consistent_scale,
        default_optimized_scale,
        default_variant_key,
        variant_scales,
        valid_interval_count,
        consistent_count,
    ):
        default_variant = config.selected_scale_default
        fallback_scale = SemanticPlaneScaleProcessor._resolve_metric_variant_scale(
            default_variant,
            consistent_scale,
            default_optimized_scale,
            variant_scales,
        )
        if not SemanticPlaneScaleProcessor._is_valid_scale(fallback_scale):
            fallback_scale = (
                consistent_scale
                if SemanticPlaneScaleProcessor._is_valid_scale(consistent_scale)
                else 1.0
            )
            default_variant = (
                "consistent"
                if SemanticPlaneScaleProcessor._is_valid_scale(consistent_scale)
                else "relative"
            )

        shift_ratio = None
        optimized_to_consistent_ratio = None
        optimized_available = (
            SemanticPlaneScaleProcessor._is_valid_scale(consistent_scale)
            and SemanticPlaneScaleProcessor._is_valid_scale(default_optimized_scale)
            and abs(float(consistent_scale)) > 1e-12
            and int(consistent_count or 0) > 0
        )
        if optimized_available:
            optimized_to_consistent_ratio = float(default_optimized_scale) / float(consistent_scale)
            shift_ratio = abs(optimized_to_consistent_ratio - 1.0)

        optimized_shift_gate = optimized_available and shift_ratio <= config.max_optimized_shift_ratio
        minimum_ratio = config.min_optimized_to_consistent_ratio
        optimized_ratio_gate = optimized_available and (
            minimum_ratio is None or optimized_to_consistent_ratio >= minimum_ratio
        )

        if optimized_shift_gate and optimized_ratio_gate:
            selected_info = {
                "scale": float(default_optimized_scale),
                "variant": default_variant_key or "optimized",
                "reason": "optimized_shift_within_threshold",
                "optimized_shift_ratio": float(shift_ratio),
                "gate_passed": True,
            }
        else:
            if int(consistent_count or 0) <= 0:
                reason = "consistent_count_zero"
            elif not optimized_available:
                reason = "optimized_or_consistent_scale_unavailable"
            elif not optimized_ratio_gate:
                reason = "optimized_below_consistent_ratio_threshold"
            else:
                reason = "optimized_shift_exceeds_threshold"

            selected_info = {
                "scale": float(fallback_scale),
                "variant": default_variant,
                "reason": reason,
                "optimized_shift_ratio": None if shift_ratio is None else float(shift_ratio),
                "gate_passed": False,
            }

        selected_info.update(
            {
                "valid_interval_count": int(valid_interval_count or 0),
                "consistent_count": int(consistent_count or 0),
                "min_valid_intervals_for_optimized": None,
                "min_consistent_count_for_optimized": None,
                "valid_interval_gate_passed": True,
                "consistent_count_gate_passed": True,
                "optimized_shift_gate_passed": optimized_shift_gate,
                "optimized_ratio_gate_passed": optimized_ratio_gate,
                "optimized_to_consistent_ratio": optimized_to_consistent_ratio,
            }
        )
        return selected_info

    @staticmethod
    def _count_scale_interval_candidates(scale_estimator, vlm_results_list):
        """Count model-independent semantic anchors before depth geometry filtering."""
        raw_anchor_count = 0
        selected_anchor_count = 0
        candidate_anchor_count = 0

        for vlm_result in vlm_results_list:
            dynamic_height_anchors = [
                {**anchor, "anchor_group": anchor.get("anchor_group", "dynamic")}
                for anchor in vlm_result.get("dynamic_height_anchors", [])
            ]
            static_height_anchors = [
                {**anchor, "anchor_group": anchor.get("anchor_group", "static")}
                for anchor in vlm_result.get("static_height_anchors", [])
            ]
            raw_anchor_count += len(dynamic_height_anchors) + len(static_height_anchors)

            if scale_estimator.height_anchor_source == "dynamic":
                height_anchors = dynamic_height_anchors
            elif scale_estimator.height_anchor_source == "static":
                height_anchors = static_height_anchors
            else:
                height_anchors = dynamic_height_anchors + static_height_anchors

            height_anchors = [
                anchor
                for anchor in height_anchors
                if scale_estimator._is_valid_height_interval(anchor)
            ]
            selected_anchor_count += len(height_anchors)
            for anchor in height_anchors:
                if scale_estimator.use_vlm_anchor_confidence_filter:
                    confidence_eval = scale_estimator._evaluate_vlm_anchor_confidence(anchor)
                    if not confidence_eval.get("passed_vlm_confidence_filter", True):
                        continue
                candidate_anchor_count += 1

        return {
            "raw_anchor_count": raw_anchor_count,
            "selected_anchor_count": selected_anchor_count,
            "candidate_anchor_count": candidate_anchor_count,
        }

    @staticmethod
    def _load_vlm_results_for_frame_ids(vlm_results_dir, frame_ids):
        """Load VLM results by requested frame IDs, independent of depth outputs."""
        requested_frame_ids = set(frame_ids)
        vlm_results_list = []

        for json_file in file_utils.natural_sorted(os.listdir(vlm_results_dir)):
            if not json_file.endswith(".json"):
                continue
            frame_prefix = os.path.splitext(json_file)[0]
            frame_id = frame_prefix.split("_")[0]
            if frame_id not in requested_frame_ids:
                continue
            vlm_results_list.append(
                file_utils.read_json(os.path.join(vlm_results_dir, json_file))
            )

        return vlm_results_list

    @staticmethod
    def _summarize_scale_interval_rejections(
        candidate_count,
        anchor_measurements,
        depth_geometry_filter_enabled=True,
    ):
        """Summarize why model-independent candidates did not become scale intervals."""
        valid_count = sum(
            1 for item in anchor_measurements if item.get("scale_interval") is not None
        )
        if not depth_geometry_filter_enabled:
            return {
                "depth_geometry_filter_enabled": False,
                "candidate_count": int(candidate_count),
                "valid_count": int(candidate_count),
                "solver_interval_count": valid_count,
                "invalid_count": 0,
                "recorded_invalid_count": 0,
                "unrecorded_invalid_count": 0,
                "reason_counts": {},
            }

        recorded_invalid_count = 0
        reason_counts = {}

        for item in anchor_measurements:
            if item.get("scale_interval") is not None:
                continue

            recorded_invalid_count += 1
            if item.get("depth_geometry_filter_stage"):
                reason = item["depth_geometry_filter_stage"]
            elif not item.get("passed_vlm_confidence_filter", True):
                reason = "vlm_confidence_filter"
            elif not item.get("passed_quality_filter", True):
                reason = "anchor_quality_filter"
            elif item.get("relative_height") is not None:
                reason = "invalid_relative_height"
            else:
                reason = "missing_depth_geometry"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        unrecorded_invalid_count = max(
            int(candidate_count) - valid_count - recorded_invalid_count, 0
        )
        if unrecorded_invalid_count > 0:
            reason_counts["missing_valid_anchor_points_or_measurement"] = (
                reason_counts.get("missing_valid_anchor_points_or_measurement", 0)
                + unrecorded_invalid_count
            )

        return {
            "depth_geometry_filter_enabled": True,
            "candidate_count": int(candidate_count),
            "valid_count": valid_count,
            "solver_interval_count": valid_count,
            "invalid_count": max(int(candidate_count) - valid_count, 0),
            "recorded_invalid_count": recorded_invalid_count,
            "unrecorded_invalid_count": unrecorded_invalid_count,
            "reason_counts": reason_counts,
        }

    @staticmethod
    def _collect_scale_interval_rejection_details(anchor_measurements):
        details = []
        for item in anchor_measurements:
            if item.get("scale_interval") is not None:
                continue
            stage = item.get("depth_geometry_filter_stage")
            if not stage and not item.get("passed_vlm_confidence_filter", True):
                stage = "vlm_confidence_filter"
            if not stage and not item.get("passed_quality_filter", True):
                stage = "anchor_quality_filter"
            if not stage and item.get("relative_height") is not None:
                stage = "invalid_relative_height"
            if not stage:
                stage = "missing_depth_geometry"

            details.append(
                {
                    "stage": stage,
                    "frame_name": item.get("frame_name"),
                    "object_class": item.get("object_class"),
                    "object_id": item.get("object_id"),
                    "anchor_group": item.get("anchor_group"),
                    "height_interval": item.get("height_interval"),
                    "relative_height": item.get("relative_height"),
                    "anchor_point_count": item.get("anchor_point_count"),
                    "plane_point_count": item.get("plane_point_count"),
                    "anchor_mask_pixel_count": item.get("anchor_mask_pixel_count"),
                    "filtered_anchor_mask_pixel_count": item.get("filtered_anchor_mask_pixel_count"),
                    "valid_anchor_point_count": item.get("valid_anchor_point_count"),
                    "conf_value_count": item.get("conf_value_count"),
                    "anchor_conf_min": item.get("anchor_conf_min"),
                    "anchor_conf_max": item.get("anchor_conf_max"),
                    "frame_conf_min_threshold": item.get("frame_conf_min_threshold"),
                    "frame_conf_max_threshold": item.get("frame_conf_max_threshold"),
                    "vlm_anchor_confidence": item.get("vlm_anchor_confidence"),
                    "vlm_anchor_confidence_threshold": item.get("vlm_anchor_confidence_threshold"),
                }
            )
        return details

    @staticmethod
    def _print_scale_interval_rejection_details(details, limit=50):
        if not details:
            return
        print("Depth geometry rejection details:")
        for idx, item in enumerate(details[:limit], start=1):
            parts = [
                f"#{idx}",
                f"stage={item.get('stage')}",
                f"frame={item.get('frame_name')}",
                f"class={item.get('object_class')}",
                f"id={item.get('object_id')}",
                f"group={item.get('anchor_group')}",
            ]
            optional_keys = (
                "anchor_mask_pixel_count",
                "filtered_anchor_mask_pixel_count",
                "valid_anchor_point_count",
                "anchor_point_count",
                "plane_point_count",
                "relative_height",
                "conf_value_count",
                "anchor_conf_min",
                "anchor_conf_max",
                "frame_conf_min_threshold",
                "frame_conf_max_threshold",
                "vlm_anchor_confidence",
                "vlm_anchor_confidence_threshold",
            )
            for key in optional_keys:
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}={value}")
            print("  " + ", ".join(parts))
        if len(details) > limit:
            print(f"  ... {len(details) - limit} more rejection detail(s) omitted")

    @staticmethod
    def _print_shared_scale_runtime_summary(
        frame_ids,
        candidate_stats,
        rejection_summary,
        consistent_scale,
        consistent_count,
        optimized_scale,
        selected_scale_info=None,
        consistent_fallback_info=None,
    ):
        subseq_name = f"{frame_ids[0]} -> {frame_ids[-1]}" if frame_ids else "empty"
        print("=== Shared Scale Estimation Results ===")
        print(f"Scale subsequence: {subseq_name}")
        print(f"Total scale intervals collected: {candidate_stats.get('candidate_anchor_count', 0)}")
        if config.use_depth_geometry_filter:
            print(
                "Valid scale intervals after depth geometry: "
                f"{rejection_summary.get('valid_count', 0)}"
            )
            invalid_count = rejection_summary.get("invalid_count", 0)
            if invalid_count:
                print(
                    "Rejected scale interval candidates: "
                    f"{invalid_count} ({rejection_summary.get('reason_counts', {})})"
                )
        else:
            print("Depth geometry filtering: disabled")
            print(f"Scale intervals used by solver: {rejection_summary.get('valid_count', 0)}")

        consistent_text = f"{float(consistent_scale):.4f}" if consistent_scale is not None else "-"
        optimized_text = f"{float(optimized_scale):.4f}" if optimized_scale is not None else "-"
        print(f"Consistent scale: {consistent_text}")
        print(f"Consistent count: {consistent_count}")
        if consistent_fallback_info and consistent_fallback_info.get("applied"):
            print(
                "Consistent fallback: "
                f"{consistent_fallback_info.get('source')}, "
                f"history_mean={consistent_fallback_info.get('history_mean')}, "
                f"history_count={consistent_fallback_info.get('history_count')}"
            )
        print(f"Optimized scale: {optimized_text}")
        if selected_scale_info and config.enable_selected_scale:
            selected_scale = selected_scale_info.get("scale")
            selected_text = f"{float(selected_scale):.4f}" if selected_scale is not None else "-"
            print(
                "Selected scale: "
                f"{selected_text} "
                f"({selected_scale_info.get('variant')}, {selected_scale_info.get('reason')})"
            )
    
    @staticmethod
    def _should_fallback_to_all_dynamic_anchors(scale_estimator, vlm_results_list):
        if (
            not config.fallback_to_all_dynamic_anchors_if_subsequence_empty
            or config.height_anchor_source != "dynamic"
            or not scale_estimator.optimizer_dynamic_classes
        ):
            return False

        allowed_classes = scale_estimator.optimizer_dynamic_classes
        for vlm_result in vlm_results_list:
            for anchor in vlm_result.get("dynamic_height_anchors", []):
                object_class = str(anchor.get("object_class", "") or "").strip().lower()
                if object_class not in allowed_classes:
                    continue
                dynamic_anchor = {**anchor, "anchor_group": anchor.get("anchor_group", "dynamic")}
                if scale_estimator._evaluate_vlm_anchor_confidence(dynamic_anchor)[
                    "passed_vlm_confidence_filter"
                ]:
                    return False
        return True

    @staticmethod
    def _scale_camera_extrinsics(extrinsics, scale):
        scaled = extrinsics.copy()
        if config.scale_camera_translation:
            scaled[:, 3] = scaled[:, 3] * scale
        return scaled

    @staticmethod
    def estimate_scale(vlm_results_dir, depth_model, use_shared_scale, sequence_name, frame_ids):
        """估计语义平面尺度"""
        scale_estimator = model_manager.initialize_semantic_plane_scale_estimator()
        scale_estimator.dynamic_anchor_height_scale = config.dynamic_anchor_height_scale
        
        # 确定路径
        scale_output_dirs = SemanticPlaneScaleProcessor._get_scale_output_dirs(depth_model, sequence_name)
        relative_output_dir = scale_output_dirs["relative"]
        consistent_output_dir = scale_output_dirs["consistent"]
        optimized_variant_dirs = {
            "optimized_scale_legacy": scale_output_dirs["optimized_legacy"],
            "optimized_scale_original": scale_output_dirs["optimized_original"],
            "optimized_scale_weighted_bi": scale_output_dirs["optimized_weighted_bi"],
            "optimized_scale_weighted_ifpa": scale_output_dirs["optimized_weighted_ifpa"],
        }
        enabled_optimized_variants = set(config.enabled_optimized_variants)
        optimized_variant_dirs = {
            key: value
            for key, value in optimized_variant_dirs.items()
            if key in enabled_optimized_variants
        }
        optimized_output_dir = scale_output_dirs["optimized"]
        selected_output_dir = scale_output_dirs["selected"] if config.enable_selected_scale else None
        absolute_output_dir = scale_output_dirs["absolute"]
        scale_debug_dir = SemanticPlaneScaleProcessor._get_scale_debug_dir(depth_model, sequence_name)
        if not config.save_scale_intermediate_results:
            SemanticPlaneScaleProcessor.cleanup_scale_debug_outputs(depth_model, sequence_name)
        
        pointmap_dir = os.path.join(relative_output_dir, "pointmap")
        conf_dir = os.path.join(relative_output_dir, "conf")
        camera_params_dir = os.path.join(relative_output_dir, "camera_params")
        
        # 确保目录存在
        all_output_dirs = [consistent_output_dir, optimized_output_dir, absolute_output_dir] + list(optimized_variant_dirs.values())
        if selected_output_dir is not None:
            all_output_dirs.append(selected_output_dir)
        SemanticPlaneScaleProcessor._reset_scale_output_dirs(all_output_dirs)
        
        subsequence_frames = set(frame_ids)
        candidate_vlm_results_list = SemanticPlaneScaleProcessor._load_vlm_results_for_frame_ids(
            vlm_results_dir, subsequence_frames
        )
        
        if use_shared_scale:
            print("\n=== Using shared scale mode ===")
            # 收集数据
            pointmap_list = []
            conf_list = []
            extrinsics_list = []
            vlm_results_list = []
            sequence_name_list = []
            frame_data_list = []
            
            for framename in file_utils.natural_sorted(os.listdir(pointmap_dir)):
                if framename.endswith(".npy"):
                    frame_prefix = framename.split(".")[0]
                    frame_id = frame_prefix.split("_")[0]
                    
                    if frame_id in subsequence_frames:
                        # 加载数据
                        pointmap = np.load(os.path.join(pointmap_dir, framename))
                        conf = np.load(os.path.join(conf_dir, frame_prefix + ".npy"))
                        camera_params = file_utils.read_json(os.path.join(camera_params_dir, frame_prefix + ".json"))
                        extrinsics = np.array(camera_params["extrinsics"])
                        vlm_result = file_utils.read_json(os.path.join(vlm_results_dir, frame_prefix + ".json"))
                        
                        # 收集数据
                        pointmap_list.append(pointmap)
                        conf_list.append(conf)
                        extrinsics_list.append(extrinsics)
                        vlm_results_list.append(vlm_result)
                        sequence_name_list.append(sequence_name)
                        
                        frame_data_list.append({
                            "frame_prefix": frame_prefix,
                            "pointmap": pointmap,
                            "extrinsics": extrinsics,
                            "camera_params": camera_params
                        })
            
            # 计算共享尺度
            if pointmap_list:
                original_dynamic_classes = scale_estimator.optimizer_dynamic_classes.copy()
                use_all_dynamic_anchors = (
                    SemanticPlaneScaleProcessor._should_fallback_to_all_dynamic_anchors(
                        scale_estimator, vlm_results_list
                    )
                )
                if use_all_dynamic_anchors:
                    scale_estimator.optimizer_dynamic_classes = set()
                    print(
                        "  No eligible selected dynamic anchors in this subsequence; "
                        "falling back to all dynamic anchor classes."
                    )
                try:
                    candidate_stats = SemanticPlaneScaleProcessor._count_scale_interval_candidates(
                        scale_estimator, candidate_vlm_results_list
                    )
                    (
                        consistent_scale,
                        consistent_count,
                        optimized_scale,
                        all_scale_intervals,
                        anchor_measurements,
                    ) = scale_estimator.estimate_metric_scale_batch(
                        pointmap_list, conf_list, extrinsics_list, vlm_results_list, sequence_name_list
                    )
                finally:
                    if use_all_dynamic_anchors:
                        scale_estimator.optimizer_dynamic_classes = original_dynamic_classes
                consensus_result = scale_estimator.last_consensus_result or {}
                bi_result = scale_estimator.last_bi_result or {}
                ifpa_result = scale_estimator.last_ifpa_result or {}
                optimization_result = scale_estimator.last_optimization_result or {}
                variant_results = scale_estimator.last_variant_results or {}
                rejection_summary = SemanticPlaneScaleProcessor._summarize_scale_interval_rejections(
                    candidate_stats["candidate_anchor_count"],
                    anchor_measurements,
                    depth_geometry_filter_enabled=config.use_depth_geometry_filter,
                )
                rejection_details = SemanticPlaneScaleProcessor._collect_scale_interval_rejection_details(
                    anchor_measurements
                )
                subseq_name = f"{frame_ids[0]}_{frame_ids[-1]}" if frame_ids else "empty"
                debug_filename = f"shared_scale_{subseq_name}.json"
                if not all_scale_intervals:
                    consistent_scale = 1.0
                    optimized_scale = 1.0
                    variant_results = {}
                consistent_scale, optimized_scale, consistent_fallback_info = (
                    SemanticPlaneScaleProcessor._apply_consistent_scale_history_fallback(
                        consistent_scale,
                        optimized_scale,
                        consistent_count,
                        scale_debug_dir,
                        current_filename=debug_filename,
                    )
                )
                if consistent_fallback_info["applied"]:
                    variant_results = {}

                variant_scales_for_selection = {
                    key: value["scale"]
                    for key, value in variant_results.get("optimized_variants", {}).items()
                    if value.get("scale") is not None
                }
                default_variant_key = variant_results.get("default_optimized_variant_key")
                default_optimized_scale = variant_scales_for_selection.get(
                    default_variant_key,
                    optimized_scale,
                )
                selected_scale_info = SemanticPlaneScaleProcessor._select_scale(
                    consistent_scale,
                    default_optimized_scale,
                    default_variant_key,
                    variant_scales_for_selection,
                    rejection_summary["valid_count"],
                    consistent_count,
                )
                SemanticPlaneScaleProcessor._print_shared_scale_runtime_summary(
                    frame_ids,
                    candidate_stats,
                    rejection_summary,
                    consistent_scale,
                    consistent_count,
                    optimized_scale,
                    selected_scale_info=selected_scale_info,
                    consistent_fallback_info=consistent_fallback_info,
                )
                SemanticPlaneScaleProcessor._save_scale_debug_json(
                    scale_debug_dir,
                    debug_filename,
                    {
                        "sequence_name": sequence_name,
                        "sequence_difficulty_profile": config.active_semantic_scale_profile,
                        "frame_ids": list(frame_ids),
                        "relative_height_mode": config.relative_height_mode,
                        "use_depth_geometry_filter": config.use_depth_geometry_filter,
                        "depth_geometry_min_anchor_points": config.depth_geometry_min_anchor_points,
                        "depth_geometry_min_relative_height": config.depth_geometry_min_relative_height,
                        "consistent_scale": consistent_scale,
                        "consistent_count": consistent_count,
                        "optimized_scale": optimized_scale,
                        "consistent_scale_fallback_applied": consistent_fallback_info["applied"],
                        "consistent_scale_fallback_source": consistent_fallback_info["source"],
                        "consistent_scale_history_mean": consistent_fallback_info["history_mean"],
                        "consistent_scale_history_count": consistent_fallback_info["history_count"],
                        "consistent_scale_before_fallback": consistent_fallback_info["consistent_scale_before_fallback"],
                        "optimized_scale_before_fallback": consistent_fallback_info["optimized_scale_before_fallback"],
                        "enable_selected_scale": config.enable_selected_scale,
                        "selected_scale": selected_scale_info["scale"],
                        "selected_scale_variant": selected_scale_info["variant"],
                        "selected_scale_reason": selected_scale_info["reason"],
                        "selected_scale_default": config.selected_scale_default,
                        "selected_scale_gate_passed": selected_scale_info["gate_passed"],
                        "selected_scale_valid_intervals": selected_scale_info["valid_interval_count"],
                        "min_valid_intervals_for_optimized": selected_scale_info["min_valid_intervals_for_optimized"],
                        "selected_scale_consistent_count": selected_scale_info["consistent_count"],
                        "min_consistent_count_for_optimized": selected_scale_info["min_consistent_count_for_optimized"],
                        "selected_scale_valid_interval_gate_passed": selected_scale_info["valid_interval_gate_passed"],
                        "selected_scale_consistent_count_gate_passed": selected_scale_info["consistent_count_gate_passed"],
                        "selected_scale_optimized_shift_gate_passed": selected_scale_info["optimized_shift_gate_passed"],
                        "selected_scale_optimized_shift_ratio": selected_scale_info["optimized_shift_ratio"],
                        "max_optimized_shift_ratio": config.max_optimized_shift_ratio,
                        "optimized_scale_solver": optimization_result.get("optimized_scale_solver"),
                        "weighted_search_mode": optimization_result.get("weighted_search_mode"),
                        "consensus_prior_mode": optimization_result.get("consensus_prior_mode"),
                        "consensus_prior_strength": optimization_result.get("consensus_prior_strength"),
                        "consensus_prior_interval_inside_ratio": optimization_result.get("consensus_prior_interval_inside_ratio"),
                        "used_consensus_prior": optimization_result.get("used_consensus_prior"),
                        "consensus_prior_scale": optimization_result.get("consensus_prior_scale"),
                        "consensus_prior_interval": optimization_result.get("consensus_prior_interval"),
                        "optimized_scale_search_bounds": optimization_result.get("search_bounds"),
                        "optimized_scale_used_search_bounds": optimization_result.get("used_search_bounds"),
                        "optimized_scale_constraint_count": optimization_result.get("constraint_count"),
                        "consensus_solver": consensus_result.get("solver"),
                        "consensus_scale": consensus_result.get("consensus_scale"),
                        "consensus_interval": consensus_result.get("consensus_interval"),
                        "bi_scale": bi_result.get("bi_scale"),
                        "bi_interval": bi_result.get("bi_interval"),
                        "bi_keep_threshold": bi_result.get("keep_threshold"),
                        "bi_total_weight": bi_result.get("total_weight"),
                        "bi_kept_segment_count": bi_result.get("kept_segment_count"),
                        "bi_all_segment_count": bi_result.get("all_segment_count"),
                        "bi_segments": bi_result.get("bi_segments", []),
                        "ifpa_scale": ifpa_result.get("ifpa_scale"),
                        "ifpa_interval": ifpa_result.get("ifpa_interval"),
                        "ifpa_intervals": ifpa_result.get("ifpa_intervals", []),
                        "ifpa_best_score": ifpa_result.get("ifpa_best_score"),
                        "ifpa_grid_size": ifpa_result.get("ifpa_grid_size"),
                        "ifpa_use_weight": ifpa_result.get("ifpa_use_weight"),
                        "ifpa_top_point_count": ifpa_result.get("ifpa_top_point_count"),
                        "ifpa_top_points": ifpa_result.get("ifpa_top_points", []),
                        "ifpa_score_profile": ifpa_result.get("ifpa_score_profile", []),
                        "default_optimized_variant_key": variant_results.get("default_optimized_variant_key"),
                        "optimized_variants": variant_results.get("optimized_variants", {}),
                        "raw_anchor_count": candidate_stats["raw_anchor_count"],
                        "selected_anchor_count": candidate_stats["selected_anchor_count"],
                        "candidate_anchor_count": candidate_stats["candidate_anchor_count"],
                        "valid_scale_interval_count": rejection_summary["valid_count"],
                        "scale_interval_solver_count": len(all_scale_intervals),
                        "scale_interval_rejection_summary": rejection_summary,
                        "scale_interval_rejection_details": rejection_details,
                        "scale_interval_count": len(all_scale_intervals),
                        "anchor_measurements": anchor_measurements,
                    },
                )
            else:
                print("\n=== No frames processed, using default scale ===")
                consistent_scale = 1.0
                optimized_scale = 1.0
                variant_scales_for_selection = {}
                default_variant_key = None
                default_optimized_scale = optimized_scale
                selected_scale_info = SemanticPlaneScaleProcessor._select_scale(
                    consistent_scale,
                    default_optimized_scale,
                    default_variant_key,
                    variant_scales_for_selection,
                    0,
                    0,
                )
            
            # 应用尺度
            variant_scales = {
                variant_key: variant_scales_for_selection.get(variant_key, default_optimized_scale)
                for variant_key in optimized_variant_dirs
            }
            for frame_data in frame_data_list:
                frame_prefix = frame_data["frame_prefix"]
                pointmap = frame_data["pointmap"]
                extrinsics = frame_data["extrinsics"]
                camera_params = frame_data["camera_params"]
                
                consistent_pointmap = pointmap * consistent_scale
                consistent_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                    extrinsics, consistent_scale
                )

                SemanticPlaneScaleProcessor._save_scaled_outputs(
                    consistent_output_dir, frame_prefix, consistent_pointmap, consistent_extrinsics, camera_params
                )
                for variant_key, variant_scale in variant_scales.items():
                    variant_pointmap = pointmap * variant_scale
                    variant_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                        extrinsics, variant_scale
                    )
                    SemanticPlaneScaleProcessor._save_scaled_outputs(
                        optimized_variant_dirs[variant_key],
                        frame_prefix,
                        variant_pointmap,
                        variant_extrinsics,
                        camera_params,
                    )

                optimized_pointmap = pointmap * default_optimized_scale
                optimized_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                    extrinsics, default_optimized_scale
                )
                SemanticPlaneScaleProcessor._save_scaled_outputs(
                    optimized_output_dir, frame_prefix, optimized_pointmap, optimized_extrinsics, camera_params
                )
                SemanticPlaneScaleProcessor._save_scaled_outputs(
                    absolute_output_dir, frame_prefix, optimized_pointmap, optimized_extrinsics, camera_params
                )
                if selected_output_dir is not None:
                    selected_scale = selected_scale_info["scale"]
                    selected_pointmap = pointmap * selected_scale
                    selected_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                        extrinsics, selected_scale
                    )
                    SemanticPlaneScaleProcessor._save_scaled_outputs(
                        selected_output_dir,
                        frame_prefix,
                        selected_pointmap,
                        selected_extrinsics,
                        camera_params,
                    )
        else:
            print("\n=== Using per-frame scale mode ===")
            # 逐帧处理
            for framename in file_utils.natural_sorted(os.listdir(pointmap_dir)):
                if framename.endswith(".npy"):
                    frame_prefix = framename.split(".")[0]
                    frame_id = frame_prefix.split("_")[0]
                    
                    if frame_id in subsequence_frames:
                        # 加载数据
                        pointmap = np.load(os.path.join(pointmap_dir, framename))
                        conf = np.load(os.path.join(conf_dir, frame_prefix + ".npy"))
                        camera_params = file_utils.read_json(os.path.join(camera_params_dir, frame_prefix + ".json"))
                        extrinsics = np.array(camera_params["extrinsics"])
                        vlm_result = file_utils.read_json(os.path.join(vlm_results_dir, frame_prefix + ".json"))
                        
                        # 估计尺度
                        candidate_stats = SemanticPlaneScaleProcessor._count_scale_interval_candidates(
                            scale_estimator, [vlm_result]
                        )
                        (
                            consistent_scale,
                            consistent_count,
                            optimized_scale,
                            scale_intervals,
                            anchor_measurements,
                        ) = scale_estimator.estimate_metric_scale(
                            pointmap, conf, extrinsics, vlm_result, sequence_name
                        )
                        consensus_result = scale_estimator.last_consensus_result or {}
                        bi_result = scale_estimator.last_bi_result or {}
                        ifpa_result = scale_estimator.last_ifpa_result or {}
                        optimization_result = scale_estimator.last_optimization_result or {}
                        variant_results = scale_estimator.last_variant_results or {}
                        rejection_summary = SemanticPlaneScaleProcessor._summarize_scale_interval_rejections(
                            candidate_stats["candidate_anchor_count"],
                            anchor_measurements,
                            depth_geometry_filter_enabled=config.use_depth_geometry_filter,
                        )
                        rejection_details = SemanticPlaneScaleProcessor._collect_scale_interval_rejection_details(
                            anchor_measurements
                        )
                        if not scale_intervals:
                            consistent_scale = 1.0
                            optimized_scale = 1.0
                            variant_results = {}
                        debug_filename = f"{frame_prefix}.json"
                        consistent_scale, optimized_scale, consistent_fallback_info = (
                            SemanticPlaneScaleProcessor._apply_consistent_scale_history_fallback(
                                consistent_scale,
                                optimized_scale,
                                consistent_count,
                                scale_debug_dir,
                                current_filename=debug_filename,
                            )
                        )
                        if consistent_fallback_info["applied"]:
                            variant_results = {}

                        variant_scales_for_selection = {
                            key: value["scale"]
                            for key, value in variant_results.get("optimized_variants", {}).items()
                            if value.get("scale") is not None
                        }
                        default_variant_key = variant_results.get("default_optimized_variant_key")
                        default_optimized_scale = variant_scales_for_selection.get(
                            default_variant_key,
                            optimized_scale,
                        )
                        selected_scale_info = SemanticPlaneScaleProcessor._select_scale(
                            consistent_scale,
                            default_optimized_scale,
                            default_variant_key,
                            variant_scales_for_selection,
                            rejection_summary["valid_count"],
                            consistent_count,
                        )
                        SemanticPlaneScaleProcessor._save_scale_debug_json(
                            scale_debug_dir,
                            debug_filename,
                            {
                                "sequence_name": sequence_name,
                                "sequence_difficulty_profile": config.active_semantic_scale_profile,
                                "frame_prefix": frame_prefix,
                                "relative_height_mode": config.relative_height_mode,
                                "use_depth_geometry_filter": config.use_depth_geometry_filter,
                                "depth_geometry_min_anchor_points": config.depth_geometry_min_anchor_points,
                                "depth_geometry_min_relative_height": config.depth_geometry_min_relative_height,
                                "consistent_scale": consistent_scale,
                                "consistent_count": consistent_count,
                                "optimized_scale": optimized_scale,
                                "consistent_scale_fallback_applied": consistent_fallback_info["applied"],
                                "consistent_scale_fallback_source": consistent_fallback_info["source"],
                                "consistent_scale_history_mean": consistent_fallback_info["history_mean"],
                                "consistent_scale_history_count": consistent_fallback_info["history_count"],
                                "consistent_scale_before_fallback": consistent_fallback_info["consistent_scale_before_fallback"],
                                "optimized_scale_before_fallback": consistent_fallback_info["optimized_scale_before_fallback"],
                                "enable_selected_scale": config.enable_selected_scale,
                                "selected_scale": selected_scale_info["scale"],
                                "selected_scale_variant": selected_scale_info["variant"],
                                "selected_scale_reason": selected_scale_info["reason"],
                                "selected_scale_default": config.selected_scale_default,
                                "selected_scale_gate_passed": selected_scale_info["gate_passed"],
                                "selected_scale_valid_intervals": selected_scale_info["valid_interval_count"],
                                "min_valid_intervals_for_optimized": selected_scale_info["min_valid_intervals_for_optimized"],
                                "selected_scale_consistent_count": selected_scale_info["consistent_count"],
                                "min_consistent_count_for_optimized": selected_scale_info["min_consistent_count_for_optimized"],
                                "selected_scale_valid_interval_gate_passed": selected_scale_info["valid_interval_gate_passed"],
                                "selected_scale_consistent_count_gate_passed": selected_scale_info["consistent_count_gate_passed"],
                                "selected_scale_optimized_shift_gate_passed": selected_scale_info["optimized_shift_gate_passed"],
                                "selected_scale_optimized_shift_ratio": selected_scale_info["optimized_shift_ratio"],
                                "max_optimized_shift_ratio": config.max_optimized_shift_ratio,
                                "optimized_scale_solver": optimization_result.get("optimized_scale_solver"),
                                "weighted_search_mode": optimization_result.get("weighted_search_mode"),
                                "consensus_prior_mode": optimization_result.get("consensus_prior_mode"),
                                "consensus_prior_strength": optimization_result.get("consensus_prior_strength"),
                                "consensus_prior_interval_inside_ratio": optimization_result.get("consensus_prior_interval_inside_ratio"),
                                "used_consensus_prior": optimization_result.get("used_consensus_prior"),
                                "consensus_prior_scale": optimization_result.get("consensus_prior_scale"),
                                "consensus_prior_interval": optimization_result.get("consensus_prior_interval"),
                                "optimized_scale_search_bounds": optimization_result.get("search_bounds"),
                                "optimized_scale_used_search_bounds": optimization_result.get("used_search_bounds"),
                                "optimized_scale_constraint_count": optimization_result.get("constraint_count"),
                                "consensus_solver": consensus_result.get("solver"),
                                "consensus_scale": consensus_result.get("consensus_scale"),
                                "consensus_interval": consensus_result.get("consensus_interval"),
                                "bi_scale": bi_result.get("bi_scale"),
                                "bi_interval": bi_result.get("bi_interval"),
                                "bi_keep_threshold": bi_result.get("keep_threshold"),
                                "bi_total_weight": bi_result.get("total_weight"),
                                "bi_kept_segment_count": bi_result.get("kept_segment_count"),
                                "bi_all_segment_count": bi_result.get("all_segment_count"),
                                "bi_segments": bi_result.get("bi_segments", []),
                                "ifpa_scale": ifpa_result.get("ifpa_scale"),
                                "ifpa_interval": ifpa_result.get("ifpa_interval"),
                                "ifpa_intervals": ifpa_result.get("ifpa_intervals", []),
                                "ifpa_best_score": ifpa_result.get("ifpa_best_score"),
                                "ifpa_grid_size": ifpa_result.get("ifpa_grid_size"),
                                "ifpa_use_weight": ifpa_result.get("ifpa_use_weight"),
                                "ifpa_top_point_count": ifpa_result.get("ifpa_top_point_count"),
                                "ifpa_top_points": ifpa_result.get("ifpa_top_points", []),
                                "ifpa_score_profile": ifpa_result.get("ifpa_score_profile", []),
                                "default_optimized_variant_key": variant_results.get("default_optimized_variant_key"),
                                "optimized_variants": variant_results.get("optimized_variants", {}),
                                "raw_anchor_count": candidate_stats["raw_anchor_count"],
                                "selected_anchor_count": candidate_stats["selected_anchor_count"],
                                "candidate_anchor_count": candidate_stats["candidate_anchor_count"],
                                "valid_scale_interval_count": rejection_summary["valid_count"],
                                "scale_interval_solver_count": len(scale_intervals),
                                "scale_interval_rejection_summary": rejection_summary,
                                "scale_interval_rejection_details": rejection_details,
                                "scale_interval_count": len(scale_intervals),
                                "anchor_measurements": anchor_measurements,
                            },
                        )
                        
                        consistent_pointmap = pointmap * consistent_scale
                        consistent_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                            extrinsics, consistent_scale
                        )

                        SemanticPlaneScaleProcessor._save_scaled_outputs(
                            consistent_output_dir, frame_prefix, consistent_pointmap, consistent_extrinsics, camera_params
                        )
                        variant_scales = {
                            variant_key: variant_scales_for_selection.get(variant_key, default_optimized_scale)
                            for variant_key in optimized_variant_dirs
                        }

                        for variant_key, variant_scale in variant_scales.items():
                            variant_pointmap = pointmap * variant_scale
                            variant_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                                extrinsics, variant_scale
                            )
                            SemanticPlaneScaleProcessor._save_scaled_outputs(
                                optimized_variant_dirs[variant_key],
                                frame_prefix,
                                variant_pointmap,
                                variant_extrinsics,
                                camera_params,
                            )

                        optimized_pointmap = pointmap * default_optimized_scale
                        optimized_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                            extrinsics, default_optimized_scale
                        )
                        SemanticPlaneScaleProcessor._save_scaled_outputs(
                            optimized_output_dir, frame_prefix, optimized_pointmap, optimized_extrinsics, camera_params
                        )
                        SemanticPlaneScaleProcessor._save_scaled_outputs(
                            absolute_output_dir, frame_prefix, optimized_pointmap, optimized_extrinsics, camera_params
                        )
                        if selected_output_dir is not None:
                            selected_scale = selected_scale_info["scale"]
                            selected_pointmap = pointmap * selected_scale
                            selected_extrinsics = SemanticPlaneScaleProcessor._scale_camera_extrinsics(
                                extrinsics, selected_scale
                            )
                            SemanticPlaneScaleProcessor._save_scaled_outputs(
                                selected_output_dir,
                                frame_prefix,
                                selected_pointmap,
                                selected_extrinsics,
                                camera_params,
                            )
        
        return consistent_scale, optimized_scale

class P3ImagePreparer:
    """P3图像准备器"""
    
    @staticmethod
    def prepare_images(vlm_results_dir, sequence_name):
        """准备P3图像"""
        import shutil
        
        p3_images_dir = config.get_path('p3_images', sequence_name)
        file_utils.ensure_directory(p3_images_dir)
        
        # 加载VLM结果
        vlm_results = []
        for json_file in os.listdir(vlm_results_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(vlm_results_dir, json_file)
                vlm_results.append(file_utils.read_json(json_path))
        
        # 复制图像。并行模型会共用同一份 p3_images，必须整文件替换，
        # 不能边写边读。
        for result in vlm_results:
            src_image_path = result["image_path"]
            dest_image_path = os.path.join(p3_images_dir, os.path.basename(src_image_path))
            src_size = os.path.getsize(src_image_path)
            if config.resume_existing_results and os.path.exists(dest_image_path) and os.path.getsize(dest_image_path) == src_size:
                continue
            tmp_path = f"{dest_image_path}.{os.getpid()}.tmp"
            shutil.copy2(src_image_path, tmp_path)
            os.replace(tmp_path, dest_image_path)

class ProcessingPipeline:
    """处理管道"""
    
    def __init__(self):
        """初始化处理管道"""
        self.sequences = []
    
    def prepare_sequences(self):
        """准备序列"""
        self.sequences = sequence_utils.get_sequences(
            config.data_path, 
            config.all_sequences, 
            config.sequence_name
        )

    @staticmethod
    def _should_process_all_frames():
        return config.all_sequences or not bool(config.sequence_name)

    def _get_sequence_frame_ids(self, sequence_path):
        frame_ids = sequence_utils.get_sequence_frame_ids(sequence_path, camera_id=1)
        if self._should_process_all_frames():
            return frame_ids
        if config.frame_count is not None and config.frame_count > 0:
            return frame_ids[:config.frame_count]
        return frame_ids

    def _get_sequence_subsequences(self, sequence_path):
        frame_ids = self._get_sequence_frame_ids(sequence_path)
        subsequence_size = max(1, int(config.subsequence_size))
        return data_utils.generate_frame_id_subsequences(frame_ids, subsequence_size)

    @staticmethod
    def _format_subsequence(frame_ids):
        if not frame_ids:
            return "empty"
        if len(frame_ids) == 1:
            return frame_ids[0]
        return f"{frame_ids[0]} -> {frame_ids[-1]}"

    def _process_vlm_for_sequence(self, sequence_path, sequence_name, device=None):
        log_prefix = _device_prefix(device)
        print(f"\n--- {log_prefix} Processing sequence: {sequence_name} ---")
        subsequences = self._get_sequence_subsequences(sequence_path)

        for i, subsequence_frame_ids in enumerate(subsequences):
            print(
                f"  {log_prefix} Subsequence {i+1}/{len(subsequences)} "
                f"(frames {self._format_subsequence(subsequence_frame_ids)})"
            )

            timer_key = f"VLM Processing (Subseq {i+1})_{sequence_name}"
            timer.start(timer_key)
            VLMProcessor.process_frames(sequence_path, sequence_name, subsequence_frame_ids, device_label=device)
            timer.end(timer_key)

            timer.print_results(timer_key)

    def _prepare_p3_for_sequence(self, sequence_name, device=None):
        print(f"\n--- {_device_prefix(device)} Processing sequence: {sequence_name} ---")
        vlm_results_dir = config.get_path('vlm_results', sequence_name)
        timer_key = f"prepare_p3_{sequence_name}"
        timer.start(timer_key)
        P3ImagePreparer.prepare_images(vlm_results_dir, sequence_name)
        DepthProcessor.refresh_existing_depth_visualizations(sequence_name)
        timer.end(timer_key)
        timer.print_results(timer_key)

    def _sam_outputs_complete(self, sequence_name, frame_ids):
        if not config.resume_existing_results:
            return False
        sam_dir = config.get_path("sam_output", sequence_name)
        if not os.path.isdir(sam_dir):
            return False
        existing = set(os.listdir(sam_dir))
        results_dir = config.get_path("vlm_results", sequence_name)
        needed = set()
        for frame_id in frame_ids:
            json_path = os.path.join(results_dir, f"{frame_id}_camera1.json")
            if not os.path.isfile(json_path):
                return False
            data = file_utils.read_json(json_path)
            has_box = bool(
                data.get("static_height_anchors")
                or data.get("dynamic_height_anchors")
                or data.get("semantic_planes")
            )
            if has_box:
                needed.add(f"{frame_id}_camera1")
        if not needed:
            return True
        return needed.issubset(existing)

    def _depth_outputs_complete(self, sequence_name, frame_ids):
        if not config.resume_existing_results:
            return False
        _, _, relative_scale_dir = DepthProcessor._get_depth_output_dirs(sequence_name, config.depth_model)
        pointmap_dir = os.path.join(relative_scale_dir, "pointmap")
        if not os.path.isdir(pointmap_dir):
            return False
        existing = {
            os.path.splitext(name)[0]
            for name in os.listdir(pointmap_dir)
            if name.endswith(".npy")
            and os.path.getsize(os.path.join(pointmap_dir, name)) > 100
        }
        needed = {f"{frame_id}_camera1" for frame_id in frame_ids}
        return bool(needed) and needed.issubset(existing)

    def _metric_results_parent(self):
        return os.path.join(config.get_path("output_root"), config.dataset_name, "metric_results")

    def _existing_sequence_metric_path(self, sequence_name):
        if not config.resume_existing_results:
            return None
        parent = self._metric_results_parent()
        if not os.path.isdir(parent):
            return None
        prefix = f"{config.depth_model}_"
        matches = []
        for name in os.listdir(parent):
            if not name.startswith(prefix):
                continue
            path = os.path.join(parent, name, "sequences", f"{sequence_name}.json")
            if os.path.isfile(path) and os.path.getsize(path) > 100:
                matches.append(path)
        if not matches:
            return None
        matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        return matches[0]

    def _cleanup_depth_outputs(self, sequence_name):
        output_dir, _, _ = DepthProcessor._get_depth_output_dirs(sequence_name, config.depth_model)
        if not output_dir or not os.path.isdir(output_dir):
            return []
        removed_paths = []
        for entry_name in os.listdir(output_dir):
            entry_path = os.path.join(output_dir, entry_name)
            if os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            elif os.path.isfile(entry_path):
                os.remove(entry_path)
            removed_paths.append(entry_path)
        return removed_paths

    def _copy_sequence_metric_into_current_run(self, sequence_name, existing_path):
        from metric_scale.metrics.metrics import metric_calculator

        current_path = metric_calculator._sequence_result_path(config.depth_model, sequence_name)
        if os.path.abspath(existing_path) == os.path.abspath(current_path):
            return current_path
        file_utils.ensure_directory(os.path.dirname(current_path))
        shutil.copy2(existing_path, current_path)
        txt_src = existing_path[:-5] + ".txt"
        txt_dst = current_path[:-5] + ".txt"
        if os.path.isfile(txt_src):
            shutil.copy2(txt_src, txt_dst)
        return current_path

    def _finalize_sequence_metrics_and_cleanup(self, sequence_path, sequence_name, device=None):
        from metric_scale.metrics.metrics import metric_calculator
        from metric_scale.visualization.visualization import result_visualizer

        log_prefix = _device_prefix(device)
        existing = self._existing_sequence_metric_path(sequence_name)
        if existing:
            self._copy_sequence_metric_into_current_run(sequence_name, existing)
            print(f"  {log_prefix} Skip metrics; already saved for {sequence_name}")
            if config.cleanup_depth_outputs_after_metrics:
                removed_paths = self._cleanup_depth_outputs(sequence_name)
                print(
                    f"  {log_prefix} Cleaned leftover depth outputs for {sequence_name} "
                    f"(removed {len(removed_paths)} entries)"
                )
            return

        result_visualizer.visualize_all_results(
            sequence_path,
            config.frame_count,
            config.depth_model,
            sequence_name,
        )
        sequence_result = metric_calculator.calculate_all_metrics(
            sequence_path,
            config.frame_count,
            config.depth_model,
            sequence_name,
            distance_threshold=config.metric_depth_threshold,
        )
        metric_calculator.save_sequence_result(sequence_result, config.depth_model)
        if config.cleanup_depth_outputs_after_metrics:
            removed_paths = self._cleanup_depth_outputs(sequence_name)
            output_dir, _, _ = DepthProcessor._get_depth_output_dirs(sequence_name, config.depth_model)
            print(
                f"  {log_prefix} Cleaned depth model outputs for sequence: {sequence_name} "
                f"({output_dir}, removed {len(removed_paths)} entries)"
            )

    def _process_sam_for_sequence(self, sequence_path, sequence_name, device=None):
        log_prefix = _device_prefix(device)
        print(f"\n--- {log_prefix} Processing sequence: {sequence_name} ---")
        frame_ids = self._get_sequence_frame_ids(sequence_path)
        if self._sam_outputs_complete(sequence_name, frame_ids):
            print(f"  {log_prefix} Skip SAM; outputs already complete for {sequence_name} ({len(frame_ids)} frames).")
            return
        vlm_results_dir = config.get_path('vlm_results', sequence_name)
        subsequences = self._get_sequence_subsequences(sequence_path)

        for i, subsequence_frame_ids in enumerate(subsequences):
            print(
                f"  {log_prefix} Subsequence {i+1}/{len(subsequences)} "
                f"(frames {self._format_subsequence(subsequence_frame_ids)})"
            )
            timer_key = f"SAM Processing (Subseq {i+1})_{sequence_name}"
            timer.start(timer_key)
            SAMProcessor.process_segmentation(
                vlm_results_dir,
                sequence_name,
                frame_ids=subsequence_frame_ids,
                device=device,
            )
            timer.end(timer_key)
            timer.print_results(timer_key)

    def _process_depth_for_sequence(self, sequence_path, sequence_name, device=None):
        log_prefix = _device_prefix(device)
        print(f"\n--- {log_prefix} Processing sequence: {sequence_name} ---")
        frame_ids = self._get_sequence_frame_ids(sequence_path)
        if self._depth_outputs_complete(sequence_name, frame_ids):
            print(f"  {log_prefix} Skip depth; pointmaps already complete for {sequence_name} ({len(frame_ids)} frames).")
            return
        subsequences = self._get_sequence_subsequences(sequence_path)

        current_start_index = 0
        for i, subsequence_frame_ids in enumerate(subsequences):
            print(
                f"  {log_prefix} Subsequence {i+1}/{len(subsequences)} "
                f"(frames {self._format_subsequence(subsequence_frame_ids)})"
            )

            timer_key = f"Depth Estimation ({config.depth_model}) (Subseq {i+1})_{sequence_name}"
            timer.start(timer_key)
            DepthProcessor.run_depth_estimation(
                sequence_name,
                subsequence_frame_ids,
                start_index=current_start_index,
                device=device,
                sequence_path=sequence_path,
            )
            timer.end(timer_key)

            timer.print_results(timer_key)
            current_start_index += len(subsequence_frame_ids)

    def _process_scale_for_sequence(self, sequence_path, sequence_name, device=None):
        log_prefix = _device_prefix(device)
        print(f"\n--- {log_prefix} Processing sequence: {sequence_name} ---")
        subsequences = self._get_sequence_subsequences(sequence_path)
        vlm_results_dir = config.get_path('vlm_results', sequence_name)

        active_profile = None
        if config.sequence_difficulty_enabled:
            active_profile = config.activate_semantic_scale_sequence_profile(sequence_name)
            if active_profile is None:
                print(
                    f"  {log_prefix} Sequence difficulty profile: disabled "
                    "(dataset summary unavailable; using base semantic_scale config)"
                )
            elif active_profile.get("mode") == "best_variant_vs_baseline":
                baseline_rel = active_profile.get("baseline_rel")
                baseline_delta1 = active_profile.get("baseline_delta1")
                baseline_text = "rel=N/A, delta1=N/A"
                if baseline_rel is not None and baseline_delta1 is not None:
                    baseline_text = (
                        f"rel={baseline_rel:.4f}, delta1={baseline_delta1:.4f}"
                    )
                print(
                    f"  {log_prefix} Sequence difficulty profile: "
                    f"{active_profile['profile_name']} "
                    f"(baseline {active_profile['baseline_variant']}: {baseline_text}, "
                    f"easy if any candidate delta1 gain >= "
                    f"{active_profile['easy_d1_gain_threshold']:.4f})"
                )
                print(
                    f"  {log_prefix} Difficulty reason: "
                    f"{active_profile.get('classification_reason')}"
                )
                if active_profile.get("trigger_variant"):
                    trigger_metrics = active_profile.get("trigger_variant_metrics") or {}
                    trigger_rel = trigger_metrics.get("rel")
                    trigger_delta1 = trigger_metrics.get("delta1")
                    trigger_d1_gain = trigger_metrics.get("d1_gain_vs_baseline")
                    trigger_rel_delta = trigger_metrics.get("rel_delta_vs_baseline")
                    print(
                        f"  {log_prefix} Difficulty trigger: "
                        f"{active_profile['trigger_variant']} "
                        f"(rel={trigger_rel:.4f}, delta1={trigger_delta1:.4f}, "
                        f"rel_delta={trigger_rel_delta:+.4f}, d1_gain={trigger_d1_gain:+.4f})"
                    )
                if active_profile.get("per_variant_metrics"):
                    print(
                        f"  {log_prefix} Difficulty candidates: "
                        + ", ".join(
                            f"{name}(rel={value['rel']:.4f}, delta1={value['delta1']:.4f}, "
                            f"rel_delta={value['rel_delta_vs_baseline']:+.4f}, "
                            f"d1_gain={value['d1_gain_vs_baseline']:+.4f})"
                            for name, value in active_profile["per_variant_metrics"].items()
                        )
                    )
                if active_profile.get("missing_variant_metrics"):
                    print(
                        f"  {log_prefix} Difficulty missing metrics: "
                        + ", ".join(
                            f"{name}={reason}"
                            for name, reason in active_profile["missing_variant_metrics"].items()
                        )
                    )
                if active_profile.get("hard_variants"):
                    print(
                        f"  {log_prefix} Difficulty hard variants: "
                        + ", ".join(active_profile["hard_variants"])
                    )
            else:
                metric_value_text = (
                    "N/A"
                    if active_profile["metric_value"] is None
                    else f"{active_profile['metric_value']:.6f}"
                )
                print(
                    f"  {log_prefix} Sequence difficulty profile: "
                    f"{active_profile['profile_name']} "
                    f"(metric {active_profile['metric_variant']}."
                    f"{active_profile['metric_scope']}.{active_profile['metric_name']}="
                    f"{metric_value_text}, "
                    f"easy={active_profile['easy_threshold']}, "
                    f"mid={active_profile['mid_threshold']})"
                )
            if active_profile is not None and active_profile.get("metric_status") != "ok":
                print(
                    f"  {log_prefix} Difficulty metric status: "
                    f"{active_profile['metric_status']} "
                    f"(frames_evaluated={active_profile.get('frames_evaluated')})"
                )
            if active_profile is not None and active_profile.get("distance_bin"):
                print(
                    f"  {log_prefix} Difficulty metric distance bin: "
                    f"{active_profile['distance_bin']}"
                )
            model_manager.release_model('semantic_plane_scale')
        else:
            config.clear_semantic_scale_sequence_profile()

        sequence_override = config.activate_semantic_scale_sequence_override(sequence_name)
        if sequence_override:
            print(f"  {log_prefix} Explicit scale anchor override: {sequence_override}")

        DepthProcessor.refresh_existing_depth_visualizations(sequence_name)
        try:
            for i, subsequence_frame_ids in enumerate(subsequences):
                print(
                    f"  {log_prefix} Subsequence {i+1}/{len(subsequences)} "
                    f"(frames {self._format_subsequence(subsequence_frame_ids)})"
                )

                timer_key = f"Semantic Plane Scale Estimation (Subseq {i+1})_{sequence_name}"
                timer.start(timer_key)
                SemanticPlaneScaleProcessor.estimate_scale(
                    vlm_results_dir,
                    config.depth_model,
                    config.use_shared_scale,
                    sequence_name,
                    subsequence_frame_ids,
                )
                timer.end(timer_key)

                timer.print_results(timer_key)
        finally:
            model_manager.release_model('semantic_plane_scale')
            config.clear_semantic_scale_sequence_profile()


    def process_vlm(self):
        """处理VLM"""
        print("\n=== Stage 1: VLM Processing ===")
        
        for sequence_path, sequence_name in self.sequences:
            self._process_vlm_for_sequence(sequence_path, sequence_name)
        
        # 释放模型
        model_manager.release_model('scale_anchor')
        model_manager.release_model('vision')
    
    def prepare_p3_images(self):
        """准备P3图像"""
        print("\n=== Stage 2: Prepare P3 Images ===")
        
        for sequence_path, sequence_name in self.sequences:
            self._prepare_p3_for_sequence(sequence_name)
    
    def process_sam(self):
        """处理SAM"""
        print("\n=== Stage 3: SAM Processing ===")

        for sequence_path, sequence_name in self.sequences:
            self._process_sam_for_sequence(sequence_path, sequence_name)
        
        # 释放模型
        model_manager.release_model('sam_tracker')

    def _should_parallelize_sam(self):
        if not config.parallel:
            return False
        if not torch.cuda.is_available():
            return False
        return len(self._get_parallel_gpu_ids()) > 1 and len(self._build_sam_tasks()) > 1

    def _build_sam_tasks(self):
        tasks = []
        for sequence_path, sequence_name in self.sequences:
            frame_ids = sequence_utils.get_sequence_frame_ids(sequence_path, camera_id=1)
            subsequences = data_utils.generate_frame_id_subsequences(
                frame_ids,
                config.subsequence_size,
                config.frame_count,
            )
            vlm_results_dir = config.get_path('vlm_results', sequence_name)

            for subsequence_index, subsequence_frame_ids in enumerate(subsequences, start=1):
                tasks.append(
                    {
                        "sequence_path": sequence_path,
                        "sequence_name": sequence_name,
                        "frame_ids": subsequence_frame_ids,
                        "vlm_results_dir": vlm_results_dir,
                        "subsequence_index": subsequence_index,
                        "subsequence_total": len(subsequences),
                    }
                )
        return tasks

    def _process_sam_parallel(self):
        gpu_ids = self._get_parallel_gpu_ids()
        tasks = self._build_sam_tasks()

        print(f"Parallel SAM scheduling enabled: {len(tasks)} tasks across GPUs {gpu_ids}")

        tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
        for task_index, task in enumerate(tasks):
            gpu_id = gpu_ids[task_index % len(gpu_ids)]
            tasks_by_gpu[gpu_id].append(task)

        for gpu_id in gpu_ids:
            assigned_tasks = tasks_by_gpu[gpu_id]
            if not assigned_tasks:
                continue
            descriptions = [
                f"{task['sequence_name']}[{task['subsequence_index']}/{task['subsequence_total']}]"
                for task in assigned_tasks
            ]
            print(f"  GPU {gpu_id}: {', '.join(descriptions)}")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        expected_results = 0

        for gpu_id in gpu_ids:
            assigned_tasks = tasks_by_gpu[gpu_id]
            if not assigned_tasks:
                continue
            expected_results += len(assigned_tasks)
            process = ctx.Process(
                target=run_sam_tasks_on_gpu,
                args=(gpu_id, assigned_tasks, result_queue),
            )
            process.start()
            processes.append(process)

        error_messages = []
        received_results = 0
        while received_results < expected_results:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
                    break
                continue

            received_results += 1
            if result["status"] == "ok":
                print(
                    f"  GPU {result['gpu_id']} finished "
                    f"{result['sequence_name']}[{result['subsequence_index']}/{result['subsequence_total']}] "
                    f"in {result['duration']:.2f}s"
                )
            else:
                error_messages.append(result)

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0 and not error_messages:
                error_messages.append(
                    {
                        "status": "error",
                        "gpu_id": None,
                        "error": f"SAM worker exited with code {process.exitcode}",
                        "traceback": "",
                    }
                )

        if error_messages:
            first_error = error_messages[0]
            raise RuntimeError(
                "Parallel SAM processing failed: "
                f"{first_error['error']}\n{first_error.get('traceback', '')}"
            )
    
    def process_depth(self):
        """处理深度估计"""
        print("\n=== Stage 4: Depth Estimation ===")

        for sequence_path, sequence_name in self.sequences:
            self._process_depth_for_sequence(sequence_path, sequence_name)
        
        # 释放模型
        model_manager.release_model(f'depth_{config.depth_model}')

    def _should_parallelize_depth(self):
        if not config.parallel:
            return False
        if not torch.cuda.is_available():
            return False
        return len(self._get_parallel_gpu_ids()) > 1 and len(self._build_depth_tasks()) > 1

    @staticmethod
    def _get_parallel_gpu_ids():
        if not torch.cuda.is_available():
            return []
        device_count = torch.cuda.device_count()
        max_gpus = config.max_gpus if config.max_gpus and config.max_gpus > 0 else device_count
        return list(range(min(device_count, max_gpus)))

    def _build_depth_tasks(self):
        tasks = []
        for sequence_path, sequence_name in self.sequences:
            frame_ids = sequence_utils.get_sequence_frame_ids(sequence_path, camera_id=1)
            subsequences = data_utils.generate_frame_id_subsequences(
                frame_ids,
                config.subsequence_size,
                config.frame_count,
            )

            current_start_index = 0
            for subsequence_index, subsequence_frame_ids in enumerate(subsequences, start=1):
                tasks.append(
                    {
                        "sequence_name": sequence_name,
                        "frame_ids": subsequence_frame_ids,
                        "start_index": current_start_index,
                        "model_type": config.depth_model,
                        "subsequence_index": subsequence_index,
                        "subsequence_total": len(subsequences),
                    }
                )
                current_start_index += len(subsequence_frame_ids)
        return tasks

    def _process_depth_parallel(self):
        gpu_ids = self._get_parallel_gpu_ids()
        tasks = self._build_depth_tasks()

        print(f"Parallel depth scheduling enabled: {len(tasks)} tasks across GPUs {gpu_ids}")

        tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
        for task_index, task in enumerate(tasks):
            gpu_id = gpu_ids[task_index % len(gpu_ids)]
            tasks_by_gpu[gpu_id].append(task)

        for gpu_id in gpu_ids:
            assigned_tasks = tasks_by_gpu[gpu_id]
            if not assigned_tasks:
                continue
            descriptions = [
                f"{task['sequence_name']}[{task['subsequence_index']}/{task['subsequence_total']}]"
                for task in assigned_tasks
            ]
            print(f"  GPU {gpu_id}: {', '.join(descriptions)}")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        expected_results = 0

        for gpu_id in gpu_ids:
            assigned_tasks = tasks_by_gpu[gpu_id]
            if not assigned_tasks:
                continue
            expected_results += len(assigned_tasks)
            process = ctx.Process(
                target=run_depth_tasks_on_gpu,
                args=(gpu_id, assigned_tasks, result_queue),
            )
            process.start()
            processes.append(process)

        completed_results = []
        error_messages = []

        received_results = 0
        while received_results < expected_results:
            try:
                result = result_queue.get(timeout=1.0)
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
                    break
                continue

            received_results += 1
            if result["status"] == "ok":
                completed_results.append(result)
                print(
                    f"  GPU {result['gpu_id']} finished "
                    f"{result['sequence_name']}[{result['subsequence_index']}/{result['subsequence_total']}] "
                    f"in {result['duration']:.2f}s"
                )
            else:
                error_messages.append(result)

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0 and not error_messages:
                error_messages.append(
                    {
                        "status": "error",
                        "gpu_id": None,
                        "error": f"Worker exited with code {process.exitcode}",
                        "traceback": "",
                    }
                )

        if error_messages:
            first_error = error_messages[0]
            raise RuntimeError(
                "Parallel depth estimation failed: "
                f"{first_error['error']}\n{first_error.get('traceback', '')}"
            )
    
    def process_semantic_plane_scale(self):
        """处理语义平面尺度估计"""
        print("\n=== Stage 5: Semantic Plane Scale Estimation ===")
        
        for sequence_path, sequence_name in self.sequences:
            self._process_scale_for_sequence(sequence_path, sequence_name)
        
        # 释放模型
        model_manager.release_model('semantic_plane_scale')

    def _should_parallelize_sequences(self):
        if config._get_section("runtime").get("depth_only", False):
            return False
        if not config.parallel:
            return False
        if not torch.cuda.is_available():
            return False
        if len(self._get_parallel_gpu_ids()) <= 1:
            return False
        if len(self.sequences) <= 1:
            return False
        return True

    def _build_sequence_tasks(self):
        return [
            {
                "sequence_path": sequence_path,
                "sequence_name": sequence_name,
            }
            for sequence_path, sequence_name in self.sequences
        ]

    def _process_sequences_parallel(self):
        gpu_ids = self._get_parallel_gpu_ids()
        tasks = self._build_sequence_tasks()

        print(f"Parallel sequence scheduling enabled: {len(tasks)} sequences across GPUs {gpu_ids}")

        tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
        for task_index, task in enumerate(tasks):
            gpu_id = gpu_ids[task_index % len(gpu_ids)]
            tasks_by_gpu[gpu_id].append(task)

        for gpu_id in gpu_ids:
            assigned_tasks = tasks_by_gpu[gpu_id]
            if not assigned_tasks:
                continue
            descriptions = [task["sequence_name"] for task in assigned_tasks]
            print(f"  GPU {gpu_id}: {', '.join(descriptions)}")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        expected_results = 0
        try:
            for gpu_id in gpu_ids:
                assigned_tasks = tasks_by_gpu[gpu_id]
                if not assigned_tasks:
                    continue
                expected_results += len(assigned_tasks)
                process = ctx.Process(
                    target=run_sequence_pipeline_on_gpu,
                    args=(gpu_id, assigned_tasks, result_queue),
                )
                process.start()
                processes.append(process)

            error_messages = []
            received_results = 0
            while received_results < expected_results:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    if any(process.exitcode not in (None, 0) for process in processes):
                        break
                    continue

                received_results += 1
                if result["status"] == "ok":
                    print(
                        f"  GPU {result['gpu_id']} finished "
                        f"{result['sequence_name']} in {result['duration']:.2f}s"
                    )
                else:
                    error_messages.append(result)

            for process in processes:
                process.join()

            for process in processes:
                if process.exitcode != 0 and not error_messages:
                    error_messages.append(
                        {
                            "status": "error",
                            "gpu_id": None,
                            "error": f"Sequence worker exited with code {process.exitcode}",
                            "traceback": "",
                        }
                    )

            if error_messages:
                first_error = error_messages[0]
                raise RuntimeError(
                    "Parallel sequence processing failed: "
                    f"{first_error['error']}\n{first_error.get('traceback', '')}"
                )
        except KeyboardInterrupt:
            _terminate_processes(processes, "sequence")
            raise
        finally:
            if not result_queue._closed:
                result_queue.close()
                result_queue.cancel_join_thread()
    
    def _process_sequence_task(self, sequence_path, sequence_name, device=None):
        if config._get_section("runtime").get("depth_only", False):
            import tempfile
            frame_ids = self._get_sequence_frame_ids(sequence_path)
            vlm_dir = config.get_path("vlm_results", sequence_name)
            sources = []
            for frame_id in frame_ids:
                name = f"{frame_id}_camera1"
                image_path = os.path.join(sequence_path, name + ".jpg")
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(image_path)
                if not os.path.isfile(os.path.join(vlm_dir, name + ".json")):
                    raise FileNotFoundError(f"Missing existing VLM result: {name}")
                sources.append(image_path)
            # Preserve the old depth-only mode: a fresh input directory avoids stale frames.
            with tempfile.TemporaryDirectory(prefix="metric_depth_only_") as input_dir:
                for image_path in sources:
                    shutil.copy2(image_path, os.path.join(input_dir, os.path.basename(image_path)))
                old_input = config._config["paths"].get("p3_images")
                config._config["paths"]["p3_images"] = input_dir
                try:
                    self._process_depth_for_sequence(sequence_path, sequence_name, device=device)
                finally:
                    if old_input is None:
                        config._config["paths"].pop("p3_images", None)
                    else:
                        config._config["paths"]["p3_images"] = old_input
            return
        if self._existing_sequence_metric_path(sequence_name):
            self._finalize_sequence_metrics_and_cleanup(sequence_path, sequence_name, device=device)
            return
        self._process_sam_for_sequence(sequence_path, sequence_name, device=device)
        device_suffix = "" if device is None else f"_{device}"
        model_manager.release_model(f"sam_tracker{device_suffix}")
        self._prepare_p3_for_sequence(sequence_name, device=device)
        self._process_depth_for_sequence(sequence_path, sequence_name, device=device)
        self._process_scale_for_sequence(sequence_path, sequence_name, device=device)
        self._finalize_sequence_metrics_and_cleanup(sequence_path, sequence_name, device=device)
        
    def run(self):
        """运行整个处理流程"""
        print("=" * 80)
        print("Processing Stage Timing Results:")
        print("=" * 80)
        
        self.prepare_sequences()

        # A clean checkout must be able to build VLM JSON before SAM/depth/scale.
        # When resume_existing_results is enabled, the VLM stage skips saved JSON.
        if not config._get_section("runtime").get("depth_only", False):
            self.process_vlm()

        if self._should_parallelize_sequences():
            self._process_sequences_parallel()
        else:
            for sequence_path, sequence_name in self.sequences:
                self._process_sequence_task(sequence_path, sequence_name)
        
        print("=" * 80)
        
        # 释放所有模型
        model_manager.release_all()

# 全局处理管道实例
pipeline = ProcessingPipeline()
