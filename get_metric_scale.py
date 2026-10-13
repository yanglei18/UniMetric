#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import time

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from metric_scale.config.config import config
from metric_scale.processing.processing import pipeline
from metric_scale.visualization.visualization import result_visualizer
from metric_scale.metrics.metrics import metric_calculator
from metric_scale.utils.utils import sequence_utils

class DemoRunner:
    """演示运行器"""
    
    @staticmethod
    def run_demo():
        """运行演示"""
        from metric_scale.processing.frame_loader import FrameLoader
        from metric_scale.models.models import model_manager
        
        print("\n=== Running Demo ===")
        
        # 加载帧数据
        sequence_path = config.get_path('vehicle_data')
        sequence_name = config.sequence_names[0] if config.sequence_names else ""
        frame_data = FrameLoader(os.path.join(sequence_path, sequence_name), camera_id=1, frame_id="00000")
        
        if frame_data.process_data():
            frame_data.save_result()
        
        # 初始化VLM尺度估计器
        scale_estimator = model_manager.initialize_vlm_scale_estimator()
        
        # 示例尺度对用于测试
        test_scale_pairs = []
        
        # 可选地添加一些基于场景中已知对象的测试尺度对
        for obj in frame_data.object_list:
            if obj["type"] == "Car" and len(obj["relative_dimension"]) > 0:
                # 假设一辆典型汽车的长度为4.5米
                test_scale_pairs.append((obj["relative_dimension"][0], obj["gt_dimension"][0]))
                break  # 只使用第一辆车进行测试
        
        # 估计有无尺度对的度量尺度
        print("\n=== Estimating scale without scale pairs ===")
        metric_scale_without_pairs = scale_estimator.estimate_metric_scale(frame_data.object_list, frame_data.image_path)
        
        if test_scale_pairs:
            print("\n=== Estimating scale with test scale pairs ===")
            metric_scale_with_pairs = scale_estimator.estimate_metric_scale(frame_data.object_list, frame_data.image_path, test_scale_pairs)
            print("\nNo test scale pairs available, skipping joint estimation")
            metric_scale_with_pairs = metric_scale_without_pairs
        
        # 打印最终结果
        print(f"\n=== Final Results ===")
        print(f"Scale without pairs: {metric_scale_without_pairs:.4f}")
        if test_scale_pairs:
            print(f"Scale with pairs: {metric_scale_with_pairs:.4f}")
        
        # 应用尺度到相对维度和位置
        print("\nScaled object dimensions and locations (using best scale):")
        final_scale = metric_scale_with_pairs
        for i, obj in enumerate(frame_data.object_list):
            scaled_dim = obj["relative_dimension"] * final_scale
            scaled_loc = obj["relative_location"] * final_scale
            print(f"Object {i+1} ({obj['type']}):")
            print(f"  Relative dimension: {obj['relative_dimension']}")
            print(f"  Scaled dimension: {scaled_dim}")
            print(f"  GT dimension: {obj['gt_dimension']}")
            print(f"  Relative location: {obj['relative_location']}")
            print(f"  Scaled location: {scaled_loc}")
            print(f"  GT location: {obj['gt_location']}")
        
        # 释放模型
        model_manager.release_all()

class MainRunner:
    """主运行器"""

    DEPTH_OUTPUT_DIR_NAMES = {
        "da3_outputs",
        "pi3_outputs",
        "vggt_outputs",
        "vggt_omega_outputs",
        "dust3r_outputs",
        "mast3r_outputs",
        "must3r_outputs",
        "scal3r_outputs",
        "stream3r_outputs",
        "lingbot_map_outputs",
        "dvlt_outputs",
        "4rc_outputs",
        "moge_output",
    }

    DEPTH_OUTPUT_PATH_KEYS = (
        "da3_output",
        "pi3_output",
        "vggt_output",
        "vggt_omega_output",
        "dust3r_output",
        "must3r_output",
        "scal3r_output",
        "stream3r_output",
        "lingbot_map_output",
        "dvlt_output",
        "fourrc_output",
    )

    @staticmethod
    def _get_depth_output_dir(sequence_name):
        depth_model = config.depth_model
        if depth_model == "pi3_long":
            return config.get_path("pi3_output", sequence_name)
        if depth_model == "dust3r_long":
            return config.get_path("dust3r_output", sequence_name)
        if depth_model == "must3r_long":
            return config.get_path("must3r_output", sequence_name)
        if depth_model == "scal3r_long":
            return config.get_path("scal3r_output", sequence_name)
        if depth_model == "stream3r_long":
            return config.get_path("stream3r_output", sequence_name)
        if depth_model == "lingbot_map_long":
            return config.get_path("lingbot_map_output", sequence_name)
        if depth_model == "dvlt_long":
            return config.get_path("dvlt_output", sequence_name)
        if depth_model == "4rc_long":
            return config.get_path("fourrc_output", sequence_name)
        if depth_model == "vggt_omega_long":
            return config.get_path("vggt_omega_output", sequence_name)
        if depth_model == "vggt_long":
            return config.get_path("vggt_output", sequence_name)
        if depth_model == "moge":
            da3_output = config.get_path("da3_output", sequence_name)
            return os.path.join(os.path.dirname(da3_output), "moge_output")
        if depth_model == "da3_long":
            return config.get_path("da3_output", sequence_name)
        raise ValueError(f"Unsupported depth model: {depth_model}")

    @staticmethod
    def _iter_depth_output_dirs(sequence_name):
        output_dirs = []
        seen = set()

        for path_key in MainRunner.DEPTH_OUTPUT_PATH_KEYS:
            output_dir = config.get_path(path_key, sequence_name)
            if output_dir and output_dir not in seen:
                output_dirs.append(output_dir)
                seen.add(output_dir)

        da3_output = config.get_path("da3_output", sequence_name)
        if da3_output:
            moge_output = os.path.join(os.path.dirname(da3_output), "moge_output")
            if moge_output not in seen:
                output_dirs.append(moge_output)
                seen.add(moge_output)

        current_output_dir = MainRunner._get_depth_output_dir(sequence_name)
        sequence_output_dir = os.path.dirname(current_output_dir) if current_output_dir else None
        if sequence_output_dir and os.path.isdir(sequence_output_dir):
            for entry_name in os.listdir(sequence_output_dir):
                if entry_name not in MainRunner.DEPTH_OUTPUT_DIR_NAMES:
                    continue
                output_dir = os.path.join(sequence_output_dir, entry_name)
                if output_dir not in seen:
                    output_dirs.append(output_dir)
                    seen.add(output_dir)

        return [output_dir for output_dir in output_dirs if os.path.isdir(output_dir)]

    @staticmethod
    def cleanup_post_metric_outputs(sequence_name):
        removed_paths = []
        for output_dir in MainRunner._iter_depth_output_dirs(sequence_name):
            for entry_name in os.listdir(output_dir):
                if entry_name == "relative_scale":
                    continue

                entry_path = os.path.join(output_dir, entry_name)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                    removed_paths.append(entry_path)
                elif os.path.isfile(entry_path):
                    os.remove(entry_path)
                    removed_paths.append(entry_path)

        return removed_paths

    @staticmethod
    def cleanup_current_depth_output_contents(sequence_name):
        """Remove current depth-model outputs after metrics have been saved."""
        output_dir = MainRunner._get_depth_output_dir(sequence_name)
        if not output_dir or not os.path.isdir(output_dir):
            return []

        removed_paths = []
        for entry_name in os.listdir(output_dir):
            entry_path = os.path.join(output_dir, entry_name)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
                removed_paths.append(entry_path)
            elif os.path.isfile(entry_path):
                os.remove(entry_path)
                removed_paths.append(entry_path)

        return removed_paths
    
    @staticmethod
    def run():
        """运行主流程"""
        total_start_time = time.time()
        
        print("\n=== Starting Metric Scale Estimation ===")
        print(f"Processing {config.frame_count} frames with subsequence size {config.subsequence_size}")
        print(f"Using depth model: {config.depth_model}")
        print(f"Shared scale mode: {config.use_shared_scale}")
        config.print_summary()
        
        # 运行处理管道
        pipeline.run()
        
        # 处理可视化和指标计算
        MainRunner.process_visualization_and_metrics()
        
        total_end_time = time.time()
        total_exec_time = total_end_time - total_start_time
        print(f"\n=== Total Execution Time ===")
        print(f"Total time: {total_exec_time:.4f}s")
    
    @staticmethod
    def process_visualization_and_metrics():
        """处理可视化和指标计算"""
        # 获取序列列表
        sequences = sequence_utils.get_sequences(
            config.data_path, 
            config.all_sequences, 
            config.sequence_name
        )

        sequence_results = []
        dataset_metrics_start = time.time()
        for sequence_path, sequence_name in sequences:
            # 可视化结果
            result_visualizer.visualize_all_results(sequence_path, config.frame_count, config.depth_model, sequence_name)
            
            # 计算指标
            sequence_result = metric_calculator.calculate_all_metrics(
                sequence_path,
                config.frame_count,
                config.depth_model,
                sequence_name,
                distance_threshold=config.metric_depth_threshold,
            )
            sequence_results.append(sequence_result)
            metric_calculator.save_sequence_result(sequence_result, config.depth_model)
            metric_calculator.save_dataset_summary(
                sequence_results,
                config.depth_model,
                total_sequences=len(sequences),
                elapsed=time.time() - dataset_metrics_start,
            )

            if config.cleanup_depth_outputs_after_metrics:
                removed_paths = MainRunner.cleanup_current_depth_output_contents(sequence_name)
                output_dir = MainRunner._get_depth_output_dir(sequence_name)
                print(
                    f"Cleaned depth model outputs for sequence: {sequence_name} "
                    f"({output_dir}, removed {len(removed_paths)} entries)"
                )

if __name__ == "__main__":
    # 运行主流程
    MainRunner.run()
    # 可选：运行演示
    # DemoRunner.run_demo()
