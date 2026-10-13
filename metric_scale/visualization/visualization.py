import os
import cv2
import numpy as np
from metric_scale.processing.frame_loader import FrameLoader
from metric_scale.config.config import config
from metric_scale.utils.utils import camera_to_lidar_points
from metric_scale.utils.utils import file_utils, sequence_utils

class Visualizer:
    """可视化器基类"""
    
    @staticmethod
    def ensure_directory(directory):
        """确保目录存在"""
        file_utils.ensure_directory(directory)

class BEVVisualizer(Visualizer):
    """BEV可视化器"""

    @staticmethod
    def _get_selected_frame_ids(sequence_path, frame_count):
        frame_ids = sequence_utils.get_sequence_frame_ids(sequence_path, camera_id=1)
        if frame_count is not None and frame_count > 0:
            return frame_ids[:frame_count]
        return frame_ids
    
    @staticmethod
    def visualize_results(sequence_path, frame_count, depth_model, sequence_name, frame_cache=None):
        """可视化结果"""
        if not config.enable_visualization:
            return

        for frame_id in BEVVisualizer._get_selected_frame_ids(sequence_path, frame_count):
            frame_data = None if frame_cache is None else frame_cache.get(frame_id)
            if frame_data is None:
                frame_data = FrameLoader(sequence_path, camera_id=1, frame_id=frame_id)
                if not frame_data.process_data():
                    continue

            bev_image_gt = frame_data.project_points_to_bev(frame_data.points)
            frame_name = frame_data.image_path.split("/")[-1].split('.')[0]
            lidar2camera = frame_data.lidar2camera
            
            # 获取相对尺度点云路径
            if depth_model == "pi3_long":
                relative_scale_dir = os.path.join(config.get_path('pi3_output', sequence_name), "relative_scale")
                absolute_scale_dir = os.path.join(config.get_path('pi3_output', sequence_name), "absolute_scale")
                visualization_dir = os.path.join(config.get_path('pi3_output', sequence_name), "visualization")
            elif depth_model == "scal3r_long":
                relative_scale_dir = config.get_path('scal3r_relative_scale', sequence_name)
                absolute_scale_dir = config.get_path('scal3r_absolute_scale', sequence_name)
                visualization_dir = os.path.join(config.get_path('scal3r_output', sequence_name), "visualization")
            elif depth_model == "lingbot_map_long":
                relative_scale_dir = config.get_path('lingbot_map_relative_scale', sequence_name)
                absolute_scale_dir = config.get_path('lingbot_map_absolute_scale', sequence_name)
                visualization_dir = os.path.join(config.get_path('lingbot_map_output', sequence_name), "visualization")
            elif depth_model == "vggt_omega_long":
                relative_scale_dir = config.get_path('vggt_omega_relative_scale', sequence_name)
                absolute_scale_dir = config.get_path('vggt_omega_absolute_scale', sequence_name)
                visualization_dir = os.path.join(config.get_path('vggt_omega_output', sequence_name), "visualization")
            elif depth_model == "stream3r_long":
                relative_scale_dir = os.path.join(config.get_path('stream3r_output', sequence_name), "relative_scale")
                absolute_scale_dir = os.path.join(config.get_path('stream3r_output', sequence_name), "absolute_scale")
                visualization_dir = os.path.join(config.get_path('stream3r_output', sequence_name), "visualization")
            elif depth_model == "vggt_long":
                relative_scale_dir = config.get_path('vggt_relative_scale', sequence_name)
                absolute_scale_dir = config.get_path('vggt_absolute_scale', sequence_name)
                visualization_dir = os.path.join(config.get_path('vggt_output', sequence_name), "visualization")
            elif depth_model == "da3_long":
                relative_scale_dir = config.get_path('da3_relative_scale', sequence_name)
                absolute_scale_dir = config.get_path('da3_absolute_scale', sequence_name)
                visualization_dir = os.path.join(config.get_path('da3_output', sequence_name), "visualization")
            else:
                raise ValueError(f"Unsupported depth model: {depth_model}")
            
            # 加载相对尺度点云并可视化
            pointmap_path = os.path.join(relative_scale_dir, "pointmap", frame_name + ".npy")
            if os.path.exists(pointmap_path):
                points = np.load(pointmap_path).reshape(-1, 3)
                lidar_points = camera_to_lidar_points(points, np.linalg.inv(lidar2camera))
                bev_image_pred = frame_data.project_points_to_bev(lidar_points)
                blended_bev_rel = cv2.addWeighted(bev_image_gt, 0.5, bev_image_pred, 0.5, 0)
            else:
                blended_bev_rel = bev_image_gt
            
            # 加载绝对尺度点云并可视化
            pointmap_path = os.path.join(absolute_scale_dir, "pointmap", frame_name + ".npy")
            if os.path.exists(pointmap_path):
                points = np.load(pointmap_path).reshape(-1, 3)
                lidar_points = camera_to_lidar_points(points, np.linalg.inv(lidar2camera))
                bev_image_pred = frame_data.project_points_to_bev(lidar_points)
                blended_bev_abs = cv2.addWeighted(bev_image_gt, 0.5, bev_image_pred, 0.5, 0)
            else:
                blended_bev_abs = bev_image_gt
            
            # 水平拼接相对和绝对尺度的BEV图像
            blended_bev = np.hstack((blended_bev_rel, blended_bev_abs))
            
            # 保存可视化结果
            Visualizer.ensure_directory(visualization_dir)
            cv2.imwrite(os.path.join(visualization_dir, frame_name + ".jpg"), blended_bev)

class DepthVisualizer(Visualizer):
    """深度可视化器"""
    
    @staticmethod
    def visualize_depth_map(depth_map, output_path):
        """可视化深度图"""
        # 归一化深度值到0-255范围
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(output_path, depth_colormap)
    
    @staticmethod
    def visualize_point_cloud(points, output_path, color=None):
        """可视化点云"""
        # 这里可以实现点云可视化，例如使用Open3D
        pass

class ScaleVisualizer(Visualizer):
    """尺度可视化器"""
    
    @staticmethod
    def visualize_scale_intervals(scale_intervals, output_path):
        """可视化尺度区间"""
        # 创建一个简单的可视化，显示尺度区间分布
        import matplotlib.pyplot as plt
        
        if not scale_intervals:
            return
        
        scales = [interval[0] for interval in scale_intervals]
        confidences = [interval[1] for interval in scale_intervals]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(scales, confidences, alpha=0.5)
        plt.xlabel('Scale')
        plt.ylabel('Confidence')
        plt.title('Scale Intervals Distribution')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()

class ResultVisualizer(Visualizer):
    """结果可视化器"""
    
    @staticmethod
    def visualize_all_results(sequence_path, frame_count, depth_model, sequence_name, frame_cache=None):
        """可视化所有结果"""
        if not config.enable_visualization:
            return

        print(f"\n=== Visualizing results for sequence: {sequence_name} ===")
        print("=" * 80)
        print("Visualization & Statistics Timing Results:")
        print("=" * 80)
        
        # 可视化BEV结果
        import time
        start_time = time.time()
        BEVVisualizer.visualize_results(
            sequence_path,
            frame_count,
            depth_model,
            sequence_name,
            frame_cache=frame_cache,
        )
        end_time = time.time()
        print(f"Results Visualization: {end_time - start_time:.4f}s")
        # 这里可以添加其他可视化功能

# 全局可视化器实例
bev_visualizer = BEVVisualizer()
depth_visualizer = DepthVisualizer()
scale_visualizer = ScaleVisualizer()
result_visualizer = ResultVisualizer()
