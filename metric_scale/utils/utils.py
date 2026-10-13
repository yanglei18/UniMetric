import time
import os
import json
import re
import torch
import gc
import numpy as np
from metric_scale.config.path_config import ensure_dir

class Timer:
    """计时器"""
    
    def __init__(self):
        """初始化计时器"""
        self.start_times = {}
        self.end_times = {}
    
    def start(self, name):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end(self, name):
        """结束计时"""
        self.end_times[name] = time.time()
    
    def get_duration(self, name):
        """获取持续时间"""
        if name in self.start_times and name in self.end_times:
            return self.end_times[name] - self.start_times[name]
        return 0
    
    def print_results(self, stage_name, total_time=None):
        """打印计时结果"""
        exec_time = self.get_duration(stage_name)
        if total_time:
            percentage = (exec_time / total_time) * 100
            print(f"{stage_name:<40} {exec_time:.4f}s ({percentage:.2f}%)")
        else:
            print(f"{stage_name:<40} {exec_time:.4f}s")

class FileUtils:
    """文件工具"""
    
    @staticmethod
    def read_json(file_path):
        """读取JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data, file_path, indent=2):
        """写入JSON文件"""
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    
    @staticmethod
    def list_files(directory, pattern=None):
        """列出目录中的文件"""
        files = []
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if pattern is None or pattern in file:
                    files.append(file_path)
        return files
    
    @staticmethod
    def ensure_directory(directory):
        """确保目录存在"""
        ensure_dir(directory)

    @staticmethod
    def natural_sort_key(value):
        """自然排序键，兼容未补零数字文件名。"""
        parts = re.split(r'(\d+)', str(value))
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    @staticmethod
    def natural_sorted(values):
        """按自然顺序排序。"""
        return sorted(values, key=FileUtils.natural_sort_key)

class MemoryUtils:
    """内存工具"""
    
    @staticmethod
    def release_memory(obj=None):
        """释放内存"""
        if obj is not None:
            del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_gpu_memory():
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return {
                'total': torch.cuda.get_device_properties(0).total_memory,
                'used': torch.cuda.memory_allocated(0),
                'free': torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
            }
        return None

class DataUtils:
    """数据处理工具"""
    
    @staticmethod
    def generate_subsequences(frame_count, subsequence_size):
        """生成子序列"""
        subsequences = []
        num_subsequences = (frame_count + subsequence_size - 1) // subsequence_size
        
        for i in range(num_subsequences):
            start_frame = i * subsequence_size
            end_frame = min((i + 1) * subsequence_size, frame_count)
            actual_subsequence_size = end_frame - start_frame
            subsequences.append((start_frame, actual_subsequence_size))
        
        return subsequences

    @staticmethod
    def generate_frame_id_subsequences(frame_ids, subsequence_size, max_frames=None):
        """按真实帧ID切分子序列。"""
        normalized_frame_ids = list(frame_ids)
        if max_frames is not None and max_frames > 0:
            normalized_frame_ids = normalized_frame_ids[:max_frames]

        subsequences = []
        for start_idx in range(0, len(normalized_frame_ids), subsequence_size):
            chunk = normalized_frame_ids[start_idx:start_idx + subsequence_size]
            if chunk:
                subsequences.append(chunk)
        return subsequences
    
    @staticmethod
    def get_frame_ids(start_frame, count):
        """获取帧ID列表"""
        return [f"{start_frame + i:05d}" for i in range(count)]
    
    @staticmethod
    def calculate_scale_factors(img_width, img_height, pointmap_width, pointmap_height):
        """计算坐标映射的缩放因子"""
        return {
            'scale_x': pointmap_width / img_width,
            'scale_y': pointmap_height / img_height
        }

class SequenceUtils:
    """序列处理工具"""

    @staticmethod
    def _normalize_sequence_names(sequence_name):
        if sequence_name is None:
            return tuple()
        if isinstance(sequence_name, (list, tuple, set)):
            raw_items = sequence_name
        else:
            sequence_name = str(sequence_name).strip()
            if not sequence_name or sequence_name.lower() == "none":
                return tuple()
            raw_items = sequence_name.split(",") if "," in sequence_name else [sequence_name]

        sequence_names = []
        seen = set()
        for item in raw_items:
            name = str(item).strip()
            if not name or name.lower() == "none":
                continue
            if name in seen:
                continue
            sequence_names.append(name)
            seen.add(name)
        return tuple(sequence_names)
    
    @staticmethod
    def get_sequences(data_path, all_sequences=False, sequence_name=None):
        """获取序列列表"""
        sequences = []

        sequence_names = SequenceUtils._normalize_sequence_names(sequence_name)

        if all_sequences or not sequence_names:
            # 获取所有序列目录
            sequence_dirs = file_utils.natural_sorted(
                [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            )
            for sequence_dir in sequence_dirs:
                sequence_path = os.path.join(data_path, sequence_dir)
                sequences.append((sequence_path, sequence_dir))
        else:
            # 处理指定序列列表
            for sequence_dir in sequence_names:
                sequence_path = os.path.join(data_path, sequence_dir)
                sequences.append((sequence_path, sequence_dir))
        
        return sequences

    @staticmethod
    def get_sequence_frame_ids(sequence_path, camera_id=1):
        """从序列目录里枚举真实存在的帧ID，不假设编号连续。"""
        frame_pattern = re.compile(rf"^(?P<frame_id>.+)_camera{camera_id}\.(jpg|jpeg|png)$", re.IGNORECASE)
        frame_ids = []

        for filename in os.listdir(sequence_path):
            file_path = os.path.join(sequence_path, filename)
            if not os.path.isfile(file_path):
                continue
            match = frame_pattern.match(filename)
            if match:
                frame_ids.append(match.group("frame_id"))

        return file_utils.natural_sorted(set(frame_ids))

# 兼容旧代码的函数
def print_timing_results(stage_name, start_time, end_time, total_time=None):
    """
    Print timing results for a processing stage
    
    Args:
        stage_name: Name of the processing stage
        start_time: Start time of the stage
        end_time: End time of the stage
        total_time: Total execution time for percentage calculation (optional)
    """
    exec_time = end_time - start_time
    if total_time:
        percentage = (exec_time / total_time) * 100
        print(f"{stage_name:<40} {exec_time:.4f}s ({percentage:.2f}%)")
    else:
        print(f"{stage_name:<40} {exec_time:.4f}s")

def release_resources(model):
    """
    Release model resources
    
    Args:
        model: Model to release
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def camera_to_lidar_points(camera_points, extrinsic):
    """
    将相机坐标系下的点云转换为LiDAR坐标系下的点云
    
    Args:
        camera_points: 相机坐标系下的点云，形状为 (N, 3)
        extrinsic: 相机外参矩阵，形状为 (4, 4) 或 (3, 4)
    
    Returns:
        np.ndarray: LiDAR坐标系下的点云，形状为 (N, 3)
    """
    # 确保外参矩阵是3x4格式用于赋值
    if extrinsic.shape == (4, 4):
        # 如果是4x4矩阵，只取前3行前4列
        Tr_camera_to_lidar = extrinsic[:3, :4]
    else:
        # 直接使用3x4矩阵
        Tr_camera_to_lidar = extrinsic
    # 将3D点转换为齐次坐标 (x, y, z, 1)
    points_camera_extend = np.concatenate((camera_points, np.ones((camera_points.shape[0], 1))), axis=1)
    # 创建相机到LiDAR的变换矩阵
    camera2lidar = np.eye(4)
    camera2lidar[:3, :4] = Tr_camera_to_lidar
    # 应用外参变换
    points_lidar = np.matmul(camera2lidar, points_camera_extend.T).T[:, :3]
    return points_lidar

# 全局工具实例
timer = Timer()
file_utils = FileUtils()
memory_utils = MemoryUtils()
data_utils = DataUtils()
sequence_utils = SequenceUtils()
