import torch
import gc
from metric_scale.models.mllm_agent import ModelConfig, get_model
from metric_scale.models.vlm_semantic_scale_anchor import VLMSemanticScaleAnchor
from metric_scale.models.semantic_scale_estimator import SemanticPlaneScaleEstimator
from metric_scale.models.vlm_scale_estimator import VLMScaleEstimator
from metric_scale.models.sam2_tracker import SAM2Tracker
from metric_scale.models.pi3x_long_depth_estimator import Pi3LongDepthEstimator
from metric_scale.models.da3_long_depth_estimator import Da3LongDepthEstimator
from metric_scale.models.vggt_long_depth_estimator import VGGTLongDepthEstimator
from metric_scale.models.vggt_omega_long_depth_estimator import VGGTOmegaLongDepthEstimator
from metric_scale.models.moge_depth_estimator import MogeDepthEstimator
from metric_scale.models.dust3r_long_depth_estimator import Dust3rLongDepthEstimator
from metric_scale.models.must3r_long_depth_estimator import Must3rLongDepthEstimator
from metric_scale.models.scal3r_long_depth_estimator import Scal3rLongDepthEstimator
from metric_scale.models.lingbot_map_long_depth_estimator import LingBotMapLongDepthEstimator
from metric_scale.models.dvlt_long_depth_estimator import DVLTLongDepthEstimator
from metric_scale.models.fourrc_long_depth_estimator import FourRCLongDepthEstimator
from metric_scale.models.stream3r_long_depth_estimator import STream3RLongDepthEstimator
from metric_scale.config.config import config

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.models = {}
        self._initialized = {}
    
    def initialize_vision_model(self):
        """初始化视觉模型"""
        if 'vision' not in self._initialized:
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
            self.models['vision'] = vision_model
            self._initialized['vision'] = True
        return self.models['vision']
    
    def initialize_scale_anchor(self):
        """初始化尺度锚点模型"""
        if 'scale_anchor' not in self._initialized:
            vision_model = self.initialize_vision_model()
            scale_anchor = VLMSemanticScaleAnchor(vision_model)
            self.models['scale_anchor'] = scale_anchor
            self._initialized['scale_anchor'] = True
        return self.models['scale_anchor']
    
    def initialize_sam_tracker(self, device=None):
        """初始化SAM跟踪器"""
        device_suffix = "" if device is None else f"_{device}"
        key = f"sam_tracker{device_suffix}"
        if key not in self._initialized:
            tracker = SAM2Tracker(
                model_cfg=config.get_path('sam_config'),
                sam2_checkpoint=config.get_path('sam_checkpoint'),
                device=device,
            )
            self.models[key] = tracker
            self._initialized[key] = True
        return self.models[key]
    
    def initialize_depth_estimator(self, model_type=None, device=None):
        """初始化深度估计模型"""
        model_type = model_type or config.depth_model
        device_suffix = "" if device is None else f"_{device}"
        key = f'depth_{model_type}{device_suffix}'
        
        if key not in self._initialized:
            if model_type == "pi3_long":
                depth_estimator = Pi3LongDepthEstimator(
                    model_type='Pi3X',
                    device=device,
                    pretrained_model=config.pi3_weights,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "vggt_long":
                depth_estimator = VGGTLongDepthEstimator(
                    pretrained_model=config.vggt_weights,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "vggt_omega_long":
                depth_estimator = VGGTOmegaLongDepthEstimator(
                    project_root=config.vggt_omega_project_root,
                    checkpoint=config.vggt_omega_checkpoint,
                    device=device,
                    image_resolution=config.vggt_omega_image_resolution,
                    preprocess_mode=config.vggt_omega_preprocess_mode,
                    geometry_intrinsics_source=config.vggt_omega_geometry_intrinsics_source,
                    enable_alignment=config.vggt_omega_enable_alignment,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "moge":
                depth_estimator = MogeDepthEstimator(
                    pretrained_model=config.moge_model_name,
                    device=device,
                )
            elif model_type == "dust3r_long":
                depth_estimator = Dust3rLongDepthEstimator(
                    model_name=config.dust3r_model_name,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "must3r_long":
                depth_estimator = Must3rLongDepthEstimator(
                    model_name=config.must3r_model_name,
                    retrieval_ckpt_path=config.must3r_retrieval_ckpt_path,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "scal3r_long":
                depth_estimator = Scal3rLongDepthEstimator(
                    project_root=config.scal3r_project_root,
                    config_path=config.scal3r_config_path,
                    checkpoint=config.scal3r_checkpoint,
                    use_native_confidence=config.scal3r_use_native_confidence,
                    device=device,
                    block_size=config.long_chunk_size,
                    overlap_size=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "lingbot_map_long":
                depth_estimator = LingBotMapLongDepthEstimator(
                    project_root=config.lingbot_map_project_root,
                    checkpoint=config.lingbot_map_checkpoint,
                    device=device,
                    mode=config.lingbot_map_mode,
                    image_size=config.lingbot_map_image_size,
                    patch_size=config.lingbot_map_patch_size,
                    num_scale_frames=config.lingbot_map_num_scale_frames,
                    keyframe_interval=config.lingbot_map_keyframe_interval,
                    max_frame_num=config.lingbot_map_max_frame_num,
                    kv_cache_sliding_window=config.lingbot_map_kv_cache_sliding_window,
                    camera_num_iterations=config.lingbot_map_camera_num_iterations,
                    use_sdpa=config.lingbot_map_use_sdpa,
                    offload_to_cpu=config.lingbot_map_offload_to_cpu,
                    window_size=config.lingbot_map_window_size,
                    overlap_size=config.lingbot_map_overlap_size,
                    overlap_keyframes=config.lingbot_map_overlap_keyframes,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "dvlt_long":
                depth_estimator = DVLTLongDepthEstimator(
                    project_root=config.dvlt_project_root,
                    checkpoint=config.dvlt_checkpoint,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "4rc_long":
                depth_estimator = FourRCLongDepthEstimator(
                    project_root=config.fourrc_project_root,
                    checkpoint=config.fourrc_checkpoint,
                    device=device,
                    image_size=config.fourrc_image_size,
                    patch_size=config.fourrc_patch_size,
                    dtype=config.fourrc_dtype,
                    use_center_as_anchor=config.fourrc_use_center_as_anchor,
                    preprocess_mode=config.fourrc_preprocess_mode,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "stream3r_long":
                depth_estimator = STream3RLongDepthEstimator(
                    project_root=config.stream3r_project_root,
                    checkpoint=config.stream3r_checkpoint,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            elif model_type == "da3_long":
                depth_estimator = Da3LongDepthEstimator(
                    model_name=config.da3_model_name,
                    device=device,
                    chunk_size=config.long_chunk_size,
                    overlap=config.long_overlap,
                    delete_temp_files=config.long_delete_temp_files,
                    temp_root=config.long_temp_root,
                )
            else:
                raise ValueError(f"Unsupported depth model: {model_type}")
            
            self.models[key] = depth_estimator
            self._initialized[key] = True
        
        return self.models[key]
    
    def initialize_semantic_plane_scale_estimator(self):
        """初始化语义平面尺度估计器"""
        if 'semantic_plane_scale' not in self._initialized:
            scale_estimator = SemanticPlaneScaleEstimator()
            self.models['semantic_plane_scale'] = scale_estimator
            self._initialized['semantic_plane_scale'] = True
        return self.models['semantic_plane_scale']
    
    def initialize_vlm_scale_estimator(self):
        """初始化VLM尺度估计器"""
        if 'vlm_scale' not in self._initialized:
            vision_model = self.initialize_vision_model()
            scale_estimator = VLMScaleEstimator(vision_model)
            self.models['vlm_scale'] = scale_estimator
            self._initialized['vlm_scale'] = True
        return self.models['vlm_scale']
    
    def get_model(self, model_name):
        """获取模型"""
        if model_name == 'vision':
            return self.initialize_vision_model()
        elif model_name == 'scale_anchor':
            return self.initialize_scale_anchor()
        elif model_name == 'sam_tracker':
            return self.initialize_sam_tracker()
        elif model_name == 'semantic_plane_scale':
            return self.initialize_semantic_plane_scale_estimator()
        elif model_name == 'vlm_scale':
            return self.initialize_vlm_scale_estimator()
        elif model_name.startswith('depth_'):
            depth_model_type = model_name[len('depth_'):]
            return self.initialize_depth_estimator(depth_model_type)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def release_model(self, model_name):
        """释放模型"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self._initialized:
                del self._initialized[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def release_all(self):
        """释放所有模型"""
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.release_model(model_name)
    
    def __enter__(self):
        """进入上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        self.release_all()

class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_depth_estimator(model_type, device=None):
        """创建深度估计模型"""
        if model_type == "pi3_long":
            return Pi3LongDepthEstimator(
                model_type='Pi3X',
                device=device,
                pretrained_model=config.pi3_weights,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "vggt_long":
            return VGGTLongDepthEstimator(
                pretrained_model=config.vggt_weights,
                device=device,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "vggt_omega_long":
            return VGGTOmegaLongDepthEstimator(
                project_root=config.vggt_omega_project_root,
                checkpoint=config.vggt_omega_checkpoint,
                device=device,
                image_resolution=config.vggt_omega_image_resolution,
                preprocess_mode=config.vggt_omega_preprocess_mode,
                geometry_intrinsics_source=config.vggt_omega_geometry_intrinsics_source,
                enable_alignment=config.vggt_omega_enable_alignment,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "moge":
            return MogeDepthEstimator(
                pretrained_model=config.moge_model_name,
                device=device,
            )
        elif model_type == "dust3r_long":
            return Dust3rLongDepthEstimator(
                model_name=config.dust3r_model_name,
                device=device,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "must3r_long":
            return Must3rLongDepthEstimator(
                model_name=config.must3r_model_name,
                retrieval_ckpt_path=config.must3r_retrieval_ckpt_path,
                device=device,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "scal3r_long":
            return Scal3rLongDepthEstimator(
                project_root=config.scal3r_project_root,
                config_path=config.scal3r_config_path,
                checkpoint=config.scal3r_checkpoint,
                use_native_confidence=config.scal3r_use_native_confidence,
                device=device,
                block_size=config.long_chunk_size,
                overlap_size=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "lingbot_map_long":
            return LingBotMapLongDepthEstimator(
                project_root=config.lingbot_map_project_root,
                checkpoint=config.lingbot_map_checkpoint,
                device=device,
                mode=config.lingbot_map_mode,
                image_size=config.lingbot_map_image_size,
                patch_size=config.lingbot_map_patch_size,
                num_scale_frames=config.lingbot_map_num_scale_frames,
                keyframe_interval=config.lingbot_map_keyframe_interval,
                max_frame_num=config.lingbot_map_max_frame_num,
                kv_cache_sliding_window=config.lingbot_map_kv_cache_sliding_window,
                camera_num_iterations=config.lingbot_map_camera_num_iterations,
                use_sdpa=config.lingbot_map_use_sdpa,
                offload_to_cpu=config.lingbot_map_offload_to_cpu,
                window_size=config.lingbot_map_window_size,
                overlap_size=config.lingbot_map_overlap_size,
                overlap_keyframes=config.lingbot_map_overlap_keyframes,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "dvlt_long":
            return DVLTLongDepthEstimator(
                project_root=config.dvlt_project_root,
                checkpoint=config.dvlt_checkpoint,
                device=device,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "4rc_long":
            return FourRCLongDepthEstimator(
                project_root=config.fourrc_project_root,
                checkpoint=config.fourrc_checkpoint,
                device=device,
                image_size=config.fourrc_image_size,
                patch_size=config.fourrc_patch_size,
                dtype=config.fourrc_dtype,
                use_center_as_anchor=config.fourrc_use_center_as_anchor,
                preprocess_mode=config.fourrc_preprocess_mode,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        elif model_type == "da3_long":
            return Da3LongDepthEstimator(
                model_name=config.da3_model_name,
                device=device,
                chunk_size=config.long_chunk_size,
                overlap=config.long_overlap,
                delete_temp_files=config.long_delete_temp_files,
                temp_root=config.long_temp_root,
            )
        else:
            raise ValueError(f"Unsupported depth model: {model_type}")
    
    @staticmethod
    def create_sam_tracker(device=None):
        """创建SAM跟踪器"""
        return SAM2Tracker(
            model_cfg=config.get_path('sam_config'),
            sam2_checkpoint=config.get_path('sam_checkpoint'),
            device=device,
        )

# 全局模型管理器实例
model_manager = ModelManager()
