import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
from metric_scale.config.config import config
from metric_scale.config.path_config import (
    ensure_dir,
    get_output_dir,
    get_sam_aligned_anchor_dir,
    get_sam_output_dir,
)

# Try to import Numba for acceleration
try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available, running without acceleration")

# Numba-accelerated helper functions if available
if NUMBA_AVAILABLE:
    @njit(parallel=False)
    def numba_search_consistent_scale(scale_intervals, candidates):
        """
        Numba-accelerated version of search_consistent_scale
        """
        best_count = 0
        best_scale = candidates[0]
        
        for s in candidates:
            count = 0
            for interval in scale_intervals:
                if interval[0] <= s <= interval[1]:
                    count += 1
            if count > best_count:
                best_count = count
                best_scale = s
        
        return best_scale, best_count

class SemanticPlaneScaleEstimator:
    MIN_RELATIVE_HEIGHT = 1e-4

    def __init__(self):
        # Initialize mask cache
        self.mask_cache = {}
        self.image_size_cache = {}
        # Initialize plane fit cache
        self.plane_fit_cache = {}
        # Initialize alignment cache
        self.alignment_cache = {}
        self.enable_visualization = config.enable_visualization
        self.write_ply = config.enable_visualization
        self.relative_height_mode = config.relative_height_mode
        self.dynamic_anchor_height_scale = config.dynamic_anchor_height_scale
        self.shared_plane_mode = config.shared_plane_mode
        self.height_anchor_source = config.height_anchor_source
        self.frame_parallel_workers = config.frame_parallel_workers
        self.use_depth_confidence = config.use_depth_confidence
        self.use_anchor_quality_filter = config.use_anchor_quality_filter
        self.use_depth_geometry_filter = config.use_depth_geometry_filter
        self.depth_geometry_min_anchor_points = config.depth_geometry_min_anchor_points
        self.depth_geometry_min_relative_height = config.depth_geometry_min_relative_height
        self.use_vlm_anchor_confidence_filter = config.use_vlm_anchor_confidence_filter
        self.vlm_anchor_conf_thresholds = {
            "dynamic": config.vlm_dynamic_anchor_confidence_threshold,
            "static": config.vlm_static_anchor_confidence_threshold,
        }
        self.anchor_quality_min_percentile = config.anchor_quality_min_percentile
        self.anchor_quality_max_percentile = config.anchor_quality_max_percentile
        self.plane_conf_percentile = config.plane_conf_percentile
        self.plane_normal_outlier_filter_enabled = config.plane_normal_outlier_filter_enabled
        self.plane_normal_filter_min_frames = config.plane_normal_filter_min_frames
        self.plane_normal_filter_max_angle_deg = config.plane_normal_filter_max_angle_deg
        self.plane_normal_filter_max_points = config.plane_normal_filter_max_points
        self.plane_normal_filter_ransac_iterations = config.plane_normal_filter_ransac_iterations
        self.plane_normal_filter_ransac_threshold = config.plane_normal_filter_ransac_threshold
        self.plane_normal_filter_ransac_confidence = config.plane_normal_filter_ransac_confidence
        self.priority_modulation_strength = config.optimized_scale_priority_modulation_strength
        self.distance_weight_map = {
            "near": config.distance_weight_near,
            "middle": config.distance_weight_middle,
            "distant": config.distance_weight_distant,
        }
        self.optimizer_dynamic_classes = set(config.optimizer_dynamic_classes)
        self.optimizer_static_classes = set(config.optimizer_static_classes)
        self.default_optimized_variant = config.default_optimized_variant
        self.enabled_optimized_variants = set(config.enabled_optimized_variants)
        self.optimized_scale_solver = config.optimized_scale_solver
        self.weighted_search_mode = config.weighted_search_mode
        self.consensus_prior_mode = config.consensus_prior_mode
        self.consensus_prior_strength = config.consensus_prior_strength
        self.consensus_prior_interval_inside_ratio = config.consensus_prior_interval_inside_ratio
        self.consensus_solver = config.consensus_solver
        self.ifpa_num_bins = config.ifpa_num_bins
        self.ifpa_use_weight = config.ifpa_use_weight
        self.bi_keep_ratio = config.bi_keep_ratio
        self.bi_min_segment_width = config.bi_min_segment_width
        self.last_bi_result = None
        self.last_ifpa_result = None
        self.last_consensus_result = None
        self.last_variant_results = None
        self.last_optimization_result = None
        self.last_variant_results = None

    def _depth_geometry_min_points(self):
        if not self.use_depth_geometry_filter:
            return 1
        return self.depth_geometry_min_anchor_points

    def _has_valid_relative_height(self, ref_height):
        if ref_height is None:
            return False
        threshold = self.depth_geometry_min_relative_height
        if not self.use_depth_geometry_filter:
            threshold = 0.0
        return ref_height > threshold

    @staticmethod
    def _array_digest(array):
        contiguous = np.ascontiguousarray(array)
        return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()

    def _make_plane_fit_cache_key(self, points, iterations, threshold, confidence):
        return (
            points.shape,
            str(points.dtype),
            self._array_digest(points),
            iterations,
            threshold,
            confidence,
        )

    def _make_alignment_cache_key(self, point_cloud, plane_normal):
        return (
            point_cloud.shape,
            str(point_cloud.dtype),
            self._array_digest(point_cloud),
            tuple(np.round(plane_normal, 8)),
        )

    @staticmethod
    def _downsample_points(points, max_points):
        if points is None or len(points) == 0 or len(points) <= max_points:
            return points
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[indices]

    @staticmethod
    def _normalize_plane_normal(normal):
        if normal is None:
            return None
        normal = np.asarray(normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-8 or not np.isfinite(norm):
            return None
        return normal / norm

    def _fit_plane_normal_for_filter(self, points):
        if points is None or len(points) < 3:
            return None
        sampled_points = self._downsample_points(points, self.plane_normal_filter_max_points)
        normal, _ = self.fit_plane_ransac(
            sampled_points,
            iterations=self.plane_normal_filter_ransac_iterations,
            threshold=self.plane_normal_filter_ransac_threshold,
            confidence=self.plane_normal_filter_ransac_confidence,
        )
        return self._normalize_plane_normal(normal)

    def _estimate_plane_normal_reference(self, normals):
        normalized_normals = [
            self._normalize_plane_normal(normal)
            for normal in normals
        ]
        normalized_normals = [
            normal for normal in normalized_normals
            if normal is not None
        ]
        if not normalized_normals:
            return None

        normal_array = np.stack(normalized_normals, axis=0)
        affinity = np.abs(normal_array @ normal_array.T)
        medoid_index = int(np.argmax(np.sum(affinity, axis=1)))
        reference = normal_array[medoid_index]
        oriented_normals = np.array([
            normal if np.dot(normal, reference) >= 0 else -normal
            for normal in normal_array
        ])
        averaged = np.mean(oriented_normals, axis=0)
        return self._normalize_plane_normal(averaged)

    def _filter_frame_data_for_global_plane(self, frame_data):
        plane_frame_data = [
            data for data in frame_data
            if len(data["valid_plane_points"]) > 0
        ]
        if (
            not self.plane_normal_outlier_filter_enabled
            or len(plane_frame_data) < self.plane_normal_filter_min_frames
        ):
            return plane_frame_data

        normals = [
            data.get("frame_plane_normal")
            for data in plane_frame_data
            if data.get("frame_plane_normal") is not None
        ]
        if len(normals) < self.plane_normal_filter_min_frames:
            return plane_frame_data

        reference_normal = self._estimate_plane_normal_reference(normals)
        if reference_normal is None:
            return plane_frame_data

        accepted_frame_data = []
        for data in plane_frame_data:
            normal = self._normalize_plane_normal(data.get("frame_plane_normal"))
            if normal is None:
                continue
            cos_angle = abs(float(np.clip(np.dot(normal, reference_normal), -1.0, 1.0)))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            data["plane_normal_filter_angle_deg"] = angle_deg
            data["plane_normal_filter_accepted"] = (
                angle_deg <= self.plane_normal_filter_max_angle_deg
            )
            if data["plane_normal_filter_accepted"]:
                accepted_frame_data.append(data)

        # Safety fallback: never make the scale solver fail solely because all
        # candidate plane normals were rejected.
        if not accepted_frame_data:
            return plane_frame_data
        return accepted_frame_data

    @staticmethod
    def _compute_visualization_bounds(point_sets):
        valid_sets = [points for points in point_sets if points is not None and len(points) > 0]
        if not valid_sets:
            center = np.zeros(3, dtype=np.float64)
            half_extent = 1.0
            return center, half_extent

        stacked = np.vstack(valid_sets)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        center = 0.5 * (mins + maxs)
        half_extent = max(float(np.max(maxs - mins)) * 0.55, 1.0)
        return center, half_extent

    @staticmethod
    def _parse_vlm_weight(value):
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return float(np.clip(parsed, 0.0, 1.0))

    @staticmethod
    def _parse_vlm_priority(value):
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 1:
            return None
        return parsed

    @staticmethod
    def _priority_upper_bound(anchor_group):
        anchor_group = str(anchor_group or "").strip().lower()
        if anchor_group == "dynamic":
            return 12
        if anchor_group == "static":
            return 12
        return 12

    def _compute_priority_factor(self, priority, anchor_group):
        if priority is None:
            return 1.0
        upper_bound = self._priority_upper_bound(anchor_group)
        if upper_bound <= 1:
            return 1.0
        normalized_priority = np.clip((priority - 1) / (upper_bound - 1), 0.0, 1.0)
        return float(1.0 - self.priority_modulation_strength * normalized_priority)

    def _parse_distance_level(self, value):
        normalized = str(value or "").strip().lower()
        if normalized not in {"near", "middle", "distant"}:
            return None
        return normalized

    def _compute_distance_factor(self, distance_level):
        if distance_level is None:
            return 1.0
        return float(self.distance_weight_map.get(distance_level, 1.0))

    def _extract_anchor_weight_info(self, anchor):
        anchor_group = anchor.get("anchor_group", "unknown")
        type_weight = self._parse_vlm_weight(anchor.get("type_weight"))
        quality_weight = self._parse_vlm_weight(anchor.get("confidence"))
        priority = self._parse_vlm_priority(anchor.get("priority"))
        distance_level = self._parse_distance_level(anchor.get("distance_level"))
        priority_factor = self._compute_priority_factor(priority, anchor_group)
        distance_factor = self._compute_distance_factor(distance_level)
        effective_type_weight = 1.0 if type_weight is None else type_weight
        modulated_type_weight = effective_type_weight * priority_factor * distance_factor
        effective_quality_weight = 1.0 if quality_weight is None else quality_weight
        combined_weight = modulated_type_weight * effective_quality_weight
        return {
            "type_weight": type_weight,
            "quality_weight": quality_weight,
            "priority": priority,
            "distance_level": distance_level,
            "distance_factor": distance_factor,
            "priority_factor": priority_factor,
            "modulated_type_weight": modulated_type_weight,
            "combined_weight": combined_weight,
            "weight_formula": "type_weight * priority_factor * distance_factor * confidence",
            "uses_vlm_weights": (
                type_weight is not None
                or quality_weight is not None
                or priority is not None
                or distance_level is not None
            ),
        }

    def _save_batch_alignment_visualization(
        self,
        aligned_plane_points,
        aligned_anchor_data,
        image_path,
        sequence_name=None,
    ):
        if not self.enable_visualization:
            return None
        frame_name = os.path.basename(image_path).split(".")[0]
        vis_dir = os.path.join(
            get_output_dir(sequence_name, dataset_name=config.dataset_name),
            "semantic_plane_scale_visualizations",
            "batch_aligned",
        )
        ensure_dir(vis_dir)

        sampled_plane_points = self._downsample_points(aligned_plane_points, 12000)
        sampled_anchor_data = []
        for anchor_data in aligned_anchor_data:
            sampled_anchor_data.append(
                {
                    "label": anchor_data["label"],
                    "points": self._downsample_points(anchor_data["points"], 2500),
                }
            )

        point_sets = [sampled_plane_points] + [item["points"] for item in sampled_anchor_data]
        center, half_extent = self._compute_visualization_bounds(point_sets)

        fig = plt.figure(figsize=(14, 6))
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax_xz = fig.add_subplot(1, 2, 2)

        if sampled_plane_points is not None and len(sampled_plane_points) > 0:
            ax_3d.scatter(
                sampled_plane_points[:, 0],
                sampled_plane_points[:, 1],
                sampled_plane_points[:, 2],
                s=1,
                c="#B0B0B0",
                alpha=0.18,
                label="plane",
            )
            ax_xz.scatter(
                sampled_plane_points[:, 0],
                sampled_plane_points[:, 2],
                s=1,
                c="#B0B0B0",
                alpha=0.18,
                label="plane",
            )

        color_values = plt.cm.tab10(np.linspace(0, 1, max(len(sampled_anchor_data), 1)))
        for idx, anchor_data in enumerate(sampled_anchor_data):
            anchor_points = anchor_data["points"]
            if anchor_points is None or len(anchor_points) == 0:
                continue
            color = color_values[idx % len(color_values)]
            ax_3d.scatter(
                anchor_points[:, 0],
                anchor_points[:, 1],
                anchor_points[:, 2],
                s=4,
                color=color,
                alpha=0.85,
                label=anchor_data["label"],
            )
            ax_xz.scatter(
                anchor_points[:, 0],
                anchor_points[:, 2],
                s=6,
                color=color,
                alpha=0.9,
                label=anchor_data["label"],
            )

        ax_3d.set_title("3D aligned view")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim(center[0] - half_extent, center[0] + half_extent)
        ax_3d.set_ylim(center[1] - half_extent, center[1] + half_extent)
        ax_3d.set_zlim(center[2] - half_extent, center[2] + half_extent)
        ax_3d.view_init(elev=24, azim=-56)

        ax_xz.set_title("XZ projection")
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_xlim(center[0] - half_extent, center[0] + half_extent)
        ax_xz.set_ylim(center[2] - half_extent, center[2] + half_extent)
        ax_xz.grid(True, alpha=0.2)
        ax_xz.set_aspect("equal", adjustable="box")

        if sampled_anchor_data:
            ax_3d.legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"Aligned plane and anchors | frame={frame_name} | anchors={len(sampled_anchor_data)}",
            fontsize=12,
        )
        fig.tight_layout()

        save_path = os.path.join(vis_dir, f"{frame_name}_aligned_batch.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def search_consistent_scale(self, scale_intervals, num_candidates=100):
        """
        Search for the scale factor that is consistent with the most objects.
        
        Args:
            scale_intervals (list): List of scale intervals for each object
            num_candidates (int): Number of candidate scale factors to evaluate
            
        Returns:
            tuple: (best_scale, best_count) - the scale factor that covers the most objects
        """
        if not scale_intervals:
            return 1.0, 0
        
        # Determine the global range to search
        global_min = min(interval[0] for interval in scale_intervals)
        global_max = max(interval[1] for interval in scale_intervals)
        
        if global_min >= global_max:
            return global_min, len(scale_intervals)
        
        # Generate candidate scale factors
        candidates = np.linspace(global_min, global_max, num_candidates)
        
        # Use Numba-accelerated version if available
        if NUMBA_AVAILABLE:
            # Convert scale_intervals to numpy array for Numba
            scale_intervals_np = np.array(scale_intervals, dtype=np.float64)
            return numba_search_consistent_scale(scale_intervals_np, candidates)
        else:
            # Count how many intervals each candidate is in
            best_count = 0
            best_scale = global_min
            
            for s in candidates:
                count = sum(1 for interval in scale_intervals if interval[0] <= s <= interval[1])
                if count > best_count:
                    best_count = count
                    best_scale = s
            
            return best_scale, best_count

    def _normalize_scale_constraints(self, scale_constraints):
        normalized_constraints = []
        for item in scale_constraints or []:
            if isinstance(item, dict):
                interval = item.get("interval")
                weight = float(item.get("weight", 1.0))
            else:
                interval = item
                weight = 1.0

            if interval is None or len(interval) != 2:
                continue

            left = float(interval[0])
            right = float(interval[1])
            if not np.isfinite(left) or not np.isfinite(right):
                continue
            if left > right:
                left, right = right, left
            if weight <= 0.0 or not np.isfinite(weight):
                continue

            normalized_constraints.append(
                {
                    "interval": [left, right],
                    "weight": weight,
                    "object_class": item.get("object_class") if isinstance(item, dict) else None,
                    "anchor_group": item.get("anchor_group") if isinstance(item, dict) else None,
                    "object_id": item.get("object_id") if isinstance(item, dict) else None,
                }
            )
        return normalized_constraints

    def _build_atomic_segments(self, scale_constraints, min_segment_width=None):
        if min_segment_width is None:
            min_segment_width = self.bi_min_segment_width
        endpoints = []
        for item in scale_constraints:
            left, right = item["interval"]
            endpoints.append(left)
            endpoints.append(right)

        endpoints = sorted(set(endpoints))
        if len(endpoints) < 2:
            return []

        segments = []
        for idx in range(len(endpoints) - 1):
            left = endpoints[idx]
            right = endpoints[idx + 1]
            if right - left < min_segment_width:
                continue
            segments.append(
                {
                    "interval": [left, right],
                    "midpoint": 0.5 * (left + right),
                }
            )
        return segments

    def _compute_segment_weights(self, scale_constraints, segments):
        weighted_segments = []
        for segment in segments:
            midpoint = segment["midpoint"]
            coverage_weight = 0.0
            covering_constraints = 0
            for item in scale_constraints:
                left, right = item["interval"]
                if left <= midpoint <= right:
                    coverage_weight += item["weight"]
                    covering_constraints += 1

            weighted_segments.append(
                {
                    "interval": segment["interval"],
                    "midpoint": midpoint,
                    "coverage_weight": coverage_weight,
                    "covering_constraints": covering_constraints,
                }
            )
        return weighted_segments

    def _empty_bi_result(self):
        return {
            "solver": "bi",
            "consensus_scale": None,
            "consensus_interval": None,
            "bi_scale": None,
            "bi_interval": None,
            "bi_segments": [],
            "keep_threshold": 0.0,
            "total_weight": 0.0,
            "kept_segment_count": 0,
            "all_segment_count": 0,
        }

    def _empty_ifpa_result(self):
        return {
            "solver": "ifpa",
            "consensus_scale": None,
            "consensus_interval": None,
            "ifpa_scale": None,
            "ifpa_interval": None,
            "ifpa_intervals": [],
            "ifpa_best_score": None,
            "ifpa_grid_size": self.ifpa_num_bins,
            "ifpa_use_weight": self.ifpa_use_weight,
            "ifpa_top_point_count": 0,
            "ifpa_top_points": [],
            "ifpa_score_profile": [],
        }

    @staticmethod
    def _merge_intervals(intervals, tolerance=1e-9):
        if not intervals:
            return []
        sorted_intervals = sorted(([float(item[0]), float(item[1])] for item in intervals), key=lambda item: (item[0], item[1]))
        merged = [sorted_intervals[0]]
        for left, right in sorted_intervals[1:]:
            last_left, last_right = merged[-1]
            if left <= last_right + tolerance:
                merged[-1][1] = max(last_right, right)
            else:
                merged.append([left, right])
        return merged

    @staticmethod
    def _interval_runs_from_mask(grid_points, selected_mask, tolerance):
        selected_indices = np.flatnonzero(selected_mask)
        if selected_indices.size == 0:
            return []

        intervals = []
        run_start = int(selected_indices[0])
        prev_index = int(selected_indices[0])
        for current_index in selected_indices[1:]:
            current_index = int(current_index)
            prev_point = float(grid_points[prev_index])
            current_point = float(grid_points[current_index])
            step = current_point - prev_point
            if current_index != prev_index + 1 or step > tolerance:
                intervals.append([float(grid_points[run_start]), float(grid_points[prev_index])])
                run_start = current_index
            prev_index = current_index

        intervals.append([float(grid_points[run_start]), float(grid_points[prev_index])])
        return intervals

    def solve_weighted_bi_scale(self, scale_constraints, keep_ratio=None, min_segment_width=None):
        if keep_ratio is None:
            keep_ratio = self.bi_keep_ratio
        if min_segment_width is None:
            min_segment_width = self.bi_min_segment_width

        constraints = self._normalize_scale_constraints(scale_constraints)
        if not constraints:
            return self._empty_bi_result()

        total_weight = float(sum(item["weight"] for item in constraints))
        keep_threshold = float(keep_ratio * total_weight)
        atomic_segments = self._build_atomic_segments(constraints, min_segment_width=min_segment_width)
        if not atomic_segments:
            left = min(item["interval"][0] for item in constraints)
            right = max(item["interval"][1] for item in constraints)
            return {
                "solver": "bi",
                "consensus_scale": 0.5 * (left + right),
                "consensus_interval": [left, right],
                "bi_scale": 0.5 * (left + right),
                "bi_interval": [left, right],
                "bi_segments": [],
                "keep_threshold": keep_threshold,
                "total_weight": total_weight,
                "kept_segment_count": 0,
                "all_segment_count": 0,
            }

        weighted_segments = self._compute_segment_weights(constraints, atomic_segments)
        kept_segments = [
            segment for segment in weighted_segments
            if segment["coverage_weight"] >= keep_threshold
        ]
        if not kept_segments:
            max_coverage_weight = max(segment["coverage_weight"] for segment in weighted_segments)
            kept_segments = [
                segment for segment in weighted_segments
                if segment["coverage_weight"] == max_coverage_weight
            ]

        weighted_midpoint_sum = 0.0
        kept_weight_sum = 0.0
        for segment in kept_segments:
            weighted_midpoint_sum += segment["midpoint"] * segment["coverage_weight"]
            kept_weight_sum += segment["coverage_weight"]

        if kept_weight_sum > 0.0:
            bi_scale = weighted_midpoint_sum / kept_weight_sum
        else:
            bi_scale = 0.5 * (kept_segments[0]["interval"][0] + kept_segments[-1]["interval"][1])

        bi_interval = [
            min(segment["interval"][0] for segment in kept_segments),
            max(segment["interval"][1] for segment in kept_segments),
        ]
        return {
            "solver": "bi",
            "consensus_scale": float(bi_scale),
            "consensus_interval": bi_interval,
            "bi_scale": float(bi_scale),
            "bi_interval": bi_interval,
            "bi_segments": kept_segments,
            "keep_threshold": keep_threshold,
            "total_weight": total_weight,
            "kept_segment_count": len(kept_segments),
            "all_segment_count": len(weighted_segments),
        }

    def solve_ifpa_scale(self, scale_constraints, num_bins=None, use_weight=None):
        if num_bins is None:
            num_bins = self.ifpa_num_bins
        if use_weight is None:
            use_weight = self.ifpa_use_weight

        constraints = self._normalize_scale_constraints(scale_constraints)
        if not constraints:
            return self._empty_ifpa_result()

        global_min = min(item["interval"][0] for item in constraints)
        global_max = max(item["interval"][1] for item in constraints)
        if global_max <= global_min:
            return {
                "solver": "ifpa",
                "consensus_scale": float(global_min),
                "consensus_interval": [float(global_min), float(global_max)],
                "ifpa_scale": float(global_min),
                "ifpa_interval": [float(global_min), float(global_max)],
                "ifpa_intervals": [[float(global_min), float(global_max)]],
                "ifpa_best_score": 0.0,
                "ifpa_grid_size": int(num_bins),
                "ifpa_use_weight": bool(use_weight),
                "ifpa_top_point_count": 1,
                "ifpa_top_points": [float(global_min)],
                "ifpa_score_profile": [{"point": float(global_min), "score": 0.0, "coverage_weight": 0.0}],
            }

        grid_points = np.linspace(global_min, global_max, int(num_bins), dtype=np.float64)
        aggregated_scores = np.zeros_like(grid_points)
        aggregated_coverage = np.zeros_like(grid_points)

        for item in constraints:
            left, right = item["interval"]
            point_weight = float(item["weight"]) if use_weight else 1.0
            inside_mask = (grid_points >= left) & (grid_points <= right)
            inside_count = int(np.count_nonzero(inside_mask))
            outside_count = int(len(grid_points) - inside_count)

            if inside_count == 0 or outside_count == 0:
                continue

            aggregated_scores[inside_mask] += point_weight * outside_count
            aggregated_scores[~inside_mask] -= point_weight * inside_count
            aggregated_coverage[inside_mask] += point_weight

        best_score = float(np.max(aggregated_scores))
        selected_mask = np.isclose(aggregated_scores, best_score)
        top_points = grid_points[selected_mask]
        if top_points.size == 0:
            return self._empty_ifpa_result()

        if len(grid_points) > 1:
            grid_step = float(grid_points[1] - grid_points[0]) + self.bi_min_segment_width
        else:
            grid_step = self.bi_min_segment_width

        ifpa_intervals = self._interval_runs_from_mask(grid_points, selected_mask, tolerance=grid_step)
        selected_interval = max(
            ifpa_intervals,
            key=lambda item: (item[1] - item[0], -abs(0.5 * (item[0] + item[1]))),
        )
        ifpa_scale = float(np.median(top_points))

        score_profile = [
            {
                "point": float(point),
                "score": float(score),
                "coverage_weight": float(coverage_weight),
            }
            for point, score, coverage_weight in zip(grid_points, aggregated_scores, aggregated_coverage)
        ]

        return {
            "solver": "ifpa",
            "consensus_scale": ifpa_scale,
            "consensus_interval": [float(selected_interval[0]), float(selected_interval[1])],
            "ifpa_scale": ifpa_scale,
            "ifpa_interval": [float(selected_interval[0]), float(selected_interval[1])],
            "ifpa_intervals": ifpa_intervals,
            "ifpa_best_score": best_score,
            "ifpa_grid_size": int(num_bins),
            "ifpa_use_weight": bool(use_weight),
            "ifpa_top_point_count": int(top_points.size),
            "ifpa_top_points": [float(point) for point in top_points.tolist()],
            "ifpa_score_profile": score_profile,
        }

    def solve_consensus_scale(self, scale_constraints):
        if self.consensus_solver == "legacy":
            legacy_result = {
                "solver": "legacy",
                "consensus_scale": None,
                "consensus_interval": None,
            }
            self.last_bi_result = None
            self.last_ifpa_result = None
            self.last_consensus_result = legacy_result
            return legacy_result

        if self.consensus_solver == "ifpa":
            ifpa_result = self.solve_ifpa_scale(scale_constraints)
            self.last_ifpa_result = ifpa_result
            self.last_bi_result = None
            self.last_consensus_result = ifpa_result
            return ifpa_result

        bi_result = self.solve_weighted_bi_scale(scale_constraints)
        self.last_bi_result = bi_result
        self.last_ifpa_result = None
        self.last_consensus_result = bi_result
        return bi_result

    def _solve_consensus_by_name(self, scale_intervals, scale_constraints, solver_name, consistent_scale=None):
        if solver_name == "legacy":
            base_scale = consistent_scale
            if base_scale is None:
                if not scale_intervals:
                    return {
                        "solver": "legacy",
                        "consensus_scale": None,
                        "consensus_interval": None,
                    }
                base_scale, _ = self.search_consistent_scale(scale_intervals)
            return {
                "solver": "legacy",
                "consensus_scale": float(base_scale),
                "consensus_interval": None,
            }
        if solver_name == "bi":
            return self.solve_weighted_bi_scale(scale_constraints)
        if solver_name == "ifpa":
            return self.solve_ifpa_scale(scale_constraints)
        raise ValueError(f"Unsupported consensus solver: {solver_name}")

    def _resolve_default_optimized_variant_key(self):
        return self.default_optimized_variant

    def _compute_variant_results(self, scale_intervals, scale_constraints):
        if not scale_intervals:
            return {
                "consistent_scale": None,
                "consistent_count": 0,
                "consensus_results": {},
                "optimized_variants": {},
                "default_optimized_variant_key": self._resolve_default_optimized_variant_key(),
            }

        variant_definitions = (
            ("optimized_scale_legacy", "legacy", "legacy"),
            ("optimized_scale_original", "legacy", "original"),
            ("optimized_scale_weighted_bi", "bi", "weighted"),
            ("optimized_scale_weighted_ifpa", "ifpa", "weighted"),
        )
        active_variant_definitions = tuple(
            definition
            for definition in variant_definitions
            if definition[0] in self.enabled_optimized_variants
        )

        consistent_scale, consistent_count = self.search_consistent_scale(scale_intervals)
        required_consensus_solvers = {definition[1] for definition in active_variant_definitions}
        consensus_results = {}
        for solver_name in ("legacy", "bi", "ifpa"):
            if solver_name not in required_consensus_solvers:
                continue
            consensus_results[solver_name] = self._solve_consensus_by_name(
                scale_intervals,
                scale_constraints,
                solver_name,
                consistent_scale=consistent_scale,
            )

        optimized_variants = {}

        for variant_key, consensus_solver_name, optimization_solver_name in active_variant_definitions:
            consensus_result = consensus_results[consensus_solver_name]
            optimization_init = (
                consensus_result["consensus_scale"]
                if consensus_result["consensus_scale"] is not None
                else consistent_scale
            )
            optimized_scale = self.optimize_scale(
                scale_constraints,
                optimization_init,
                search_bounds=consensus_result.get("consensus_interval"),
                solver_name=optimization_solver_name,
                consensus_result=consensus_result,
            )
            optimized_variants[variant_key] = {
                "scale": optimized_scale,
                "consensus_solver": consensus_solver_name,
                "optimized_scale_solver": optimization_solver_name,
                "weighted_search_mode": self.weighted_search_mode if optimization_solver_name == "weighted" else "global",
                "consensus_prior_mode": self.consensus_prior_mode if optimization_solver_name == "weighted" else "none",
                "consensus_prior_strength": self.consensus_prior_strength if optimization_solver_name == "weighted" else 0.0,
                "consensus_scale": consensus_result.get("consensus_scale"),
                "consensus_interval": consensus_result.get("consensus_interval"),
                "optimization_info": dict(self.last_optimization_result or {}),
            }

        default_key = self._resolve_default_optimized_variant_key()
        return {
            "consistent_scale": consistent_scale,
            "consistent_count": consistent_count,
            "consensus_results": consensus_results,
            "optimized_variants": optimized_variants,
            "default_optimized_variant_key": default_key,
        }

    @staticmethod
    def _empty_legacy_consensus_result():
        return {
            "solver": "legacy",
            "consensus_scale": None,
            "consensus_interval": None,
        }

    def optimize_scale(
        self,
        scale_constraints,
        initial_scale,
        num_steps=100,
        search_bounds=None,
        solver_name=None,
        consensus_result=None,
    ):
        """
        Optimize the scale factor within the consistent interval using gradient descent.
        
        Args:
            scale_constraints (list): List of scale interval constraints for each object
            initial_scale (float): Initial scale factor from consistency search
            num_steps (int): Number of optimization steps
            search_bounds (list | tuple | None): Optional [min, max] bounds for optimization
            
        Returns:
            float: Optimized scale factor
        """
        effective_solver_name = solver_name or self.optimized_scale_solver

        if not scale_constraints:
            self.last_optimization_result = {
                "optimized_scale_solver": effective_solver_name,
                "weighted_search_mode": self.weighted_search_mode if effective_solver_name == "weighted" else "global",
                "consensus_prior_mode": self.consensus_prior_mode if effective_solver_name == "weighted" else "none",
                "consensus_prior_strength": self.consensus_prior_strength if effective_solver_name == "weighted" else 0.0,
                "consensus_prior_interval_inside_ratio": self.consensus_prior_interval_inside_ratio if effective_solver_name == "weighted" else 0.0,
                "used_consensus_prior": False,
                "consensus_prior_scale": None,
                "consensus_prior_interval": None,
                "search_bounds": search_bounds,
                "used_search_bounds": False,
                "constraint_count": 0,
            }
            return initial_scale

        def parse_weighted_constraint(item):
            if isinstance(item, dict):
                return item["interval"], float(item.get("weight", 1.0))
            return item, 1.0

        def parse_legacy_constraint(item):
            if isinstance(item, dict):
                return item["interval"], 1.0
            return item, 1.0

        if effective_solver_name == "legacy":
            parse_constraint = parse_legacy_constraint
            effective_search_bounds = None
        elif effective_solver_name == "original":
            parse_constraint = parse_weighted_constraint
            effective_search_bounds = None
        else:
            parse_constraint = parse_weighted_constraint
            effective_search_bounds = search_bounds if self.weighted_search_mode == "consensus_bounded" else None

        prior_mode = self.consensus_prior_mode if effective_solver_name == "weighted" else "none"
        prior_strength = self.consensus_prior_strength if effective_solver_name == "weighted" else 0.0
        prior_inside_ratio = (
            self.consensus_prior_interval_inside_ratio
            if effective_solver_name == "weighted"
            else 0.0
        )
        prior_scale = None if consensus_result is None else consensus_result.get("consensus_scale")
        prior_interval = None if consensus_result is None else consensus_result.get("consensus_interval")

        if prior_mode == "point" and prior_scale is None:
            prior_mode = "none"
        if prior_mode == "interval":
            if prior_interval is None:
                if prior_scale is not None:
                    prior_mode = "point"
                else:
                    prior_mode = "none"

        def interval_loss(s):
            """Calculate loss for a given scale factor s."""
            loss = 0.0
            for item in scale_constraints:
                interval, weight = parse_constraint(item)
                a, b = interval
                if s < a:
                    loss += weight * ((s - a) ** 2)
                elif s > b:
                    loss += weight * ((s - b) ** 2)
            if prior_strength > 0.0 and prior_mode != "none":
                if prior_mode == "point":
                    loss += prior_strength * ((s - float(prior_scale)) ** 2)
                else:
                    prior_left = float(prior_interval[0])
                    prior_right = float(prior_interval[1])
                    prior_center = 0.5 * (prior_left + prior_right)
                    prior_half_width = max(0.5 * (prior_right - prior_left), 1e-6)
                    if s < prior_left:
                        normalized_distance = (prior_left - s) / prior_half_width
                        loss += prior_strength * (normalized_distance ** 2)
                    elif s > prior_right:
                        normalized_distance = (s - prior_right) / prior_half_width
                        loss += prior_strength * (normalized_distance ** 2)
                    else:
                        normalized_center_offset = (s - prior_center) / prior_half_width
                        loss += prior_strength * prior_inside_ratio * (normalized_center_offset ** 2)
            return loss

        # Use golden section search for optimization
        def golden_section_search(f, a, b, tol=1e-5):
            gr = (np.sqrt(5) + 1) / 2
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            
            while abs(b - a) > tol:
                if f(c) < f(d):
                    b = d
                else:
                    a = c
                c = b - (b - a) / gr
                d = a + (b - a) / gr
            return (a + b) / 2
        
        # Determine the optimization range
        parsed_constraints = [parse_constraint(item) for item in scale_constraints]
        if effective_search_bounds is not None:
            global_min = float(effective_search_bounds[0])
            global_max = float(effective_search_bounds[1])
        else:
            global_min = min(interval[0] for interval, _ in parsed_constraints)
            global_max = max(interval[1] for interval, _ in parsed_constraints)
        # Perform optimization
        optimized_scale = golden_section_search(interval_loss, global_min, global_max)
        self.last_optimization_result = {
            "optimized_scale_solver": effective_solver_name,
            "weighted_search_mode": self.weighted_search_mode if effective_solver_name == "weighted" else "global",
            "consensus_prior_mode": prior_mode,
            "consensus_prior_strength": prior_strength,
            "consensus_prior_interval_inside_ratio": prior_inside_ratio,
            "used_consensus_prior": prior_strength > 0.0 and prior_mode != "none",
            "consensus_prior_scale": None if prior_scale is None else float(prior_scale),
            "consensus_prior_interval": None if prior_interval is None else [float(prior_interval[0]), float(prior_interval[1])],
            "search_bounds": None if effective_search_bounds is None else [float(effective_search_bounds[0]), float(effective_search_bounds[1])],
            "used_search_bounds": effective_search_bounds is not None,
            "constraint_count": len(parsed_constraints),
        }
        return optimized_scale

    def depth_to_camera_points(self, depth_map, intrinsics):
        """
        将深度图转换为相机坐标系下的点云
        Args:
            depth_map: 深度图，形状为 (H, W)
            intrinsics: 相机内参矩阵，形状为 (3, 3)
        
        Returns:
            np.ndarray: 相机坐标系下的点云，形状为 (N, 3)，N为有效点数量
        """
        # 获取深度图的高度和宽度
        h, w = depth_map.shape
        
        # 创建像素坐标网格
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # 展平坐标和深度值
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        depth_values = depth_map.flatten()
        
        # 过滤无效深度值（0或负值）
        valid_mask = depth_values > 1e-5
        u_valid = u_coords[valid_mask]
        v_valid = v_coords[valid_mask]
        depth_valid = depth_values[valid_mask]
        
        # 计算内参矩阵的逆
        inv_intrinsics = np.linalg.inv(intrinsics)
        
        # 使用向量化操作计算相机坐标系下的点云
        # 创建归一化图像坐标矩阵 [N, 3]
        normalized_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=1)
        
        # 应用内参的逆变换获取相机坐标系下的射线方向 [N, 3]
        ray_dirs = normalized_coords @ inv_intrinsics.T
        
        # 乘以深度值得到相机坐标系下的点 [N, 3]
        points_camera = ray_dirs * depth_valid[:, np.newaxis]
        
        return points_camera

    def camera_to_lidar_points(self, camera_points, extrinsic):
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

    def _get_image_size(self, image_path):
        if not image_path:
            return None
        if image_path in self.image_size_cache:
            return self.image_size_cache[image_path]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self.image_size_cache[image_path] = None
            return None
        image_size = image.shape[:2]
        self.image_size_cache[image_path] = image_size
        return image_size

    def _scale_bbox_to_mask_shape(self, bbox, image_path, target_shape):
        target_h, target_w = target_shape
        image_size = self._get_image_size(image_path)
        source_h, source_w = image_size if image_size is not None else target_shape

        x1, y1, x2, y2 = map(float, bbox)
        scale_x = float(target_w) / max(float(source_w), 1.0)
        scale_y = float(target_h) / max(float(source_h), 1.0)

        x1 = int(np.floor(x1 * scale_x))
        y1 = int(np.floor(y1 * scale_y))
        x2 = int(np.ceil(x2 * scale_x))
        y2 = int(np.ceil(y2 * scale_y))

        x1 = max(0, min(target_w, x1))
        y1 = max(0, min(target_h, y1))
        x2 = max(0, min(target_w, x2))
        y2 = max(0, min(target_h, y2))
        return x1, y1, x2, y2
    
    def extract_semantic_anchor_mask(self, depth, anchor, image_path, sequence_name=None):
        # Create a mask for all plane regions
        H, W = depth.shape
        anchor_mask = np.zeros((H, W), dtype=bool)
        object_id = anchor.get("object_id")
        object_class = anchor.get("object_class")
        # Construct mask file path
        image_name = os.path.basename(image_path).split('.')[0]
        output_subdir = os.path.join(
            get_sam_output_dir(sequence_name, dataset_name=config.dataset_name),
            image_name,
        )
        mask_filename = f"{object_class}_mask_{object_id}.png"
        mask_path = os.path.join(output_subdir, mask_filename)
        
        # Check cache first
        cache_key = (mask_path, H, W)
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = mask > 128
                self.mask_cache[cache_key] = mask
                return mask

        bbox = anchor.get("object_bbox_2d")
        if bbox:
            x1, y1, x2, y2 = self._scale_bbox_to_mask_shape(
                bbox,
                image_path,
                (H, W),
            )
            anchor_mask[y1:y2, x1:x2] = True

        self.mask_cache[cache_key] = anchor_mask
        return anchor_mask
    
    def extract_semantic_plane_mask(self, depth, semantic_planes, image_path, sequence_name=None):
        """
        Extract mask for semantic plane regions.
        
        Args:
            depth (np.ndarray): [H, W] depth map
            semantic_planes (list): List of semantic plane dictionaries
            image_path (str): Path to the original image
            sequence_name (str): Name of the sequence being processed
            
        Returns:
            np.ndarray: [H, W] boolean mask indicating plane regions
        """
        
        # Create a mask for all plane regions
        H, W = depth.shape
        plane_mask = np.zeros((H, W), dtype=bool)
        
        # Process each semantic plane
        for plane in semantic_planes:
            object_id = plane.get("object_id")
            plane_type = plane.get("plane_type")
            
            # Construct mask file path
            image_name = os.path.basename(image_path).split('.')[0]
            output_subdir = os.path.join(
                get_sam_output_dir(sequence_name, dataset_name=config.dataset_name),
                image_name,
            )
            mask_filename = f"{plane_type}_mask_{object_id}.png"
            mask_path = os.path.join(output_subdir, mask_filename)
            
            # Check cache first
            cache_key = (mask_path, H, W)
            if cache_key in self.mask_cache:
                mask = self.mask_cache[cache_key]
                plane_mask |= mask
                continue
            
            # Check if mask file exists
            if os.path.exists(mask_path):
                # Load mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize mask to match depth map shape if needed
                    if mask.shape != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    # Convert to binary mask
                    mask = mask > 128
                    # Cache the result
                    self.mask_cache[cache_key] = mask
                    # Combine with plane_mask
                    plane_mask |= mask
            else:
                # If mask file doesn't exist, use bounding box as fallback
                bbox = plane.get("plane_bbox_2d")
                if bbox:
                    x1, y1, x2, y2 = self._scale_bbox_to_mask_shape(
                        bbox,
                        image_path,
                        (H, W),
                    )
                    # Create bounding box mask
                    plane_mask[y1:y2, x1:x2] = True
        
        return plane_mask

    def fit_plane_ransac(self, points, iterations=500, threshold=0.01, confidence=0.99):
        """
        Fit plane to points using RANSAC algorithm with optimized parameters and caching.
        
        Args:
            points (np.ndarray): [N, 3] points to fit plane to
            iterations (int): Maximum number of RANSAC iterations
            threshold (float): Distance threshold for inliers
            confidence (float): Desired confidence in the result
            
        Returns:
            tuple: (normal, distance) where normal is [3,] plane normal vector,
                   distance is the distance from origin to plane
        """
        cache_key = self._make_plane_fit_cache_key(points, iterations, threshold, confidence)
        
        # Check cache first
        if cache_key in self.plane_fit_cache:
            return self.plane_fit_cache[cache_key]
        
        best_normal = None
        best_distance = 0
        best_inlier_count = 0
        
        n_points = len(points)
        if n_points < 3:
            return None, 0
        
        # Calculate adaptive iterations based on confidence
        # k = log(1 - confidence) / log(1 - (1 - outlier_ratio)^3)
        # Assuming worst case 50% outliers
        outlier_ratio = 0.5
        adaptive_iterations = int(np.log(1 - confidence) / np.log(1 - (1 - outlier_ratio)**3))
        adaptive_iterations = min(adaptive_iterations, iterations)
        
        rng_seed = int(hashlib.blake2b(str(cache_key).encode("utf-8"), digest_size=8).hexdigest(), 16)
        rng = np.random.default_rng(rng_seed)

        for i in range(adaptive_iterations):
            # Use a deterministic RNG so repeated runs with the same inputs fit the same plane.
            sample_indices = rng.choice(n_points, 3, replace=False)
            sample_points = points[sample_indices]
            
            # Calculate plane normal using cross product
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            # Normalize the normal vector
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue  # Degenerate case, skip
            normal = normal / norm
            
            # Calculate distance from origin to plane
            distance = np.dot(normal, sample_points[0])
            
            # Calculate distances from all points to the plane using vectorized operation
            # Plane equation: ax + by + cz - d = 0
            # Distance = |ax + by + cz - d| / sqrt(a² + b² + c²)
            # Since normal is normalized, denominator is 1
            distances = np.abs(np.dot(points, normal) - distance)
            
            # Count inliers
            inlier_count = np.sum(distances < threshold)
            
            # Update best plane if current has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_normal = normal
                best_distance = distance
                
                # Early termination if we found a plane with enough inliers
                if best_inlier_count > n_points * 0.9:
                    break
        
        # Cache the result
        self.plane_fit_cache[cache_key] = (best_normal, best_distance)
        
        return best_normal, best_distance

    def align_point_cloud(self, point_cloud, plane_normal):
        """
        Rotate point cloud to align plane normal with Z-axis with caching.
        
        Args:
            point_cloud (np.ndarray): [H, W, 3] or [N, 3] point cloud
            plane_normal (np.ndarray): [3,] plane normal vector
            
        Returns:
            np.ndarray: Aligned point cloud
        """
        original_shape = point_cloud.shape
        if len(original_shape) == 3:
            point_cloud_reshaped = point_cloud.reshape(-1, 3)
        else:
            point_cloud_reshaped = point_cloud
        
        cache_key = self._make_alignment_cache_key(point_cloud_reshaped, plane_normal)
        
        # Check cache first
        if cache_key in self.alignment_cache:
            aligned_points = self.alignment_cache[cache_key]
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                aligned_points = aligned_points.reshape(original_shape)
            return aligned_points
        
        # Target normal is Z-axis
        target_normal = np.array([0, 0, 1])
        
        # Calculate rotation to align plane_normal with target_normal
        # Get rotation axis as cross product
        axis = np.cross(plane_normal, target_normal)
        axis_norm = np.linalg.norm(axis)
        
        aligned_points = point_cloud_reshaped.copy()
        
        if axis_norm > 1e-6:  # If not already aligned
            axis = axis / axis_norm
            # Calculate rotation angle
            angle = np.arccos(np.clip(np.dot(plane_normal, target_normal), -1.0, 1.0))
            
            # Create rotation object
            rot = Rotation.from_rotvec(angle * axis)
            
            # Apply rotation
            aligned_points = rot.apply(point_cloud_reshaped)
        
        # Cache the result (stored as 2D array for consistency)
        self.alignment_cache[cache_key] = aligned_points
        
        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            aligned_points = aligned_points.reshape(original_shape)
        
        return aligned_points

    def measure_object_heights(self, aligned_point_cloud, object_masks):
        """
        Measure relative heights of objects in aligned coordinate system.
        
        Args:
            aligned_point_cloud (np.ndarray): [H, W, 3] aligned point cloud
            object_masks (list): List of [H, W] binary masks for each object
            
        Returns:
            list: List of relative heights for each object
        """
        heights = []
        
        for mask in object_masks:
            # Extract object points
            object_points = aligned_point_cloud[mask]
            
            if len(object_points) == 0:
                heights.append(0.0)
                continue
            
            # Calculate height as z_max - z_min
            z_values = object_points[:, 2]
            height = np.max(z_values) - np.min(z_values)
            heights.append(height)
        
        return heights

    @staticmethod
    def _estimate_anchor_top_z(aligned_anchor_points):
        if aligned_anchor_points is None or len(aligned_anchor_points) == 0:
            return None
        return float(np.max(aligned_anchor_points[:, 2]))

    @staticmethod
    def _estimate_anchor_bottom_z(aligned_anchor_points):
        if aligned_anchor_points is None or len(aligned_anchor_points) == 0:
            return None
        return float(np.min(aligned_anchor_points[:, 2]))

    @staticmethod
    def _estimate_ground_z(aligned_plane_points):
        if aligned_plane_points is None or len(aligned_plane_points) == 0:
            return None
        return float(np.mean(aligned_plane_points[:, 2]))

    @staticmethod
    def _normalize_conf_map(conf_map):
        if conf_map is None:
            return None
        conf_array = np.asarray(conf_map)
        if conf_array.ndim == 3 and conf_array.shape[-1] == 1:
            return conf_array[..., 0]
        return conf_array

    def _apply_conf_percentile_mask(self, base_mask, conf_map, percentile, min_points=32):
        if base_mask is None:
            return base_mask
        if not self.use_depth_confidence or conf_map is None:
            return base_mask

        conf_map = self._normalize_conf_map(conf_map)

        valid_conf_mask = base_mask & np.isfinite(conf_map)
        valid_conf_values = conf_map[valid_conf_mask]
        if valid_conf_values.size == 0:
            return base_mask
        if valid_conf_values.size < min_points:
            return valid_conf_mask

        threshold = float(np.quantile(valid_conf_values, percentile))
        filtered_mask = valid_conf_mask & (conf_map >= threshold)
        if int(np.count_nonzero(filtered_mask)) < max(8, min_points // 4):
            return valid_conf_mask
        return filtered_mask

    def _extract_valid_points(self, world_pointmap, base_mask, conf_map=None, conf_percentile=None, min_points=32):
        filtered_mask = base_mask
        if conf_percentile is not None:
            filtered_mask = self._apply_conf_percentile_mask(
                filtered_mask,
                conf_map,
                conf_percentile,
                min_points=min_points,
            )
        points = world_pointmap[filtered_mask]
        points = points[np.isfinite(points).all(axis=1)]
        return points, filtered_mask

    def _compute_frame_conf_thresholds(self, conf_map):
        if (
            not self.use_anchor_quality_filter
            or not self.use_depth_confidence
            or conf_map is None
        ):
            return None

        conf_map = self._normalize_conf_map(conf_map)

        valid_conf = conf_map[np.isfinite(conf_map)]
        if valid_conf.size == 0:
            return None
        return {
            "min_threshold": float(np.quantile(valid_conf, self.anchor_quality_min_percentile)),
            "max_threshold": float(np.quantile(valid_conf, self.anchor_quality_max_percentile)),
        }

    def _evaluate_anchor_quality(self, conf_map, anchor_mask, frame_conf_thresholds):
        quality_info = {
            "passed_quality_filter": True,
            "anchor_conf_min": None,
            "anchor_conf_max": None,
            "frame_conf_min_threshold": None,
            "frame_conf_max_threshold": None,
            "conf_value_count": 0,
        }

        if not self.use_anchor_quality_filter:
            return quality_info
        if conf_map is None or frame_conf_thresholds is None or anchor_mask is None:
            return quality_info

        conf_map = self._normalize_conf_map(conf_map)

        frame_conf_min_threshold = frame_conf_thresholds.get("min_threshold")
        frame_conf_max_threshold = frame_conf_thresholds.get("max_threshold")
        quality_info["frame_conf_min_threshold"] = frame_conf_min_threshold
        quality_info["frame_conf_max_threshold"] = frame_conf_max_threshold

        valid_anchor_conf = conf_map[anchor_mask & np.isfinite(conf_map)]
        quality_info["conf_value_count"] = int(valid_anchor_conf.size)
        if valid_anchor_conf.size == 0:
            quality_info["passed_quality_filter"] = False
            return quality_info

        anchor_conf_min = float(np.min(valid_anchor_conf))
        anchor_conf_max = float(np.max(valid_anchor_conf))
        quality_info["anchor_conf_min"] = anchor_conf_min
        quality_info["anchor_conf_max"] = anchor_conf_max
        quality_info["passed_quality_filter"] = (
            anchor_conf_min >= frame_conf_min_threshold
            and anchor_conf_max >= frame_conf_max_threshold
        )
        return quality_info

    def _resolve_active_plane_normal(self, frame_plane_normal, global_plane_normal):
        if self.shared_plane_mode == "per_frame_plane":
            if frame_plane_normal is not None:
                return frame_plane_normal, "frame"
            return global_plane_normal, "global_fallback"
        return global_plane_normal, "global"

    def _prepare_dynamic_anchor(self, anchor):
        prepared = {**anchor, "anchor_group": anchor.get("anchor_group", "dynamic")}
        interval = prepared.get("height_interval")
        if self.dynamic_anchor_height_scale != 1.0 and isinstance(interval, (list, tuple)) and len(interval) == 2:
            try:
                prepared["height_interval"] = [
                    float(value) * self.dynamic_anchor_height_scale for value in interval
                ]
            except (TypeError, ValueError):
                pass
        return prepared

    def _select_height_anchors(self, vlm_results):
        dynamic_height_anchors = vlm_results.get("dynamic_height_anchors", [])
        static_height_anchors = vlm_results.get("static_height_anchors", [])

        if self.height_anchor_source == "dynamic":
            return self._filter_height_anchors_for_optimizer([
                self._prepare_dynamic_anchor(anchor)
                for anchor in dynamic_height_anchors
            ])
        if self.height_anchor_source == "static":
            return self._filter_height_anchors_for_optimizer([
                {**anchor, "anchor_group": anchor.get("anchor_group", "static")}
                for anchor in static_height_anchors
            ])
        height_anchors = (
            [self._prepare_dynamic_anchor(anchor) for anchor in dynamic_height_anchors]
            + [{**anchor, "anchor_group": anchor.get("anchor_group", "static")} for anchor in static_height_anchors]
        )
        return self._filter_height_anchors_for_optimizer(height_anchors)

    def _is_anchor_class_enabled(self, anchor):
        anchor_group = str(anchor.get("anchor_group", "unknown") or "").strip().lower()
        object_class = str(anchor.get("object_class", "") or "").strip().lower()
        if not object_class:
            return False
        if anchor_group == "dynamic" and self.optimizer_dynamic_classes:
            return object_class in self.optimizer_dynamic_classes
        if anchor_group == "static" and self.optimizer_static_classes:
            return object_class in self.optimizer_static_classes
        return True

    def _filter_height_anchors_for_optimizer(self, height_anchors):
        if not self.optimizer_dynamic_classes and not self.optimizer_static_classes:
            return height_anchors
        return [anchor for anchor in height_anchors if self._is_anchor_class_enabled(anchor)]

    def _evaluate_vlm_anchor_confidence(self, anchor):
        quality_info = {
            "passed_vlm_confidence_filter": True,
            "vlm_anchor_confidence": None,
            "vlm_anchor_confidence_threshold": None,
        }

        if not self.use_vlm_anchor_confidence_filter:
            return quality_info

        anchor_group = str(anchor.get("anchor_group", "unknown") or "").strip().lower()
        threshold = self.vlm_anchor_conf_thresholds.get(anchor_group)
        confidence = self._parse_vlm_weight(anchor.get("confidence"))

        quality_info["vlm_anchor_confidence"] = confidence
        quality_info["vlm_anchor_confidence_threshold"] = threshold

        if threshold is None or confidence is None:
            return quality_info

        quality_info["passed_vlm_confidence_filter"] = confidence >= threshold
        return quality_info

    def _extract_local_ground_points(self, world_pointmap, plane_mask, anchor_mask, conf_map=None, conf_percentile=None):
        if (
            world_pointmap is None
            or plane_mask is None
            or anchor_mask is None
            or world_pointmap.size == 0
            or not np.any(plane_mask)
            or not np.any(anchor_mask)
        ):
            return np.empty((0, 3), dtype=np.float32)

        anchor_y, anchor_x = np.where(anchor_mask)
        if anchor_x.size == 0 or anchor_y.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        y1, y2 = int(anchor_y.min()), int(anchor_y.max())
        x1, x2 = int(anchor_x.min()), int(anchor_x.max())
        box_h = max(1, y2 - y1 + 1)
        box_w = max(1, x2 - x1 + 1)

        margin_x = max(10, int(round(box_w * 0.5)))
        margin_top = max(5, int(round(box_h * 0.15)))
        margin_bottom = max(10, int(round(box_h * 0.75)))

        H, W = plane_mask.shape
        search_x1 = max(0, x1 - margin_x)
        search_x2 = min(W, x2 + margin_x + 1)
        search_y1 = max(0, y1 - margin_top)
        search_y2 = min(H, y2 + margin_bottom + 1)

        local_region = np.zeros_like(plane_mask, dtype=bool)
        local_region[search_y1:search_y2, search_x1:search_x2] = True
        local_ground_mask = plane_mask & local_region & (~anchor_mask)

        if conf_percentile is not None:
            local_ground_mask = self._apply_conf_percentile_mask(
                local_ground_mask,
                conf_map,
                conf_percentile,
                min_points=16,
            )

        local_ground_points = world_pointmap[local_ground_mask]
        local_ground_points = local_ground_points[np.isfinite(local_ground_points).all(axis=1)]
        return local_ground_points

    def _prepare_frame_context(self, pointmap, conf, extrinsics, vlm_results, sequence_name=None):
        image_path = vlm_results.get("image_path")
        semantic_planes = vlm_results.get("semantic_planes", [])
        height_anchors = self._select_height_anchors(vlm_results)

        R = extrinsics[:, :3]
        t = extrinsics[:, 3:]
        R_world = R.T
        t_world = -R_world @ t

        H, W, _ = pointmap.shape
        pointmap_reshaped = pointmap.reshape(-1, 3)
        world_points = (R_world @ pointmap_reshaped.T + t_world).T
        world_pointmap = world_points.reshape(H, W, 3)

        dummy_depth = np.zeros_like(pointmap[..., 0])
        plane_mask = self.extract_semantic_plane_mask(
            dummy_depth, semantic_planes, image_path, sequence_name
        )

        valid_plane_points, filtered_plane_mask = self._extract_valid_points(
            world_pointmap,
            plane_mask,
            conf_map=conf,
            conf_percentile=self.plane_conf_percentile,
            min_points=64,
        )

        frame_plane_normal = None
        if len(valid_plane_points) >= 3:
            if self.shared_plane_mode == "per_frame_plane":
                frame_plane_normal, _ = self.fit_plane_ransac(valid_plane_points)
            elif self.plane_normal_outlier_filter_enabled:
                frame_plane_normal = self._fit_plane_normal_for_filter(valid_plane_points)
            else:
                frame_plane_normal, _ = self.fit_plane_ransac(valid_plane_points)

        return {
            "world_pointmap": world_pointmap,
            "conf": conf,
            "frame_conf_thresholds": self._compute_frame_conf_thresholds(conf),
            "plane_mask": filtered_plane_mask,
            "valid_plane_points": valid_plane_points,
            "frame_plane_normal": frame_plane_normal,
            "height_anchors": height_anchors,
            "image_path": image_path,
            "sequence_name": sequence_name,
        }

    def _resolve_relative_height_measurement(self, aligned_anchor_points, aligned_plane_points=None, ground_source=None):
        top_z = self._estimate_anchor_top_z(aligned_anchor_points)
        if top_z is None or not np.isfinite(top_z):
            return None

        bottom_z = self._estimate_anchor_bottom_z(aligned_anchor_points)
        if bottom_z is None or not np.isfinite(bottom_z):
            return None

        used_mode = self.relative_height_mode
        ground_z = None
        resolved_ground_source = "anchor_bottom"

        if self.relative_height_mode == "peak_to_bottom":
            base_z = bottom_z
            relative_height = abs(float(top_z - base_z))
        elif self.relative_height_mode == "peak_to_ground":
            ground_z = self._estimate_ground_z(aligned_plane_points)
            if ground_z is None or not np.isfinite(ground_z) or ground_z > top_z:
                base_z = bottom_z
                used_mode = "peak_to_bottom"
                resolved_ground_source = "bottom_fallback"
                relative_height = abs(float(top_z - base_z))
            else:
                base_z = ground_z
                resolved_ground_source = ground_source or "global"
                relative_height = float(
                    max(
                        abs(bottom_z - ground_z),
                        abs(top_z - ground_z),
                    )
                )
        else:
            raise ValueError(f"Unsupported relative height mode: {self.relative_height_mode}")

        if not np.isfinite(relative_height):
            return None
        relative_height = float(relative_height)
        return {
            "configured_mode": self.relative_height_mode,
            "used_mode": used_mode,
            "top_z": float(top_z),
            "bottom_z": float(bottom_z),
            "ground_z": None if ground_z is None or not np.isfinite(ground_z) else float(ground_z),
            "ground_source": resolved_ground_source,
            "base_z": float(base_z),
            "relative_height": relative_height,
            "anchor_point_count": int(len(aligned_anchor_points)) if aligned_anchor_points is not None else 0,
            "plane_point_count": int(len(aligned_plane_points)) if aligned_plane_points is not None else 0,
        }

    def _estimate_relative_height(self, aligned_anchor_points, aligned_plane_points=None):
        measurement = self._resolve_relative_height_measurement(aligned_anchor_points, aligned_plane_points)
        if measurement is None:
            return None
        return measurement["relative_height"]

    def estimate_metric_scale_batch(self, pointmap_list, conf_list, extrinsics_list, vlm_results_list, sequence_name_list=None):
        """
        Batch version of estimate_metric_scale that uses multiple frames to estimate metric scale.
        
        Args:
            pointmap_list (list): List of [H, W, 3] point maps containing 3D coordinates for each frame
            conf_list (list): List of [H, W] confidence maps for each frame
            extrinsics_list (list): List of [3, 4] camera extrinsics matrices (w2c) for each frame
            vlm_results_list (list): List of VLM semantic scale anchor results for each frame
            sequence_name_list (list): List of sequence names for each frame (optional)
            
        Returns:
            tuple: (consistent_scale, consistent_count, optimized_scale, scale_intervals)
        """
        if not pointmap_list:
            return None, 0, None, []

        self.last_bi_result = None
        self.last_ifpa_result = None
        self.last_consensus_result = None

        # Initialize lists to collect data from all frames
        all_plane_points = []
        all_scale_intervals = []
        all_scale_constraints = []
        all_anchor_measurements = []
        frame_data = []

        frame_inputs = [
            (
                pointmap,
                conf,
                extrinsics,
                vlm_results,
                sequence_name_list[i] if sequence_name_list else None,
            )
            for i, (pointmap, conf, extrinsics, vlm_results) in enumerate(
                zip(pointmap_list, conf_list, extrinsics_list, vlm_results_list)
            )
        ]

        workers = self.frame_parallel_workers
        if workers > 1 and len(frame_inputs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                frame_data = list(
                    executor.map(
                        lambda args: self._prepare_frame_context(*args),
                        frame_inputs,
                    )
                )
        else:
            frame_data = [
                self._prepare_frame_context(*frame_input)
                for frame_input in frame_inputs
            ]

        global_plane_frame_data = self._filter_frame_data_for_global_plane(frame_data)
        for data in global_plane_frame_data:
            valid_plane_points = data["valid_plane_points"]
            if len(valid_plane_points) > 0:
                all_plane_points.append(valid_plane_points)
        
        # Check if we have enough plane points for fitting
        if not all_plane_points:
            return None, 0, None, [], []
        
        # Step 2: Combine all plane points from all frames
        combined_plane_points = np.vstack(all_plane_points)
        
        # Step 3: Global plane fitting using all frames' plane points
        global_plane_normal, _ = self.fit_plane_ransac(combined_plane_points)
        
        if global_plane_normal is None:
            print("No valid plane found from all frames, skipping")
            return None, 0, None, [], []
        
        # Step 4: Process all anchors from all frames
        for data in frame_data:
            world_pointmap = data["world_pointmap"]
            conf = data["conf"]
            frame_conf_thresholds = data["frame_conf_thresholds"]
            plane_mask = data["plane_mask"]
            valid_plane_points = data["valid_plane_points"]
            frame_plane_normal = data["frame_plane_normal"]
            height_anchors = data["height_anchors"]
            image_path = data["image_path"]
            sequence_name = data["sequence_name"]
            
            dummy_depth = np.zeros_like(world_pointmap[..., 0])
            active_plane_normal, plane_reference_mode = self._resolve_active_plane_normal(
                frame_plane_normal,
                global_plane_normal,
            )
            aligned_plane_points = (
                self.align_point_cloud(valid_plane_points, active_plane_normal)
                if len(valid_plane_points) > 0
                else np.empty((0, 3))
            )
            aligned_anchor_data = []
            for anchor in height_anchors:
                weight_info = self._extract_anchor_weight_info(anchor)
                vlm_conf_info = self._evaluate_vlm_anchor_confidence(anchor)
                if not vlm_conf_info["passed_vlm_confidence_filter"]:
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": anchor["height_interval"],
                            "scale_interval": None,
                            **weight_info,
                            **vlm_conf_info,
                        }
                    )
                    continue
                # Extract semantic anchor mask
                anchor_mask = self.extract_semantic_anchor_mask(
                    dummy_depth, anchor, image_path, sequence_name
                )
                quality_info = self._evaluate_anchor_quality(conf, anchor_mask, frame_conf_thresholds)
                if not quality_info["passed_quality_filter"]:
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": anchor["height_interval"],
                            "scale_interval": None,
                            "depth_geometry_filter_stage": "anchor_quality_filter",
                            "anchor_mask_pixel_count": int(np.count_nonzero(anchor_mask)),
                            **weight_info,
                            **vlm_conf_info,
                            **quality_info,
                        }
                    )
                    continue
                
                # Extract anchor point cloud from world pointmap
                valid_anchor_points, filtered_anchor_mask = self._extract_valid_points(
                    world_pointmap,
                    anchor_mask,
                    conf_map=conf,
                    min_points=self._depth_geometry_min_points(),
                )
                
                if len(valid_anchor_points) == 0:
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": anchor["height_interval"],
                            "scale_interval": None,
                            "depth_geometry_filter_stage": "missing_valid_anchor_points",
                            "anchor_mask_pixel_count": int(np.count_nonzero(anchor_mask)),
                            "filtered_anchor_mask_pixel_count": int(np.count_nonzero(filtered_anchor_mask)),
                            "valid_anchor_point_count": 0,
                            **weight_info,
                            **vlm_conf_info,
                            **quality_info,
                        }
                    )
                    continue
                
                # Align anchor point cloud using the frame-consistent active plane normal.
                aligned_anchor_points = self.align_point_cloud(valid_anchor_points, active_plane_normal)
                local_ground_points = self._extract_local_ground_points(
                    world_pointmap,
                    plane_mask,
                    filtered_anchor_mask,
                    conf_map=conf,
                    conf_percentile=self.plane_conf_percentile,
                )
                aligned_local_ground_points = (
                    self.align_point_cloud(local_ground_points, active_plane_normal)
                    if len(local_ground_points) > 0
                    else np.empty((0, 3))
                )
                aligned_anchor_data.append(
                    {
                        "label": f"{anchor['object_class']}_{anchor['object_id']}",
                        "points": aligned_anchor_points,
                    }
                )
                
                measurement = self._resolve_relative_height_measurement(
                    aligned_anchor_points,
                    aligned_local_ground_points if len(aligned_local_ground_points) > 0 else aligned_plane_points,
                    ground_source="local" if len(aligned_local_ground_points) > 0 else "global",
                )
                ref_height = None if measurement is None else measurement["relative_height"]
                height_interval = anchor["height_interval"]
                real_min, real_max = height_interval[0], height_interval[1]
                if self._has_valid_relative_height(ref_height):
                    scale_interval = [real_min / ref_height, real_max / ref_height]
                    all_scale_intervals.append(scale_interval)
                    all_scale_constraints.append(
                        {
                            "interval": scale_interval,
                            "weight": weight_info["combined_weight"],
                            "type_weight": weight_info["type_weight"],
                            "quality_weight": weight_info["quality_weight"],
                            "priority": weight_info["priority"],
                            "distance_level": weight_info["distance_level"],
                            "distance_factor": weight_info["distance_factor"],
                            "priority_factor": weight_info["priority_factor"],
                            "modulated_type_weight": weight_info["modulated_type_weight"],
                            "combined_weight": weight_info["combined_weight"],
                            "weight_formula": weight_info["weight_formula"],
                            "object_class": anchor.get("object_class"),
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_id": anchor.get("object_id"),
                        }
                    )
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": height_interval,
                            "scale_interval": scale_interval,
                            **weight_info,
                            "shared_plane_mode": self.shared_plane_mode,
                            "plane_reference_mode": plane_reference_mode,
                            "frame_plane_available": frame_plane_normal is not None,
                            "global_plane_available": global_plane_normal is not None,
                            **vlm_conf_info,
                            **quality_info,
                            **measurement,
                        }
                    )
                elif measurement is not None:
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": height_interval,
                            "scale_interval": None,
                            "depth_geometry_filter_stage": "invalid_relative_height",
                            **weight_info,
                            "shared_plane_mode": self.shared_plane_mode,
                            "plane_reference_mode": plane_reference_mode,
                            "frame_plane_available": frame_plane_normal is not None,
                            "global_plane_available": global_plane_normal is not None,
                            **vlm_conf_info,
                            **quality_info,
                            **measurement,
                        }
                    )
                else:
                    all_anchor_measurements.append(
                        {
                            "image_path": image_path,
                            "frame_name": os.path.basename(image_path).split(".")[0],
                            "sequence_name": sequence_name,
                            "anchor_group": anchor.get("anchor_group", "unknown"),
                            "object_class": anchor.get("object_class"),
                            "object_id": anchor.get("object_id"),
                            "height_interval": height_interval,
                            "scale_interval": None,
                            "depth_geometry_filter_stage": "missing_relative_height_measurement",
                            "anchor_point_count": int(len(aligned_anchor_points)),
                            "plane_point_count": int(len(aligned_local_ground_points) if len(aligned_local_ground_points) > 0 else len(aligned_plane_points)),
                            "shared_plane_mode": self.shared_plane_mode,
                            "plane_reference_mode": plane_reference_mode,
                            "frame_plane_available": frame_plane_normal is not None,
                            "global_plane_available": global_plane_normal is not None,
                            **weight_info,
                            **vlm_conf_info,
                            **quality_info,
                        }
                    )

            if self.enable_visualization:
                self._save_batch_alignment_visualization(
                    aligned_plane_points,
                    aligned_anchor_data,
                    image_path,
                    sequence_name,
                )
        
        # Step 5: Compute global consistent scale
        if all_scale_intervals:
            variant_results = self._compute_variant_results(all_scale_intervals, all_scale_constraints)
            self.last_variant_results = variant_results
            self.last_bi_result = variant_results["consensus_results"].get("bi")
            self.last_ifpa_result = variant_results["consensus_results"].get("ifpa")
            self.last_consensus_result = variant_results["consensus_results"].get(self.consensus_solver)
            default_variant_key = variant_results["default_optimized_variant_key"]
            default_variant = variant_results["optimized_variants"][default_variant_key]
            consistent_scale = variant_results["consistent_scale"]
            consistent_count = variant_results["consistent_count"]
            optimized_scale = default_variant["scale"]
            self.last_optimization_result = dict(default_variant["optimization_info"])
            return consistent_scale, consistent_count, optimized_scale, all_scale_intervals, all_anchor_measurements
        else:
            empty_result = (
                self._empty_legacy_consensus_result()
                if self.consensus_solver == "legacy"
                else
                self._empty_ifpa_result()
                if self.consensus_solver == "ifpa"
                else self._empty_bi_result()
            )
            self.last_consensus_result = empty_result
            self.last_variant_results = {
                "consistent_scale": None,
                "consistent_count": 0,
                "consensus_results": {},
                "optimized_variants": {},
                "default_optimized_variant_key": self._resolve_default_optimized_variant_key(),
            }
            if self.consensus_solver == "legacy":
                self.last_bi_result = None
                self.last_ifpa_result = None
            elif self.consensus_solver == "ifpa":
                self.last_ifpa_result = empty_result
                self.last_bi_result = None
            else:
                self.last_bi_result = empty_result
                self.last_ifpa_result = None
            return None, 0, None, [], all_anchor_measurements

    def estimate_metric_scale(self, pointmap, conf, extrinsics, vlm_results, sequence_name=None):
        """
        Main pipeline for metric scale estimation using semantic planes based on pointmap.
        
        Args:
            pointmap (np.ndarray): [H, W, 3] point map containing 3D coordinates
            conf (np.ndarray): [H, W] confidence map
            extrinsics (np.ndarray): [3, 4] camera extrinsics matrix (w2c)
            vlm_results (dict): VLM semantic scale anchor results
            sequence_name (str): Name of the sequence being processed
            
        Returns:
            tuple: (consistent_scale, consistent_count, optimized_scale, scale_intervals)
        """
        self.last_bi_result = None
        self.last_ifpa_result = None
        self.last_consensus_result = None
        self.last_variant_results = None
        # Process each frame
        # Get frame data
        image_path = vlm_results["image_path"]
        semantic_planes = vlm_results.get("semantic_planes", [])
        height_anchors = self._select_height_anchors(vlm_results)

        # Step 0: Convert pointmap from camera to world coordinates
        # Extract rotation matrix R and translation vector t from extrinsics
        R = extrinsics[:, :3]
        t = extrinsics[:, 3:]
        
        # Compute inverse transformation to get from camera to world
        R_world = R.T
        t_world = -R_world @ t
        
        # Apply transformation to each point in pointmap
        H, W, _ = pointmap.shape
        pointmap_reshaped = pointmap.reshape(-1, 3)
        world_points = (R_world @ pointmap_reshaped.T + t_world).T
        world_pointmap = world_points.reshape(H, W, 3)

        # Step 1: Extract semantic plane mask
        # Use a dummy depth map for mask extraction since it only needs shape information
        dummy_depth = np.zeros_like(pointmap[..., 0])
        plane_mask = self.extract_semantic_plane_mask(
            dummy_depth, semantic_planes, image_path, sequence_name
        )
        
        # Step 2: Extract valid points from plane point cloud
        # 直接从world_pointmap中提取有效点，避免创建副本
        valid_plane_points, filtered_plane_mask = self._extract_valid_points(
            world_pointmap,
            plane_mask,
            conf_map=conf,
            conf_percentile=self.plane_conf_percentile,
            min_points=64,
        )
        
        # Check if we have enough points to fit a plane
        if len(valid_plane_points) < 3:
            print("Not enough points to fit plane, skipping")
            return None, 0, None, [], []
        
        # Step 3: Fit plane using RANSAC
        plane_normal, _ = self.fit_plane_ransac(valid_plane_points)
        # Check if a valid plane was found
        if plane_normal is None:
            print("No valid plane found, skipping alignment")
            return None, 0, None, [], []

        # Step 4: Align point cloud
        # 只对齐有效点，避免处理无效数据
        aligned_plane_points = self.align_point_cloud(valid_plane_points, plane_normal)
        
        # 检查对齐后的点
        if len(aligned_plane_points) == 0:
            print("No valid aligned plane points found")
            return None, 0, None, [], []

        scale_intervals = []
        scale_constraints = []
        anchor_measurements = []
        frame_conf_thresholds = self._compute_frame_conf_thresholds(conf)
        # 预处理：计算世界坐标系到对齐坐标系的变换，避免重复计算
        # 注意：这里我们直接对齐点，而不是整个点云
        
        # 批量处理高度锚点
        frame_name = os.path.basename(image_path).split(".")[0]
        ply_dir = os.path.join(
            get_sam_aligned_anchor_dir(sequence_name, dataset_name=config.dataset_name),
            frame_name,
        )
        
        # 只在需要时创建目录
        if self.write_ply:
            ensure_dir(ply_dir)

        for anchor in height_anchors:
            weight_info = self._extract_anchor_weight_info(anchor)
            vlm_conf_info = self._evaluate_vlm_anchor_confidence(anchor)
            if not vlm_conf_info["passed_vlm_confidence_filter"]:
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": anchor["height_interval"],
                        "scale_interval": None,
                        **weight_info,
                        **vlm_conf_info,
                    }
                )
                continue
            # Step 1: Extract semantic anchor mask
            anchor_mask = self.extract_semantic_anchor_mask(
                dummy_depth, anchor, image_path, sequence_name
            )
            quality_info = self._evaluate_anchor_quality(conf, anchor_mask, frame_conf_thresholds)
            if not quality_info["passed_quality_filter"]:
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": anchor["height_interval"],
                        "scale_interval": None,
                        "depth_geometry_filter_stage": "anchor_quality_filter",
                        "anchor_mask_pixel_count": int(np.count_nonzero(anchor_mask)),
                        **weight_info,
                        **vlm_conf_info,
                        **quality_info,
                    }
                )
                continue
            
            # Step 2: Extract anchor point cloud from world pointmap
            # 直接提取有效点，避免创建副本
            valid_anchor_points, filtered_anchor_mask = self._extract_valid_points(
                world_pointmap,
                anchor_mask,
                conf_map=conf,
                min_points=self._depth_geometry_min_points(),
            )
            
            if len(valid_anchor_points) == 0:
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": anchor["height_interval"],
                        "scale_interval": None,
                        "depth_geometry_filter_stage": "missing_valid_anchor_points",
                        "anchor_mask_pixel_count": int(np.count_nonzero(anchor_mask)),
                        "filtered_anchor_mask_pixel_count": int(np.count_nonzero(filtered_anchor_mask)),
                        "valid_anchor_point_count": 0,
                        **weight_info,
                        **vlm_conf_info,
                        **quality_info,
                    }
                )
                continue
            
            # Step 3: Align anchor point cloud
            aligned_anchor_points = self.align_point_cloud(valid_anchor_points, plane_normal)
            local_ground_points = self._extract_local_ground_points(
                world_pointmap,
                filtered_plane_mask,
                filtered_anchor_mask,
                conf_map=conf,
                conf_percentile=self.plane_conf_percentile,
            )
            aligned_local_ground_points = (
                self.align_point_cloud(local_ground_points, plane_normal)
                if len(local_ground_points) > 0
                else np.empty((0, 3))
            )

            # 保存对齐后的anchor点云为PLY文件（可选）
            if self.write_ply:
                # 构造PLY文件名：使用anchor类别和ID
                ply_filename = f"{anchor['object_class']}_anchor_{anchor['object_id']}.ply"
                ply_path = os.path.join(ply_dir, ply_filename)

                # 将numpy数组转换为Open3D点云对象并写入文件
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(aligned_anchor_points)
                o3d.io.write_point_cloud(ply_path, pcd)
            
            measurement = self._resolve_relative_height_measurement(
                aligned_anchor_points,
                aligned_local_ground_points if len(aligned_local_ground_points) > 0 else aligned_plane_points,
                ground_source="local" if len(aligned_local_ground_points) > 0 else "global",
            )
            ref_height = None if measurement is None else measurement["relative_height"]
            height_interval = anchor["height_interval"]
            real_min, real_max = height_interval[0], height_interval[1]
            if self._has_valid_relative_height(ref_height):
                scale_interval = [real_min / ref_height, real_max / ref_height]
                scale_intervals.append(scale_interval)
                scale_constraints.append(
                    {
                        "interval": scale_interval,
                        "weight": weight_info["combined_weight"],
                        "type_weight": weight_info["type_weight"],
                        "quality_weight": weight_info["quality_weight"],
                        "priority": weight_info["priority"],
                        "distance_level": weight_info["distance_level"],
                        "distance_factor": weight_info["distance_factor"],
                        "priority_factor": weight_info["priority_factor"],
                        "modulated_type_weight": weight_info["modulated_type_weight"],
                        "combined_weight": weight_info["combined_weight"],
                        "weight_formula": weight_info["weight_formula"],
                        "object_class": anchor.get("object_class"),
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_id": anchor.get("object_id"),
                    }
                )
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": height_interval,
                        "scale_interval": scale_interval,
                        **weight_info,
                        **vlm_conf_info,
                        **quality_info,
                        **measurement,
                    }
                )
            elif measurement is not None:
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": height_interval,
                        "scale_interval": None,
                        "depth_geometry_filter_stage": "invalid_relative_height",
                        **weight_info,
                        **vlm_conf_info,
                        **quality_info,
                        **measurement,
                    }
                )
            else:
                anchor_measurements.append(
                    {
                        "image_path": image_path,
                        "frame_name": frame_name,
                        "sequence_name": sequence_name,
                        "anchor_group": anchor.get("anchor_group", "unknown"),
                        "object_class": anchor.get("object_class"),
                        "object_id": anchor.get("object_id"),
                        "height_interval": height_interval,
                        "scale_interval": None,
                        "depth_geometry_filter_stage": "missing_relative_height_measurement",
                        "anchor_point_count": int(len(aligned_anchor_points)),
                        "plane_point_count": int(len(aligned_local_ground_points) if len(aligned_local_ground_points) > 0 else len(aligned_plane_points)),
                        **weight_info,
                        **vlm_conf_info,
                        **quality_info,
                    }
                )
        
        # Compute scales for this frame (for backward compatibility)
        if scale_intervals:
            variant_results = self._compute_variant_results(scale_intervals, scale_constraints)
            self.last_variant_results = variant_results
            self.last_bi_result = variant_results["consensus_results"].get("bi")
            self.last_ifpa_result = variant_results["consensus_results"].get("ifpa")
            self.last_consensus_result = variant_results["consensus_results"].get(self.consensus_solver)
            default_variant_key = variant_results["default_optimized_variant_key"]
            default_variant = variant_results["optimized_variants"][default_variant_key]
            consistent_scale = variant_results["consistent_scale"]
            consistent_count = variant_results["consistent_count"]
            optimized_scale = default_variant["scale"]
            self.last_optimization_result = dict(default_variant["optimization_info"])
            return consistent_scale, consistent_count, optimized_scale, scale_intervals, anchor_measurements
        else:
            empty_result = (
                self._empty_legacy_consensus_result()
                if self.consensus_solver == "legacy"
                else
                self._empty_ifpa_result()
                if self.consensus_solver == "ifpa"
                else self._empty_bi_result()
            )
            self.last_consensus_result = empty_result
            self.last_variant_results = {
                "consistent_scale": None,
                "consistent_count": 0,
                "consensus_results": {},
                "optimized_variants": {},
                "default_optimized_variant_key": self._resolve_default_optimized_variant_key(),
            }
            if self.consensus_solver == "legacy":
                self.last_bi_result = None
                self.last_ifpa_result = None
            elif self.consensus_solver == "ifpa":
                self.last_ifpa_result = empty_result
                self.last_bi_result = None
            else:
                self.last_bi_result = empty_result
                self.last_ifpa_result = None
            return None, 0, None, [], anchor_measurements
