import argparse
import copy
import json
import math
import os

import yaml


class Config:
    """YAML-backed configuration manager with light CLI overrides."""

    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.args = self._parse_args()
        self._config = self._load_yaml_config()
        self._active_semantic_scale_override = None
        self._active_semantic_scale_profile = None
        self._sequence_difficulty_summary = None
        self._sequence_difficulty_profile_cache = {}

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Metric-Depth configuration")
        parser.add_argument(
            "--config",
            type=str,
            default=os.path.join(self.project_root, "configs", "nuscenes", "lingbot_map_long.yaml"),
            help="Path to YAML config file",
        )
        parser.add_argument("--sequence-name", type=str, default=None, help="Optional override for sequence name")
        parser.add_argument("--frame-count", type=int, default=None, help="Optional override for frame count")
        parser.add_argument("--depth-model", type=str, default=None, help="Optional override for depth model")
        parser.add_argument("--parallel", dest="parallel", action="store_true", help="Enable parallel GPU scheduling")
        parser.add_argument("--disable-parallel", dest="parallel", action="store_false", help="Disable parallel GPU scheduling")
        parser.add_argument("--parallel-vlm-api", dest="parallel_vlm_api", action="store_true", help="Enable threaded API-based VLM calls")
        parser.add_argument("--disable-parallel-vlm-api", dest="parallel_vlm_api", action="store_false", help="Disable threaded API-based VLM calls")
        parser.add_argument("--max-gpus", type=int, default=None, help="Optional override for max GPU count")
        parser.add_argument("--max-vlm-workers", type=int, default=None, help="Optional override for VLM API worker count")
        parser.add_argument("--enable-visualization", dest="enable_visualization", action="store_true", help="Enable visualization outputs")
        parser.add_argument("--disable-visualization", dest="enable_visualization", action="store_false", help="Disable visualization outputs")
        parser.set_defaults(
            parallel=None,
            parallel_vlm_api=None,
            enable_visualization=None,
        )
        return parser.parse_args()

    @staticmethod
    def _deep_update(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _load_yaml_file(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        return data

    @staticmethod
    def _normalize_string_list(raw_value):
        if raw_value in (None, "", [], ()):
            return tuple()
        if isinstance(raw_value, str):
            items = [item.strip() for item in raw_value.split(",")]
        elif isinstance(raw_value, (list, tuple, set)):
            items = [str(item).strip() for item in raw_value]
        else:
            raise ValueError(
                "Expected a list/tuple/set of strings or a comma-separated string."
            )
        normalized = [item.lower() for item in items if item]
        return tuple(normalized)

    def _load_yaml_config(self):
        config_path = os.path.abspath(self.args.config)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config_data = self._load_yaml_file(config_path)
        self._apply_cli_overrides(config_data)
        return config_data

    def _apply_cli_overrides(self, config_data):
        runtime_cfg = config_data.setdefault("runtime", {})
        parallel_cfg = config_data.setdefault("parallel", {})

        if self.args.sequence_name is not None:
            runtime_cfg["sequence_name"] = self.args.sequence_name
        if self.args.frame_count is not None:
            runtime_cfg["frame_count"] = self.args.frame_count
        if self.args.depth_model is not None:
            runtime_cfg["depth_model"] = self.args.depth_model
        if self.args.parallel is not None:
            parallel_cfg["enable"] = self.args.parallel
        if self.args.parallel_vlm_api is not None:
            parallel_cfg["vlm_api_enable"] = self.args.parallel_vlm_api
        if self.args.max_gpus is not None:
            parallel_cfg["max_gpus"] = self.args.max_gpus
        if self.args.max_vlm_workers is not None:
            parallel_cfg["max_vlm_workers"] = self.args.max_vlm_workers
        if self.args.enable_visualization is not None:
            runtime_cfg["enable_visualization"] = self.args.enable_visualization

    @staticmethod
    def _normalize_sequence_names(value):
        if value is None:
            return tuple()
        if isinstance(value, (list, tuple, set)):
            raw_items = value
        else:
            value = str(value).strip()
            if not value or value.lower() == "none":
                return tuple()
            raw_items = value.split(",") if "," in value else [value]

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

    def _get_section(self, section_name):
        section = self._config.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        if section_name == "semantic_scale" and self._active_semantic_scale_override:
            merged = copy.deepcopy(section)
            self._deep_update(merged, copy.deepcopy(self._active_semantic_scale_override))
            return merged
        return section

    def _get_base_section(self, section_name):
        section = self._config.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping")
        return section

    def _resolve_template(self, template, extra_context=None):
        if template is None:
            return None
        if not isinstance(template, str):
            return template

        runtime_cfg = self._get_section("runtime")
        sequence_names = self._normalize_sequence_names(runtime_cfg.get("sequence_name"))
        context = {
            "project_root": self.project_root,
            "sequence_name": sequence_names[0] if len(sequence_names) == 1 else "",
            "dataset": str(runtime_cfg.get("dataset", "") or "").strip(),
            "depth_model": str(runtime_cfg.get("depth_model", "") or "").strip(),
        }
        if extra_context:
            context.update(extra_context)

        paths_cfg = self._get_section("paths")
        models_cfg = self._get_section("models")
        unresolved = dict(paths_cfg)
        unresolved.update(
            {
                key: value
                for key, value in models_cfg.items()
                if isinstance(value, str)
            }
        )
        resolved = {}

        def resolve_key(key):
            if key in resolved:
                return resolved[key]
            if key in context:
                return context[key]

            value = unresolved.get(key)
            if value is None:
                raise KeyError(key)

            if not isinstance(value, str):
                resolved[key] = value
                return value

            local_context = dict(context)
            local_context.update(resolved)
            resolved_value = value.format_map(
                _SafeFormatDict(local_context, resolve_key)
            )
            resolved[key] = (
                os.path.normpath(resolved_value)
                if _looks_like_path(resolved_value)
                else resolved_value
            )
            return resolved[key]

        raw_data_path = runtime_cfg.get("data_path")
        if isinstance(raw_data_path, str):
            data_path = raw_data_path.format_map(_SafeFormatDict(dict(context), resolve_key))
            context["data_path"] = os.path.normpath(data_path) if _looks_like_path(data_path) else data_path
            if context["data_path"] and "dataset_name" not in context:
                context["dataset_name"] = os.path.basename(os.path.normpath(str(context["data_path"])))

        for key in unresolved:
            resolve_key(key)

        context.update(resolved)
        resolved_value = template.format_map(_SafeFormatDict(context, lambda item: context[item]))
        return os.path.normpath(resolved_value) if _looks_like_path(resolved_value) else resolved_value

    @staticmethod
    def _normalize_metric_variant_name(raw_value):
        value = str(raw_value or "").strip().lower()
        alias_map = {
            "relative": "relative",
            "relative_scale": "relative",
            "consistent": "consistent",
            "consistent_scale": "consistent",
            "optimized_legacy": "optimized_legacy",
            "optimized_scale_legacy": "optimized_legacy",
            "optimized_original": "optimized_original",
            "optimized_scale_original": "optimized_original",
            "optimized_weighted_bi": "optimized_weighted_bi",
            "optimized_scale_weighted_bi": "optimized_weighted_bi",
            "optimized_weighted_ifpa": "optimized_weighted_ifpa",
            "optimized_scale_weighted_ifpa": "optimized_weighted_ifpa",
            "selected": "selected",
            "selected_scale": "selected",
        }
        if value not in alias_map:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.metric_variant: {raw_value}. "
                "Expected one of: relative, consistent, optimized_legacy, "
                "optimized_original, optimized_weighted_bi, optimized_weighted_ifpa, selected."
            )
        return alias_map[value]

    @staticmethod
    def _metric_variant_payload_keys(variant_key):
        alias_map = {
            "relative": ("relative", "relative_scale"),
            "consistent": ("consistent", "consistent_scale"),
            "optimized_legacy": ("optimized_legacy", "optimized_scale_legacy"),
            "optimized_original": ("optimized_original", "optimized_scale_original"),
            "optimized_weighted_bi": ("optimized_weighted_bi", "optimized_scale_weighted_bi"),
            "optimized_weighted_ifpa": ("optimized_weighted_ifpa", "optimized_scale_weighted_ifpa"),
            "selected": ("selected", "selected_scale"),
        }
        return alias_map.get(variant_key, (variant_key,))

    def _get_sequence_difficulty_variant_payload(self, sequence_result, variant_key):
        for payload_key in self._metric_variant_payload_keys(variant_key):
            variant_payload = sequence_result.get(payload_key)
            if isinstance(variant_payload, dict):
                return variant_payload
        raise ValueError(
            "Missing sequence metric variant "
            f"{variant_key} for sequence {sequence_result.get('sequence_name')}"
        )

    @classmethod
    def _normalize_metric_variant_list(cls, raw_value):
        if raw_value in (None, "", [], (), set()):
            return tuple()
        if isinstance(raw_value, str):
            items = [item.strip() for item in raw_value.split(",")]
        elif isinstance(raw_value, (list, tuple, set)):
            items = [str(item).strip() for item in raw_value]
        else:
            raise ValueError(
                "Expected semantic_scale.sequence_difficulty candidate variants to be a list/tuple/set "
                "or a comma-separated string."
            )
        return tuple(
            cls._normalize_metric_variant_name(item)
            for item in items
            if item
        )

    @staticmethod
    def _normalize_metric_name(raw_value):
        value = str(raw_value or "").strip().lower()
        alias_map = {
            "rel": "rel",
            "mae": "mae",
            "rmse": "rmse",
            "delta1": "delta1",
            "d1": "delta1",
        }
        if value not in alias_map:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.metric_name: {raw_value}. "
                "Expected one of: rel, mae, rmse, delta1."
            )
        return alias_map[value]

    @staticmethod
    def _normalize_distance_bin_name(raw_value):
        if raw_value in (None, "", "none", "None"):
            return None
        value = str(raw_value or "").strip().lower()
        alias_map = {
            "0-20m": "0_20m",
            "0_20m": "0_20m",
            "20-40m": "20_40m",
            "20_40m": "20_40m",
            "40-60m": "40_60m",
            "40_60m": "40_60m",
            "60m+": "60m_plus",
            "60m_plus": "60m_plus",
            "60m-plus": "60m_plus",
        }
        if value not in alias_map:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.distance_bin: {raw_value}. "
                "Expected one of: 0-20m, 20-40m, 40-60m, 60m+."
            )
        return alias_map[value]

    @staticmethod
    def _normalize_metric_scope_name(raw_value):
        value = str(raw_value or "").strip().lower()
        valid_values = {"depth", "point", "depth_frame_mean", "point_frame_mean"}
        if value not in valid_values:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.metric_scope: {raw_value}. "
                "Expected one of: depth, point, depth_frame_mean, point_frame_mean."
            )
        return value

    def _validate_profile_override_key(self, key):
        valid_keys = {
            "consensus_prior_strength", "distance_weight_near", "distance_weight_middle",
            "distance_weight_distant", "dynamic_anchor_height_scale",
            "optimizer_dynamic_classes", "optimizer_static_classes",
            "height_anchor_source", "anchor_quality_min_percentile",
            "anchor_quality_max_percentile", "depth_geometry_min_relative_height",
            "default_optimized_variant", "weighted_search_mode",
            "optimized_scale_priority_modulation_strength", "max_optimized_shift_ratio",
            "min_optimized_to_consistent_ratio", "selected_scale_default",
            "vlm_dynamic_anchor_confidence_threshold",
            "vlm_static_anchor_confidence_threshold",
        }
        if key not in valid_keys:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.profiles override: {key}. "
                f"Expected one of: {sorted(valid_keys)}"
            )

    def _sequence_difficulty_section(self):
        semantic_scale_cfg = self._get_base_section("semantic_scale")
        section = semantic_scale_cfg.get("sequence_difficulty", {})
        if section in (None, ""):
            return {}
        if not isinstance(section, dict):
            raise ValueError("Config section 'semantic_scale.sequence_difficulty' must be a mapping")
        return section

    @property
    def sequence_difficulty_enabled(self):
        return bool(self._sequence_difficulty_section().get("enabled", False))

    @property
    def sequence_difficulty_mode(self):
        section = self._sequence_difficulty_section()
        raw_value = section.get("mode")
        if raw_value in (None, ""):
            # Keep the simple config style ergonomic:
            # if the user provides baseline-vs-candidate hints, treat it as the
            # easy/mid/hard delta1 split instead of forcing an explicit mode key.
            if any(
                key in section
                for key in ("easy_d1_gain_threshold", "candidate_variants", "baseline_variant")
            ):
                return "best_variant_vs_baseline"
            raw_value = "single_metric"
        value = str(raw_value or "").strip().lower()
        if value not in {"single_metric", "best_variant_vs_baseline"}:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.mode: {raw_value}. "
                "Expected one of: single_metric, best_variant_vs_baseline."
            )
        return value

    @property
    def sequence_difficulty_dataset_summary_path(self):
        raw_value = self._sequence_difficulty_section().get("dataset_summary_path")
        if raw_value in (None, "", "none", "None"):
            return None
        return self._resolve_template(raw_value)

    @property
    def sequence_difficulty_metric_variant(self):
        raw_value = self._sequence_difficulty_section().get("metric_variant", "optimized_weighted_ifpa")
        return self._normalize_metric_variant_name(raw_value)

    @property
    def sequence_difficulty_metric_scope(self):
        raw_value = self._sequence_difficulty_section().get("metric_scope", "depth_frame_mean")
        return self._normalize_metric_scope_name(raw_value)

    @property
    def sequence_difficulty_metric_name(self):
        raw_value = self._sequence_difficulty_section().get("metric_name", "rel")
        return self._normalize_metric_name(raw_value)

    @property
    def sequence_difficulty_easy_d1_gain_threshold(self):
        value = float(self._sequence_difficulty_section().get("easy_d1_gain_threshold", 0.5))
        if value < 0.0:
            raise ValueError(
                "semantic_scale.sequence_difficulty.easy_d1_gain_threshold must be >= 0.0."
            )
        return value

    @property
    def sequence_difficulty_distance_bin(self):
        raw_value = self._sequence_difficulty_section().get("distance_bin")
        return self._normalize_distance_bin_name(raw_value)

    @property
    def sequence_difficulty_metric_goal(self):
        raw_value = self._sequence_difficulty_section().get("metric_goal", "lower_is_better")
        value = str(raw_value or "").strip().lower()
        if value not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(
                f"Unsupported semantic_scale.sequence_difficulty.metric_goal: {raw_value}. "
                "Expected one of: lower_is_better, higher_is_better."
            )
        return value

    @property
    def sequence_difficulty_easy_threshold(self):
        value = float(self._sequence_difficulty_section().get("easy_threshold", 0.11))
        return value

    @property
    def sequence_difficulty_mid_threshold(self):
        value = float(self._sequence_difficulty_section().get("mid_threshold", 0.18))
        return value

    @property
    def sequence_difficulty_hard_threshold(self):
        value = float(self._sequence_difficulty_section().get("hard_threshold", 0.20))
        return value

    @property
    def sequence_difficulty_candidate_variants(self):
        raw_value = self._sequence_difficulty_section().get(
            "candidate_variants",
            (
                "consistent",
                "optimized_legacy",
                "optimized_original",
                "optimized_weighted_bi",
                "optimized_weighted_ifpa",
            ),
        )
        enabled_metric_variants = set(self.enabled_metric_variants)
        values = tuple(
            value
            for value in self._normalize_metric_variant_list(raw_value)
            if value in enabled_metric_variants
        )
        if self.sequence_difficulty_mode == "best_variant_vs_baseline" and not values:
            raise ValueError(
                "semantic_scale.sequence_difficulty.candidate_variants is required when mode is best_variant_vs_baseline."
            )
        if (
            self.sequence_difficulty_mode == "best_variant_vs_baseline"
            and self.sequence_difficulty_baseline_variant in values
        ):
            raise ValueError(
                "semantic_scale.sequence_difficulty.candidate_variants must not include "
                "baseline_variant when mode is best_variant_vs_baseline."
            )
        return values

    @property
    def sequence_difficulty_baseline_variant(self):
        raw_value = self._sequence_difficulty_section().get("baseline_variant", "relative")
        return self._normalize_metric_variant_name(raw_value)

    @property
    def sequence_difficulty_degrade_check_variant(self):
        raw_value = self._sequence_difficulty_section().get("degrade_check_variant", "consistent")
        return self._normalize_metric_variant_name(raw_value)

    @property
    def sequence_difficulty_degrade_ratio_threshold(self):
        value = float(self._sequence_difficulty_section().get("degrade_ratio_threshold", 1.5))
        if value <= 0.0:
            raise ValueError(
                "semantic_scale.sequence_difficulty.degrade_ratio_threshold must be > 0.0."
            )
        return value

    @property
    def sequence_difficulty_profiles(self):
        section = self._sequence_difficulty_section().get("profiles", {})
        if not isinstance(section, dict):
            raise ValueError("semantic_scale.sequence_difficulty.profiles must be a mapping")
        if not section and not self.sequence_difficulty_enabled:
            return {}

        normalized_profiles = {}
        expected_names = ("easy", "mid", "hard")
        for profile_name in expected_names:
            raw_profile = section.get(profile_name, {})
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    f"semantic_scale.sequence_difficulty.profiles.{profile_name} must be a mapping"
                )
            normalized_profile = {}
            for key, value in raw_profile.items():
                self._validate_profile_override_key(key)
                if key in {"optimizer_dynamic_classes", "optimizer_static_classes"}:
                    normalized_profile[key] = self._normalize_string_list(value)
                elif key in {"height_anchor_source", "default_optimized_variant",
                             "weighted_search_mode", "selected_scale_default"}:
                    normalized_profile[key] = str(value).strip()
                elif key in {"anchor_quality_min_percentile", "anchor_quality_max_percentile",
                             "depth_geometry_min_relative_height", "max_optimized_shift_ratio",
                             "optimized_scale_priority_modulation_strength", "consensus_prior_strength"}:
                    parsed_value = float(value)
                    if not math.isfinite(parsed_value) or parsed_value < 0.0:
                        raise ValueError(f"Invalid {profile_name}.{key}: {value}; expected finite >= 0")
                    normalized_profile[key] = parsed_value
                elif key in {"vlm_dynamic_anchor_confidence_threshold",
                             "vlm_static_anchor_confidence_threshold"}:
                    parsed_value = float(value)
                    if not 0.0 <= parsed_value <= 1.0:
                        raise ValueError(f"Invalid {profile_name}.{key}: {value}; expected [0,1]")
                    normalized_profile[key] = parsed_value
                elif key == "dynamic_anchor_height_scale":
                    parsed_value = float(value)
                    if not math.isfinite(parsed_value) or not 0.0 < parsed_value <= 3.0:
                        raise ValueError(f"Invalid {profile_name}.{key}: {value}; expected (0,3]")
                    normalized_profile[key] = parsed_value
                elif key == "min_optimized_to_consistent_ratio":
                    parsed_value = float(value)
                    if not 0.0 < parsed_value <= 1.0:
                        raise ValueError(f"Invalid {profile_name}.{key}: {value}; expected (0,1]")
                    normalized_profile[key] = parsed_value
                else:
                    parsed_value = float(value)
                    if not math.isfinite(parsed_value) or parsed_value <= 0.0:
                        raise ValueError(f"Invalid {profile_name}.{key}: {value}; expected finite > 0")
                    normalized_profile[key] = parsed_value
            normalized_profiles[profile_name] = normalized_profile

        if self.sequence_difficulty_enabled:
            missing_names = [name for name in expected_names if name not in normalized_profiles]
            if missing_names:
                raise ValueError(
                    "semantic_scale.sequence_difficulty.profiles is missing required entries: "
                    + ", ".join(missing_names)
                )
        return normalized_profiles

    def _validate_sequence_difficulty_thresholds(self):
        mode = self.sequence_difficulty_mode
        if mode == "single_metric":
            easy_threshold = self.sequence_difficulty_easy_threshold
            mid_threshold = self.sequence_difficulty_mid_threshold
            metric_goal = self.sequence_difficulty_metric_goal
            if metric_goal == "lower_is_better":
                if easy_threshold > mid_threshold:
                    raise ValueError(
                        "semantic_scale.sequence_difficulty.easy_threshold must be <= mid_threshold "
                        "when metric_goal is lower_is_better."
                    )
            else:
                if easy_threshold < mid_threshold:
                    raise ValueError(
                        "semantic_scale.sequence_difficulty.easy_threshold must be >= mid_threshold "
                        "when metric_goal is higher_is_better."
                    )
        else:
            _ = self.sequence_difficulty_easy_d1_gain_threshold
            _ = self.sequence_difficulty_candidate_variants
            _ = self.sequence_difficulty_baseline_variant

    def _load_sequence_difficulty_summary(self):
        if self._sequence_difficulty_summary is not None:
            return self._sequence_difficulty_summary

        summary_path = self.sequence_difficulty_dataset_summary_path
        if not summary_path:
            return None
        if not os.path.exists(summary_path):
            return None

        with open(summary_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(
                "semantic_scale.sequence_difficulty.dataset_summary_path must point to a JSON object"
            )
        sequence_results = payload.get("sequence_results")
        if not isinstance(sequence_results, list):
            raise ValueError(
                "semantic_scale.sequence_difficulty.dataset_summary_path must contain sequence_results"
            )
        self._sequence_difficulty_summary = payload
        return payload

    def _extract_sequence_difficulty_metric_value(
        self,
        sequence_result,
        variant_key=None,
        metric_name=None,
        metric_scope=None,
    ):
        if variant_key is None:
            variant_key = self.sequence_difficulty_metric_variant
        if metric_scope is None:
            metric_scope = self.sequence_difficulty_metric_scope
        if metric_name is None:
            metric_name = self.sequence_difficulty_metric_name
        distance_bin = self.sequence_difficulty_distance_bin

        variant_payload = self._get_sequence_difficulty_variant_payload(sequence_result, variant_key)

        if distance_bin is not None:
            distance_bins = variant_payload.get("distance_bins")
            if not isinstance(distance_bins, dict):
                raise ValueError(
                    "Missing distance_bins for sequence "
                    f"{sequence_result.get('sequence_name')} variant {variant_key}"
                )
            variant_payload = distance_bins.get(distance_bin)
            if not isinstance(variant_payload, dict):
                raise ValueError(
                    f"Missing distance bin {distance_bin} for sequence "
                    f"{sequence_result.get('sequence_name')} variant {variant_key}"
                )

        metric_payload = variant_payload.get(metric_scope)
        if not isinstance(metric_payload, dict):
            raise ValueError(
                f"Missing metric scope {metric_scope} for sequence "
                f"{sequence_result.get('sequence_name')} variant {variant_key}"
            )
        metric_value = metric_payload.get(metric_name)
        if metric_value is None:
            raise ValueError(
                f"Missing metric {metric_name} for sequence "
                f"{sequence_result.get('sequence_name')} variant {variant_key}"
            )
        try:
            return float(metric_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid metric value for sequence {sequence_result.get('sequence_name')}: "
                f"{metric_value}"
            ) from exc

    def _extract_sequence_difficulty_frames_evaluated(self, sequence_result, variant_key=None):
        if variant_key is None:
            variant_key = self.sequence_difficulty_metric_variant
        variant_payload = self._get_sequence_difficulty_variant_payload(sequence_result, variant_key)
        raw_value = variant_payload.get("frames_evaluated", 0)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid frames_evaluated for sequence {sequence_result.get('sequence_name')}: {raw_value}"
            ) from exc
        if value < 0:
            raise ValueError(
                f"Invalid frames_evaluated for sequence {sequence_result.get('sequence_name')}: {raw_value}"
            )
        return value

    def _classify_sequence_difficulty_single_metric(self, matched_result, sequence_name):
        frames_evaluated = self._extract_sequence_difficulty_frames_evaluated(matched_result)
        easy_threshold = self.sequence_difficulty_easy_threshold
        mid_threshold = self.sequence_difficulty_mid_threshold
        metric_goal = self.sequence_difficulty_metric_goal

        if frames_evaluated <= 0:
            metric_value = None
            profile_name = "hard"
            metric_status = "no_evaluated_frames"
        else:
            metric_value = self._extract_sequence_difficulty_metric_value(matched_result)
            metric_status = "ok"

            if metric_goal == "lower_is_better":
                if metric_value <= easy_threshold:
                    profile_name = "easy"
                elif metric_value <= mid_threshold:
                    profile_name = "mid"
                else:
                    profile_name = "hard"
            else:
                if metric_value >= easy_threshold:
                    profile_name = "easy"
                elif metric_value >= mid_threshold:
                    profile_name = "mid"
                else:
                    profile_name = "hard"

        return {
            "sequence_name": sequence_name,
            "profile_name": profile_name,
            "metric_variant": self.sequence_difficulty_metric_variant,
            "metric_scope": self.sequence_difficulty_metric_scope,
            "metric_name": self.sequence_difficulty_metric_name,
            "distance_bin": self.sequence_difficulty_distance_bin,
            "metric_goal": metric_goal,
            "metric_value": metric_value,
            "metric_status": metric_status,
            "easy_threshold": easy_threshold,
            "mid_threshold": mid_threshold,
            "frames_evaluated": frames_evaluated,
        }

    def _select_best_sequence_variant_metric(self, matched_result):
        metric_goal = self.sequence_difficulty_metric_goal
        metrics = {}
        missing_variants = {}
        for variant_key in self.sequence_difficulty_candidate_variants:
            frames_evaluated = self._extract_sequence_difficulty_frames_evaluated(
                matched_result,
                variant_key=variant_key,
            )
            if frames_evaluated <= 0:
                missing_variants[variant_key] = "no_evaluated_frames"
                continue
            try:
                metrics[variant_key] = self._extract_sequence_difficulty_metric_value(
                    matched_result,
                    variant_key=variant_key,
                )
            except ValueError as exc:
                missing_variants[variant_key] = str(exc)
                continue
        if not metrics:
            raise ValueError(
                "No valid candidate variant metrics were found for sequence "
                f"{matched_result.get('sequence_name')}"
            )
        if metric_goal == "lower_is_better":
            best_variant, best_metric = min(metrics.items(), key=lambda item: item[1])
        else:
            best_variant, best_metric = max(metrics.items(), key=lambda item: item[1])
        return best_variant, best_metric, metrics, missing_variants

    def _collect_sequence_difficulty_candidate_metrics(self, matched_result):
        candidate_metrics = {}
        missing_variants = {}
        metric_scope = self.sequence_difficulty_metric_scope
        for variant_key in self.sequence_difficulty_candidate_variants:
            frames_evaluated = self._extract_sequence_difficulty_frames_evaluated(
                matched_result,
                variant_key=variant_key,
            )
            if frames_evaluated <= 0:
                missing_variants[variant_key] = "no_evaluated_frames"
                continue
            try:
                rel_value = self._extract_sequence_difficulty_metric_value(
                    matched_result,
                    variant_key=variant_key,
                    metric_scope=metric_scope,
                    metric_name="rel",
                )
                d1_value = self._extract_sequence_difficulty_metric_value(
                    matched_result,
                    variant_key=variant_key,
                    metric_scope=metric_scope,
                    metric_name="delta1",
                )
            except ValueError as exc:
                missing_variants[variant_key] = str(exc)
                continue
            candidate_metrics[variant_key] = {
                "rel": rel_value,
                "delta1": d1_value,
                "frames_evaluated": frames_evaluated,
            }
        return candidate_metrics, missing_variants

    def _classify_sequence_difficulty_best_variant_vs_baseline(self, matched_result, sequence_name):
        baseline_variant = self.sequence_difficulty_baseline_variant
        metric_scope = self.sequence_difficulty_metric_scope
        frames_evaluated = self._extract_sequence_difficulty_frames_evaluated(
            matched_result,
            variant_key=baseline_variant,
        )
        easy_d1_gain_threshold = self.sequence_difficulty_easy_d1_gain_threshold
        metric_goal = self.sequence_difficulty_metric_goal

        if metric_goal != "lower_is_better":
            raise ValueError(
                "semantic_scale.sequence_difficulty.mode=best_variant_vs_baseline currently "
                "expects metric_goal=lower_is_better."
            )

        if frames_evaluated <= 0:
            return {
                "sequence_name": sequence_name,
                "profile_name": "hard",
                "metric_variant": None,
                "metric_scope": metric_scope,
                "metric_name": "rel+delta1",
                "distance_bin": self.sequence_difficulty_distance_bin,
                "metric_goal": metric_goal,
                "metric_value": None,
                "metric_status": "no_evaluated_frames",
                "easy_d1_gain_threshold": easy_d1_gain_threshold,
                "frames_evaluated": frames_evaluated,
                "candidate_variants": list(self.sequence_difficulty_candidate_variants),
                "baseline_variant": baseline_variant,
                "classification_reason": "no_evaluated_frames",
                "trigger_variant": None,
                "trigger_metric": None,
                "trigger_d1_gain": None,
                "trigger_variant_metrics": None,
                "baseline_metric": None,
                "baseline_rel": None,
                "baseline_delta1": None,
                "valid_candidate_variants": [],
                "hard_variants": [],
                "per_variant_metrics": {},
            }

        try:
            baseline_rel = self._extract_sequence_difficulty_metric_value(
                matched_result,
                variant_key=baseline_variant,
                metric_scope=metric_scope,
                metric_name="rel",
            )
            baseline_d1 = self._extract_sequence_difficulty_metric_value(
                matched_result,
                variant_key=baseline_variant,
                metric_scope=metric_scope,
                metric_name="delta1",
            )
        except ValueError:
            return {
                "sequence_name": sequence_name,
                "profile_name": "hard",
                "metric_variant": None,
                "metric_scope": metric_scope,
                "metric_name": "rel+delta1",
                "distance_bin": self.sequence_difficulty_distance_bin,
                "metric_goal": metric_goal,
                "metric_value": None,
                "metric_status": "missing_baseline_metrics",
                "easy_d1_gain_threshold": easy_d1_gain_threshold,
                "frames_evaluated": frames_evaluated,
                "candidate_variants": list(self.sequence_difficulty_candidate_variants),
                "baseline_variant": baseline_variant,
                "classification_reason": "missing_baseline_metrics",
                "trigger_variant": None,
                "trigger_metric": None,
                "trigger_d1_gain": None,
                "trigger_variant_metrics": None,
                "baseline_metric": None,
                "baseline_rel": None,
                "baseline_delta1": None,
                "valid_candidate_variants": [],
                "hard_variants": [],
                "per_variant_metrics": {},
            }

        candidate_metrics, missing_variant_metrics = self._collect_sequence_difficulty_candidate_metrics(
            matched_result
        )

        if not candidate_metrics:
            return {
                "sequence_name": sequence_name,
                "profile_name": "hard",
                "metric_variant": None,
                "metric_scope": metric_scope,
                "metric_name": "rel+delta1",
                "distance_bin": self.sequence_difficulty_distance_bin,
                "metric_goal": metric_goal,
                "metric_value": None,
                "metric_status": "no_valid_candidate_variants",
                "easy_d1_gain_threshold": easy_d1_gain_threshold,
                "frames_evaluated": frames_evaluated,
                "candidate_variants": list(self.sequence_difficulty_candidate_variants),
                "baseline_variant": baseline_variant,
                "classification_reason": "no_valid_candidate_variants",
                "trigger_variant": None,
                "trigger_metric": None,
                "trigger_d1_gain": None,
                "trigger_variant_metrics": None,
                "baseline_metric": {"rel": baseline_rel, "delta1": baseline_d1},
                "baseline_rel": baseline_rel,
                "baseline_delta1": baseline_d1,
                "valid_candidate_variants": [],
                "hard_variants": [],
                "per_variant_metrics": {},
                "missing_variant_metrics": dict(missing_variant_metrics),
            }

        per_variant_metrics = {}
        easy_candidates = []
        hard_variants = []
        for variant_key, metrics in candidate_metrics.items():
            d1_gain = metrics["delta1"] - baseline_d1
            rel_delta = metrics["rel"] - baseline_rel
            per_variant_metrics[variant_key] = {
                "rel": metrics["rel"],
                "delta1": metrics["delta1"],
                "d1_gain_vs_baseline": d1_gain,
                "rel_delta_vs_baseline": rel_delta,
                "frames_evaluated": metrics["frames_evaluated"],
            }
            if d1_gain >= easy_d1_gain_threshold:
                easy_candidates.append(
                    (variant_key, d1_gain, metrics["rel"], metrics["delta1"])
                )
            if metrics["delta1"] < baseline_d1:
                hard_variants.append(variant_key)

        if easy_candidates:
            trigger_variant, trigger_d1_gain, _, _ = max(
                easy_candidates,
                key=lambda item: (item[1], -item[2], item[0]),
            )
            profile_name = "easy"
            classification_reason = (
                f"{trigger_variant}.delta1_gain_vs_{baseline_variant}"
                f">={easy_d1_gain_threshold:.4f}"
            )
            trigger_variant_metrics = copy.deepcopy(per_variant_metrics[trigger_variant])
            trigger_metric = {
                "d1_gain_vs_baseline": trigger_d1_gain,
            }
        elif hard_variants:
            profile_name = "hard"
            trigger_variant = None
            trigger_d1_gain = None
            trigger_variant_metrics = None
            trigger_metric = {
                "baseline_delta1": baseline_d1,
            }
            classification_reason = (
                "any_valid_candidate_has_lower_delta1_than_"
                f"{baseline_variant}"
            )
        else:
            profile_name = "mid"
            trigger_variant = None
            trigger_d1_gain = None
            trigger_variant_metrics = None
            trigger_metric = None
            classification_reason = "mixed_delta1_relative_to_baseline"

        return {
            "sequence_name": sequence_name,
            "profile_name": profile_name,
            "metric_variant": trigger_variant,
            "metric_scope": metric_scope,
            "metric_name": "rel+delta1",
            "distance_bin": self.sequence_difficulty_distance_bin,
            "metric_goal": metric_goal,
            "metric_value": None,
            "metric_status": "ok",
            "easy_d1_gain_threshold": easy_d1_gain_threshold,
            "frames_evaluated": frames_evaluated,
            "candidate_variants": list(self.sequence_difficulty_candidate_variants),
            "baseline_variant": baseline_variant,
            "classification_reason": classification_reason,
            "trigger_variant": trigger_variant,
            "trigger_metric": trigger_metric,
            "trigger_d1_gain": trigger_d1_gain,
            "trigger_variant_metrics": trigger_variant_metrics,
            "baseline_metric": {"rel": baseline_rel, "delta1": baseline_d1},
            "baseline_rel": baseline_rel,
            "baseline_delta1": baseline_d1,
            "valid_candidate_variants": list(per_variant_metrics),
            "hard_variants": list(hard_variants),
            "per_variant_metrics": dict(per_variant_metrics),
            "missing_variant_metrics": dict(missing_variant_metrics),
        }

    def classify_sequence_difficulty_result(self, sequence_result, include_overrides=True):
        if not self.sequence_difficulty_enabled:
            return None
        if not isinstance(sequence_result, dict):
            raise ValueError("sequence_result must be a mapping")

        sequence_name = str(sequence_result.get("sequence_name", "")).strip()
        if not sequence_name:
            raise ValueError("sequence_result.sequence_name is required")

        self._validate_sequence_difficulty_thresholds()
        if self.sequence_difficulty_mode == "best_variant_vs_baseline":
            profile = self._classify_sequence_difficulty_best_variant_vs_baseline(
                sequence_result,
                sequence_name,
            )
        else:
            profile = self._classify_sequence_difficulty_single_metric(
                sequence_result,
                sequence_name,
            )

        profile["mode"] = self.sequence_difficulty_mode
        if include_overrides:
            profile["overrides"] = copy.deepcopy(
                self.sequence_difficulty_profiles[profile["profile_name"]]
            )
        return profile

    def _missing_sequence_difficulty_profile(self, sequence_name):
        profile_name = "hard"
        common = {
            "sequence_name": sequence_name,
            "profile_name": profile_name,
            "metric_scope": self.sequence_difficulty_metric_scope,
            "distance_bin": self.sequence_difficulty_distance_bin,
            "metric_goal": self.sequence_difficulty_metric_goal,
            "metric_value": None,
            "metric_status": "missing_sequence_result",
            "frames_evaluated": 0,
            "classification_reason": (
                "sequence_not_found_in_dataset_summary:"
                f"{self.sequence_difficulty_dataset_summary_path}"
            ),
        }
        if self.sequence_difficulty_mode == "best_variant_vs_baseline":
            common.update(
                {
                    "metric_variant": None,
                    "metric_name": "rel+delta1",
                    "easy_d1_gain_threshold": self.sequence_difficulty_easy_d1_gain_threshold,
                    "candidate_variants": list(self.sequence_difficulty_candidate_variants),
                    "baseline_variant": self.sequence_difficulty_baseline_variant,
                    "trigger_variant": None,
                    "trigger_metric": None,
                    "trigger_d1_gain": None,
                    "trigger_variant_metrics": None,
                    "baseline_metric": None,
                    "baseline_rel": None,
                    "baseline_delta1": None,
                    "valid_candidate_variants": [],
                    "hard_variants": [],
                    "per_variant_metrics": {},
                    "missing_variant_metrics": {
                        variant: "missing_sequence_result"
                        for variant in self.sequence_difficulty_candidate_variants
                    },
                }
            )
        else:
            common.update(
                {
                    "metric_variant": self.sequence_difficulty_metric_variant,
                    "metric_name": self.sequence_difficulty_metric_name,
                    "easy_threshold": self.sequence_difficulty_easy_threshold,
                    "mid_threshold": self.sequence_difficulty_mid_threshold,
                }
            )
        common["mode"] = self.sequence_difficulty_mode
        common["overrides"] = copy.deepcopy(self.sequence_difficulty_profiles[profile_name])
        return common

    def resolve_semantic_scale_sequence_profile(self, sequence_name):
        if not self.sequence_difficulty_enabled:
            return None
        sequence_name = str(sequence_name or "").strip()
        if not sequence_name:
            raise ValueError(
                "A non-empty sequence name is required when semantic_scale.sequence_difficulty is enabled."
            )
        if sequence_name in self._sequence_difficulty_profile_cache:
            return copy.deepcopy(self._sequence_difficulty_profile_cache[sequence_name])

        self._validate_sequence_difficulty_thresholds()
        summary_payload = self._load_sequence_difficulty_summary()
        if summary_payload is None:
            self._sequence_difficulty_profile_cache[sequence_name] = None
            return None
        sequence_results = summary_payload["sequence_results"]
        matched_result = None
        for item in sequence_results:
            if str(item.get("sequence_name", "")).strip() == sequence_name:
                matched_result = item
                break
        if matched_result is None:
            profile = self._missing_sequence_difficulty_profile(sequence_name)
            profile["dataset_summary_path"] = self.sequence_difficulty_dataset_summary_path
            self._sequence_difficulty_profile_cache[sequence_name] = copy.deepcopy(profile)
            return profile
        profile = self.classify_sequence_difficulty_result(
            matched_result,
            include_overrides=True,
        )
        profile["dataset_summary_path"] = self.sequence_difficulty_dataset_summary_path
        self._sequence_difficulty_profile_cache[sequence_name] = copy.deepcopy(profile)
        return profile

    def activate_semantic_scale_sequence_profile(self, sequence_name):
        profile = self.resolve_semantic_scale_sequence_profile(sequence_name)
        if profile is None:
            self.clear_semantic_scale_sequence_profile()
            return None
        self._active_semantic_scale_override = copy.deepcopy(profile["overrides"])
        self._active_semantic_scale_profile = copy.deepcopy(profile)
        return copy.deepcopy(profile)

    def clear_semantic_scale_sequence_profile(self):
        self._active_semantic_scale_override = None
        self._active_semantic_scale_profile = None

    @property
    def active_semantic_scale_profile(self):
        if self._active_semantic_scale_profile is None:
            return None
        return copy.deepcopy(self._active_semantic_scale_profile)

    @property
    def vision_model_path(self):
        return self._resolve_template(self._get_section("vision").get("model_path"))

    @property
    def vision_model_type(self):
        return self._get_section("vision").get("model_type", "qwen")

    @property
    def api_key(self):
        vision_cfg = self._get_section("vision")
        api_key_env = str(vision_cfg.get("api_key_env", "") or "").strip()
        if api_key_env:
            return os.environ.get(api_key_env, "")

        api_key = vision_cfg.get("api_key", "")
        if isinstance(api_key, str):
            api_key = api_key.strip()
            if api_key.startswith("$") and len(api_key) > 1:
                return os.environ.get(api_key[1:], "")
        return api_key or ""

    @property
    def model_version(self):
        return self._get_section("vision").get("model_version", "")

    @property
    def max_video_frames(self):
        return self._get_section("vision").get("max_video_frames")

    @property
    def video_fps(self):
        return self._get_section("vision").get("video_fps")

    @property
    def torch_dtype(self):
        return self._get_section("vision").get("torch_dtype", "bfloat16")

    @property
    def max_new_tokens(self):
        return self._get_section("vision").get("max_new_tokens", 2048)

    @property
    def temperature(self):
        return self._get_section("vision").get("temperature", 0.2)

    @property
    def top_p(self):
        return self._get_section("vision").get("top_p")

    @property
    def num_beams(self):
        return self._get_section("vision").get("num_beams", 1)

    @property
    def reasoning_effort(self):
        value = self._get_section("vision").get("reasoning_effort")
        if value is None:
            return None
        value = str(value).strip().lower()
        return value or None

    @property
    def thinking_type(self):
        value = self._get_section("vision").get("thinking_type")
        if value is None:
            return None
        value = str(value).strip().lower()
        return value or None

    @property
    def thinking_budget_tokens(self):
        value = self._get_section("vision").get("thinking_budget_tokens")
        if value is None:
            return None
        return int(value)

    @property
    def data_path(self):
        return self._resolve_template(self._get_section("runtime").get("data_path"))

    @property
    def dataset(self):
        dataset = str(self._get_section("runtime").get("dataset", "") or "").strip()
        if not dataset:
            raise ValueError("Missing required config: runtime.dataset")
        return dataset

    @property
    def scene_domain(self):
        return str(self._get_section("runtime").get("scene_domain", "outdoor") or "outdoor").strip().lower()

    @property
    def depth_model(self):
        return self._get_section("runtime").get("depth_model", "pi3_long")

    @property
    def dust3r_model_name(self):
        return self._resolve_template(self._get_section("models").get("dust3r_model_name", "dust3r"))

    @property
    def must3r_model_name(self):
        return self._resolve_template(self._get_section("models").get("must3r_model_name", "must3r"))

    @property
    def must3r_retrieval_ckpt_path(self):
        value = self._get_section("models").get("must3r_retrieval_ckpt_path")
        if value in (None, "", "none", "None"):
            return None
        return self._resolve_template(value)

    @property
    def scal3r_project_root(self):
        return self._resolve_template(
            self._get_section("models").get("scal3r_project_root", "{project_root}/../Scal3R")
        )

    @property
    def scal3r_config_path(self):
        scal3r_project_root = self.scal3r_project_root
        return self._resolve_template(
            self._get_section("models").get(
                "scal3r_config_path",
                "{scal3r_project_root}/configs/models/scal3r.yaml",
            ),
            {"scal3r_project_root": scal3r_project_root},
        )

    @property
    def scal3r_checkpoint(self):
        value = self._get_section("models").get("scal3r_checkpoint")
        if value in (None, "", "none", "None"):
            return None
        return self._resolve_template(
            value,
            {"scal3r_project_root": self.scal3r_project_root},
        )

    @property
    def lingbot_map_project_root(self):
        return self._resolve_template(
            self._get_section("models").get("lingbot_map_project_root", "{project_root}/../lingbot-map")
        )

    @property
    def lingbot_map_checkpoint(self):
        value = self._get_section("models").get(
            "lingbot_map_checkpoint",
            "{lingbot_map_project_root}/checkpoints/lingbot_map.pt",
        )
        if value in (None, "", "none", "None"):
            return None
        return self._resolve_template(
            value,
            {"lingbot_map_project_root": self.lingbot_map_project_root},
        )

    @property
    def lingbot_map_mode(self):
        value = str(self._get_section("models").get("lingbot_map_mode", "streaming")).strip().lower()
        if value not in {"streaming", "windowed"}:
            raise ValueError(
                f"Unsupported models.lingbot_map_mode: {value}. "
                "Expected 'streaming' or 'windowed'."
            )
        return value

    @property
    def lingbot_map_image_size(self):
        return int(self._get_section("models").get("lingbot_map_image_size", 518))

    @property
    def lingbot_map_patch_size(self):
        return int(self._get_section("models").get("lingbot_map_patch_size", 14))

    @property
    def lingbot_map_num_scale_frames(self):
        return int(self._get_section("models").get("lingbot_map_num_scale_frames", 8))

    @property
    def lingbot_map_keyframe_interval(self):
        value = self._get_section("models").get("lingbot_map_keyframe_interval")
        if value in (None, "", "none", "None"):
            return None
        return int(value)

    @property
    def lingbot_map_max_frame_num(self):
        return int(self._get_section("models").get("lingbot_map_max_frame_num", 1024))

    @property
    def lingbot_map_kv_cache_sliding_window(self):
        return int(self._get_section("models").get("lingbot_map_kv_cache_sliding_window", 64))

    @property
    def lingbot_map_camera_num_iterations(self):
        return int(self._get_section("models").get("lingbot_map_camera_num_iterations", 4))

    @property
    def lingbot_map_use_sdpa(self):
        return bool(self._get_section("models").get("lingbot_map_use_sdpa", True))

    @property
    def lingbot_map_offload_to_cpu(self):
        return bool(self._get_section("models").get("lingbot_map_offload_to_cpu", True))

    @property
    def lingbot_map_window_size(self):
        return int(self._get_section("models").get("lingbot_map_window_size", 128))

    @property
    def lingbot_map_overlap_size(self):
        value = self._get_section("models").get("lingbot_map_overlap_size")
        if value in (None, "", "none", "None"):
            return self.long_overlap
        return int(value)

    @property
    def lingbot_map_overlap_keyframes(self):
        value = self._get_section("models").get("lingbot_map_overlap_keyframes")
        if value in (None, "", "none", "None"):
            return None
        return int(value)

    @property
    def vggt_omega_project_root(self):
        return self._resolve_template(
            self._get_section("models").get(
                "vggt_omega_project_root",
                "/data1/yanglei/FoundationModel/vggt-omega",
            )
        )

    @property
    def vggt_omega_checkpoint(self):
        value = self._get_section("models").get(
            "vggt_omega_checkpoint",
            "{vggt_omega_project_root}/checkpoints/vggt_omega_1b_512.pt",
        )
        if value in (None, "", "none", "None"):
            return None
        return self._resolve_template(
            value,
            {"vggt_omega_project_root": self.vggt_omega_project_root},
        )

    @property
    def vggt_omega_image_resolution(self):
        return int(self._get_section("models").get("vggt_omega_image_resolution", 512))

    @property
    def vggt_omega_preprocess_mode(self):
        value = str(self._get_section("models").get("vggt_omega_preprocess_mode", "resize_only")).strip().lower()
        if value not in {"resize_only", "balanced"}:
            raise ValueError(
                f"Unsupported models.vggt_omega_preprocess_mode: {value}. "
                "Expected 'resize_only' or 'balanced'."
            )
        return value

    @property
    def vggt_omega_geometry_intrinsics_source(self):
        value = str(self._get_section("models").get("vggt_omega_geometry_intrinsics_source", "predicted")).strip().lower()
        if value not in {"predicted", "dataset"}:
            raise ValueError("models.vggt_omega_geometry_intrinsics_source must be 'predicted' or 'dataset'.")
        return value

    @property
    def vggt_omega_enable_alignment(self):
        return bool(self._get_section("models").get("vggt_omega_enable_alignment", False))

    @property
    def dvlt_project_root(self):
        return self._resolve_template(
            self._get_section("models").get(
                "dvlt_project_root",
                "/data1/yanglei/FoundationModel/dvlt",
            )
        )

    @property
    def dvlt_checkpoint(self):
        value = self._get_section("models").get("dvlt_checkpoint", "{dvlt_project_root}")
        if value in (None, "", "none", "None"):
            return self.dvlt_project_root
        return self._resolve_template(
            value,
            {"dvlt_project_root": self.dvlt_project_root},
        )

    @property
    def fourrc_project_root(self):
        return self._resolve_template(
            self._get_section("models").get(
                "fourrc_project_root",
                "/home/yanglei/4RC",
            )
        )

    @property
    def fourrc_checkpoint(self):
        value = self._get_section("models").get(
            "fourrc_checkpoint",
            "/data1/yanglei/FoundationModel/4RC",
        )
        if value in (None, "", "none", "None"):
            return "/data1/yanglei/FoundationModel/4RC"
        return self._resolve_template(
            value,
            {"fourrc_project_root": self.fourrc_project_root},
        )

    @property
    def fourrc_image_size(self):
        return int(self._get_section("models").get("fourrc_image_size", 512))

    @property
    def fourrc_patch_size(self):
        return int(self._get_section("models").get("fourrc_patch_size", 14))

    @property
    def fourrc_dtype(self):
        value = str(self._get_section("models").get("fourrc_dtype", "bf16-mixed")).strip()
        valid_values = {"32", "16-mixed", "bf16-mixed", "bf16-mixed-no-grad-scaling"}
        if value not in valid_values:
            raise ValueError(
                f"Unsupported models.fourrc_dtype: {value}. "
                f"Expected one of: {sorted(valid_values)}."
            )
        return value

    @property
    def fourrc_use_center_as_anchor(self):
        return bool(self._get_section("models").get("fourrc_use_center_as_anchor", False))

    @property
    def fourrc_preprocess_mode(self):
        value = str(self._get_section("models").get("fourrc_preprocess_mode", "resize_only")).strip().lower()
        if value not in {"resize_only", "official"}:
            raise ValueError(
                f"Unsupported models.fourrc_preprocess_mode: {value}. "
                "Expected 'resize_only' or 'official'."
            )
        return value

    @property
    def stream3r_project_root(self):
        return self._resolve_template(
            self._get_section("models").get(
                "stream3r_project_root",
                "{project_root}/../STream3R",
            )
        )

    @property
    def stream3r_checkpoint(self):
        value = self._get_section("models").get("stream3r_checkpoint", "yslan/STream3R")
        if value in (None, "", "none", "None"):
            return "yslan/STream3R"
        return self._resolve_template(
            value,
            {"stream3r_project_root": self.stream3r_project_root},
        )

    @property
    def sam_config_path(self):
        return self.get_path("sam_config")

    @property
    def sam_checkpoint_path(self):
        return self.get_path("sam_checkpoint")

    @property
    def dataset_name(self):
        return os.path.basename(os.path.normpath(self.data_path))

    @property
    def use_shared_scale(self):
        return self._get_section("runtime").get("use_shared_scale", True)

    @property
    def all_sequences(self):
        runtime_cfg = self._get_section("runtime")
        return runtime_cfg.get("all_sequences", False) or not bool(self.sequence_names)

    @property
    def sequence_names(self):
        return self._normalize_sequence_names(self._get_section("runtime").get("sequence_name"))

    @property
    def sequence_name(self):
        sequence_names = self.sequence_names
        if not sequence_names:
            return ""
        if len(sequence_names) == 1:
            return sequence_names[0]
        return list(sequence_names)

    @property
    def frame_count(self):
        return self._get_section("runtime").get("frame_count", 50)

    @property
    def metric_depth_threshold(self):
        value = self._get_section("runtime").get("metric_depth_threshold", 100)
        if value in (None, "", "none", "None"):
            return None
        value = float(value)
        if value <= 0.0:
            raise ValueError(
                f"Unsupported runtime.metric_depth_threshold: {value}. "
                "Expected a positive float or null."
            )
        return value

    @property
    def enable_metric_depth_bins(self):
        return bool(self._get_section("runtime").get("enable_metric_depth_bins", False))

    @property
    def metric_depth_bins(self):
        raw_value = self._get_section("runtime").get("metric_depth_bins", [20, 40, 60])
        if raw_value in (None, "", [], ()):
            return []
        if not isinstance(raw_value, (list, tuple)):
            raise ValueError(
                "Unsupported runtime.metric_depth_bins. "
                "Expected a list of positive ascending distances."
            )
        bins = [float(item) for item in raw_value]
        if any(item <= 0.0 for item in bins):
            raise ValueError(
                f"Unsupported runtime.metric_depth_bins: {raw_value}. "
                "All bin edges must be positive."
            )
        if any(bins[idx] >= bins[idx + 1] for idx in range(len(bins) - 1)):
            raise ValueError(
                f"Unsupported runtime.metric_depth_bins: {raw_value}. "
                "Bin edges must be strictly ascending."
            )
        return bins

    @property
    def enable_sparse_reconstruction(self):
        return bool(self._get_section("runtime").get("enable_sparse_reconstruction", True))

    @property
    def enable_sparse_reconstruction_visualization(self):
        return self.enable_visualization

    @property
    def sparse_reconstruction_mode(self):
        mode = str(self._get_section("runtime").get("sparse_reconstruction_mode", "frame")).strip().lower()
        if mode not in {"frame", "sequence"}:
            raise ValueError(
                f"Unsupported runtime.sparse_reconstruction_mode: {mode}. "
                "Expected one of: frame, sequence."
            )
        return mode

    @property
    def sparse_reconstruction_alignment(self):
        alignment = str(
            self._get_section("runtime").get("sparse_reconstruction_alignment", "point_cloud")
        ).strip().lower()
        if alignment not in {"trajectory", "point_cloud"}:
            raise ValueError(
                f"Unsupported runtime.sparse_reconstruction_alignment: {alignment}. "
                "Expected one of: trajectory, point_cloud."
            )
        return alignment

    @property
    def sparse_reconstruction_pose_source(self):
        pose_source = str(
            self._get_section("runtime").get("sparse_reconstruction_pose_source", "pred")
        ).strip().lower()
        if pose_source not in {"pred", "gt"}:
            raise ValueError(
                f"Unsupported runtime.sparse_reconstruction_pose_source: {pose_source}. "
                "Expected one of: pred, gt."
            )
        return pose_source

    @property
    def sparse_reconstruction_min_points(self):
        value = int(self._get_section("runtime").get("sparse_reconstruction_min_points", 10))
        if value < 1:
            raise ValueError(
                f"Unsupported runtime.sparse_reconstruction_min_points: {value}. "
                "Expected a positive integer."
            )
        return value

    @property
    def enable_pose_evaluation(self):
        return bool(self._get_section("runtime").get("enable_pose_evaluation", False))

    @property
    def enable_pose_visualization(self):
        return self.enable_visualization

    @property
    def pose_evaluation_mode(self):
        mode = str(self._get_section("runtime").get("pose_evaluation_mode", "subsequence")).strip().lower()
        if mode not in {"subsequence", "sequence"}:
            raise ValueError(
                f"Unsupported runtime.pose_evaluation_mode: {mode}. "
                "Expected one of: subsequence, sequence."
            )
        return mode

    @property
    def pose_eval_min_frames(self):
        value = int(self._get_section("runtime").get("pose_eval_min_frames", 2))
        if value < 2:
            raise ValueError(
                f"Unsupported runtime.pose_eval_min_frames: {value}. "
                "Expected an integer >= 2."
            )
        return value

    @property
    def pose_eval_scale(self):
        value = float(self._get_section("runtime").get("pose_eval_scale", 1.0))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported runtime.pose_eval_scale: {value}. "
                "Expected a positive value."
            )
        return value

    @property
    def subsequence_size(self):
        return self._get_section("runtime").get("subsequence_size", 10)

    @property
    def parallel(self):
        return self._get_section("parallel").get("enable", True)

    @property
    def parallel_vlm_api(self):
        return self._get_section("parallel").get("vlm_api_enable", False)

    @property
    def max_vlm_workers(self):
        return self._get_section("parallel").get("max_vlm_workers", 4)

    @property
    def metric_workers(self):
        value = int(self._get_section("parallel").get("metric_workers", 0))
        if value < 0:
            raise ValueError(
                f"Unsupported parallel.metric_workers: {value}. "
                "Expected an integer >= 0."
            )
        return value

    @property
    def max_gpus(self):
        return self._get_section("parallel").get("max_gpus", 0)

    @property
    def long_chunk_size(self):
        return self._get_section("depth").get("long_chunk_size", 4)

    @property
    def depth_input_pixel_limit(self):
        value = self._get_section("depth").get("input_pixel_limit", 255000)
        if value in (None, "", "none", "None"):
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"Unsupported depth.input_pixel_limit: {value}. "
                "Expected a positive integer or null."
            )
        return value

    @property
    def depth_input_max_long_edge(self):
        value = self._get_section("depth").get("input_max_long_edge")
        if value in (None, "", "none", "None"):
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"Unsupported depth.input_max_long_edge: {value}. "
                "Expected a positive integer or null."
            )
        return value

    @property
    def depth_output_resolution(self):
        value = str(self._get_section("depth").get("output_resolution", "original")).strip().lower()
        if value not in {"original", "native"}:
            raise ValueError(
                f"Unsupported depth.output_resolution: {value}. "
                "Expected 'original' or 'native'."
            )
        return value

    @property
    def depth_native_sampling_mode(self):
        value = str(self._get_section("depth").get("native_sampling_mode", "nearest")).strip().lower()
        if value not in {"nearest", "bilinear"}:
            raise ValueError(
                f"Unsupported depth.native_sampling_mode: {value}. "
                "Expected 'nearest' or 'bilinear'."
            )
        return value

    @property
    def long_overlap(self):
        return self._get_section("depth").get("long_overlap", 3)

    @property
    def long_delete_temp_files(self):
        return self._get_section("depth").get("long_delete_temp_files", True)

    @property
    def long_temp_root(self):
        temp_root = self._resolve_template(self._get_section("depth").get("long_temp_root"))
        return temp_root or None

    @property
    def relative_height_mode(self):
        value = self._get_section("semantic_scale").get("relative_height_mode", "peak_to_bottom")
        value = str(value).strip().lower()
        if value not in {"peak_to_bottom", "peak_to_ground"}:
            raise ValueError(
                f"Unsupported semantic_scale.relative_height_mode: {value}. "
                "Expected 'peak_to_bottom' or 'peak_to_ground'."
            )
        return value

    @property
    def dynamic_anchor_height_scale(self):
        value = float(self._get_section("semantic_scale").get("dynamic_anchor_height_scale", 1.0))
        if not math.isfinite(value) or not 0.0 < value <= 3.0:
            raise ValueError("semantic_scale.dynamic_anchor_height_scale must be in (0, 3].")
        return value

    @property
    def shared_plane_mode(self):
        value = self._get_section("semantic_scale").get("shared_plane_mode", "global_plane")
        value = str(value).strip().lower()
        if value not in {"global_plane", "per_frame_plane"}:
            raise ValueError(
                f"Unsupported semantic_scale.shared_plane_mode: {value}. "
                "Expected 'global_plane' or 'per_frame_plane'."
            )
        return value

    @property
    def height_anchor_source(self):
        value = self._get_section("semantic_scale").get("height_anchor_source", "both")
        value = str(value).strip().lower()
        if value not in {"dynamic", "static", "both"}:
            raise ValueError(
                f"Unsupported semantic_scale.height_anchor_source: {value}. "
                "Expected 'dynamic', 'static', or 'both'."
            )
        return value

    @property
    def frame_parallel_workers(self):
        value = int(self._get_section("semantic_scale").get("frame_parallel_workers", 0))
        if value < 0:
            raise ValueError(
                f"Unsupported semantic_scale.frame_parallel_workers: {value}. "
                "Expected an integer >= 0."
            )
        return value

    @property
    def use_depth_confidence(self):
        return bool(self._get_section("semantic_scale").get("use_depth_confidence", True))

    @property
    def use_anchor_quality_filter(self):
        return bool(self._get_section("semantic_scale").get("use_anchor_quality_filter", True))

    @property
    def use_sam_anchor_bbox_intersection_filter(self):
        return bool(self._get_section("semantic_scale").get("use_sam_anchor_bbox_intersection_filter", False))

    @property
    def use_depth_geometry_filter(self):
        return bool(self._get_section("semantic_scale").get("use_depth_geometry_filter", True))

    @property
    def save_scale_intermediate_results(self):
        return bool(self._get_section("semantic_scale").get("save_scale_intermediate_results", False))

    @property
    def depth_geometry_min_anchor_points(self):
        value = int(self._get_section("semantic_scale").get("depth_geometry_min_anchor_points", 32))
        if value < 1:
            raise ValueError(
                f"Unsupported semantic_scale.depth_geometry_min_anchor_points: {value}. "
                "Expected an integer >= 1."
            )
        return value

    @property
    def depth_geometry_min_relative_height(self):
        value = float(self._get_section("semantic_scale").get("depth_geometry_min_relative_height", 1e-4))
        if value < 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.depth_geometry_min_relative_height: {value}. "
                "Expected a float >= 0.0."
            )
        return value

    @property
    def use_vlm_anchor_confidence_filter(self):
        return bool(self._get_section("semantic_scale").get("use_vlm_anchor_confidence_filter", True))

    @property
    def vlm_dynamic_anchor_confidence_threshold(self):
        value = float(self._get_section("semantic_scale").get("vlm_dynamic_anchor_confidence_threshold", 0.7))
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.vlm_dynamic_anchor_confidence_threshold: {value}. "
                "Expected a float in [0.0, 1.0]."
            )
        return value

    @property
    def vlm_static_anchor_confidence_threshold(self):
        value = float(self._get_section("semantic_scale").get("vlm_static_anchor_confidence_threshold", 0.6))
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.vlm_static_anchor_confidence_threshold: {value}. "
                "Expected a float in [0.0, 1.0]."
            )
        return value

    @property
    def anchor_quality_min_percentile(self):
        value = float(self._get_section("semantic_scale").get("anchor_quality_min_percentile", 0.3))
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.anchor_quality_min_percentile: {value}. "
                "Expected a float in [0.0, 1.0)."
            )
        return value

    @property
    def anchor_quality_max_percentile(self):
        value = float(self._get_section("semantic_scale").get("anchor_quality_max_percentile", 0.7))
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.anchor_quality_max_percentile: {value}. "
                "Expected a float in [0.0, 1.0)."
            )
        return value

    @property
    def plane_conf_percentile(self):
        value = float(self._get_section("semantic_scale").get("plane_conf_percentile", 0.3))
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.plane_conf_percentile: {value}. "
                "Expected a float in [0.0, 1.0)."
            )
        return value

    @property
    def plane_normal_outlier_filter_enabled(self):
        return bool(self._get_section("semantic_scale").get("plane_normal_outlier_filter_enabled", True))

    @property
    def plane_normal_filter_min_frames(self):
        value = int(self._get_section("semantic_scale").get("plane_normal_filter_min_frames", 3))
        if value < 1:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_min_frames: {value}. "
                "Expected an integer >= 1."
            )
        return value

    @property
    def plane_normal_filter_max_angle_deg(self):
        value = float(self._get_section("semantic_scale").get("plane_normal_filter_max_angle_deg", 25.0))
        if not 0.0 < value <= 90.0:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_max_angle_deg: {value}. "
                "Expected a float in (0.0, 90.0]."
            )
        return value

    @property
    def plane_normal_filter_max_points(self):
        value = int(self._get_section("semantic_scale").get("plane_normal_filter_max_points", 1000))
        if value < 3:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_max_points: {value}. "
                "Expected an integer >= 3."
            )
        return value

    @property
    def plane_normal_filter_ransac_iterations(self):
        value = int(self._get_section("semantic_scale").get("plane_normal_filter_ransac_iterations", 30))
        if value < 1:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_ransac_iterations: {value}. "
                "Expected an integer >= 1."
            )
        return value

    @property
    def plane_normal_filter_ransac_threshold(self):
        value = float(self._get_section("semantic_scale").get("plane_normal_filter_ransac_threshold", 0.05))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_ransac_threshold: {value}. "
                "Expected a float > 0.0."
            )
        return value

    @property
    def plane_normal_filter_ransac_confidence(self):
        value = float(self._get_section("semantic_scale").get("plane_normal_filter_ransac_confidence", 0.95))
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.plane_normal_filter_ransac_confidence: {value}. "
                "Expected a float in (0.0, 1.0)."
            )
        return value

    @property
    def optimized_scale_priority_modulation_strength(self):
        value = float(self._get_section("semantic_scale").get("optimized_scale_priority_modulation_strength", 0.3))
        if not 0.0 <= value < 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.optimized_scale_priority_modulation_strength: {value}. "
                "Expected a float in [0.0, 1.0)."
            )
        return value

    @property
    def distance_weight_near(self):
        value = float(self._get_section("semantic_scale").get("distance_weight_near", 1.0))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.distance_weight_near: {value}. "
                "Expected a float > 0.0."
            )
        return value

    @property
    def distance_weight_middle(self):
        value = float(self._get_section("semantic_scale").get("distance_weight_middle", 0.85))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.distance_weight_middle: {value}. "
                "Expected a float > 0.0."
            )
        return value

    @property
    def distance_weight_distant(self):
        value = float(self._get_section("semantic_scale").get("distance_weight_distant", 0.65))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.distance_weight_distant: {value}. "
                "Expected a float > 0.0."
            )
        return value

    @property
    def optimizer_dynamic_classes(self):
        raw_value = self._get_section("semantic_scale").get("optimizer_dynamic_classes", [])
        return self._normalize_string_list(raw_value)

    @property
    def optimizer_static_classes(self):
        raw_value = self._get_section("semantic_scale").get("optimizer_static_classes", [])
        return self._normalize_string_list(raw_value)

    @property
    def scale_refinement_solver(self):
        section = self._get_section("semantic_scale")
        raw_value = section.get("scale_refinement_solver", section.get("optimized_scale_solver"))
        if raw_value is not None:
            value = str(raw_value or "").strip().lower()
            if value not in {"legacy", "original", "weighted"}:
                raise ValueError(
                    f"Unsupported semantic_scale.scale_refinement_solver: {value}. "
                    "Expected one of: legacy, original, weighted."
                )
            return value

        default_variant = self.default_optimized_variant
        variant_to_solver = {
            "optimized_scale_legacy": "legacy",
            "optimized_scale_original": "original",
            "optimized_scale_weighted_bi": "weighted",
            "optimized_scale_weighted_ifpa": "weighted",
        }
        if default_variant in variant_to_solver:
            return variant_to_solver[default_variant]
        raw_value = "weighted"
        value = str(raw_value or "").strip().lower()
        if value not in {"legacy", "original", "weighted"}:
            raise ValueError(
                f"Unsupported semantic_scale.scale_refinement_solver: {value}. "
                "Expected one of: legacy, original, weighted."
            )
        return value

    @property
    def optimized_scale_solver(self):
        return self.scale_refinement_solver

    @property
    def weighted_search_mode(self):
        section = self._get_section("semantic_scale")
        raw_value = section.get("weighted_search_mode", "consensus_bounded")
        value = str(raw_value or "").strip().lower()
        if value not in {"global", "consensus_bounded"}:
            raise ValueError(
                f"Unsupported semantic_scale.weighted_search_mode: {value}. "
                "Expected one of: global, consensus_bounded."
            )
        return value

    @property
    def consensus_prior_mode(self):
        section = self._get_section("semantic_scale")
        raw_value = section.get("consensus_prior_mode", "interval")
        value = str(raw_value or "").strip().lower()
        if value not in {"none", "point", "interval"}:
            raise ValueError(
                f"Unsupported semantic_scale.consensus_prior_mode: {value}. "
                "Expected one of: none, point, interval."
            )
        return value

    @property
    def consensus_prior_strength(self):
        value = float(self._get_section("semantic_scale").get("consensus_prior_strength", 0.1))
        if value < 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.consensus_prior_strength: {value}. "
                "Expected a float >= 0.0."
            )
        return value

    @property
    def consensus_prior_interval_inside_ratio(self):
        value = float(self._get_section("semantic_scale").get("consensus_prior_interval_inside_ratio", 0.2))
        if value < 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.consensus_prior_interval_inside_ratio: {value}. "
                "Expected a float >= 0.0."
            )
        return value

    @property
    def consensus_interval_solver(self):
        section = self._get_section("semantic_scale")
        raw_value = section.get("consensus_interval_solver", section.get("consensus_solver"))
        if raw_value is not None:
            value = str(raw_value or "").strip().lower()
            if value not in {"legacy", "bi", "ifpa"}:
                raise ValueError(
                    f"Unsupported semantic_scale.consensus_interval_solver: {value}. "
                    "Expected one of: legacy, bi, ifpa."
                )
            return value

        default_variant = self.default_optimized_variant
        variant_to_solver = {
            "optimized_scale_legacy": "legacy",
            "optimized_scale_original": "legacy",
            "optimized_scale_weighted_bi": "bi",
            "optimized_scale_weighted_ifpa": "ifpa",
        }
        if default_variant in variant_to_solver:
            return variant_to_solver[default_variant]
        raw_value = "bi"
        value = str(raw_value or "").strip().lower()
        if value not in {"legacy", "bi", "ifpa"}:
            raise ValueError(
                f"Unsupported semantic_scale.consensus_interval_solver: {value}. "
                "Expected one of: legacy, bi, ifpa."
            )
        return value

    @property
    def consensus_solver(self):
        return self.consensus_interval_solver

    @property
    def default_optimized_variant(self):
        section = self._get_section("semantic_scale")
        scale_solver_override = section.get("scale_refinement_solver", section.get("optimized_scale_solver"))
        consensus_solver_override = section.get("consensus_interval_solver", section.get("consensus_solver"))

        if scale_solver_override is not None or consensus_solver_override is not None:
            solver_name = str(scale_solver_override or "weighted").strip().lower()
            consensus_solver_name = str(consensus_solver_override or "bi").strip().lower()
            if solver_name == "legacy":
                value = "optimized_scale_legacy"
            elif solver_name == "original":
                value = "optimized_scale_original"
            elif consensus_solver_name == "ifpa":
                value = "optimized_scale_weighted_ifpa"
            else:
                value = "optimized_scale_weighted_bi"
        else:
            raw_value = section.get("default_optimized_variant")
            if raw_value is not None:
                value = str(raw_value or "").strip().lower()
            else:
                value = "optimized_scale_weighted_bi"

        valid_values = {
            "optimized_scale_legacy",
            "legacy",
            "optimized_scale_original",
            "original",
            "optimized_scale_weighted_bi",
            "weighted_bi",
            "optimized_scale_weighted_ifpa",
            "weighted_ifpa",
        }
        if value not in valid_values:
            raise ValueError(
                f"Unsupported semantic_scale.default_optimized_variant: {value}. "
                "Expected one of: optimized_scale_legacy, optimized_scale_original, "
                "optimized_scale_weighted_bi, optimized_scale_weighted_ifpa."
            )
        canonical_mapping = {
            "legacy": "optimized_scale_legacy",
            "original": "optimized_scale_original",
            "weighted_bi": "optimized_scale_weighted_bi",
            "weighted_ifpa": "optimized_scale_weighted_ifpa",
        }
        return canonical_mapping.get(value, value)

    @property
    def enable_optimized_scale_legacy(self):
        return bool(self._get_section("semantic_scale").get("enable_optimized_scale_legacy", False))

    @property
    def enable_optimized_scale_original(self):
        return bool(self._get_section("semantic_scale").get("enable_optimized_scale_original", True))

    @property
    def enable_optimized_scale_weighted_bi(self):
        return bool(self._get_section("semantic_scale").get("enable_optimized_scale_weighted_bi", False))

    @property
    def enable_optimized_scale_weighted_ifpa(self):
        return bool(self._get_section("semantic_scale").get("enable_optimized_scale_weighted_ifpa", False))

    @property
    def enable_selected_scale(self):
        return bool(self._get_section("semantic_scale").get("enable_selected_scale", False))

    @property
    def selected_scale_default(self):
        raw_value = self._get_section("semantic_scale").get("selected_scale_default", "consistent_scale")
        value = self._normalize_metric_variant_name(raw_value)
        valid_values = {
            "relative",
            "consistent",
            "optimized_legacy",
            "optimized_original",
            "optimized_weighted_bi",
            "optimized_weighted_ifpa",
        }
        if value not in valid_values:
            raise ValueError(
                f"Unsupported semantic_scale.selected_scale_default: {raw_value}. "
                "Expected one of: relative_scale, consistent_scale, optimized_scale_legacy, "
                "optimized_scale_original, optimized_scale_weighted_bi, optimized_scale_weighted_ifpa."
            )
        enabled_defaults = {"relative", "consistent"}
        if self.enable_optimized_scale_legacy:
            enabled_defaults.add("optimized_legacy")
        if self.enable_optimized_scale_original:
            enabled_defaults.add("optimized_original")
        if self.enable_optimized_scale_weighted_bi:
            enabled_defaults.add("optimized_weighted_bi")
        if self.enable_optimized_scale_weighted_ifpa:
            enabled_defaults.add("optimized_weighted_ifpa")
        if self.enable_selected_scale and value not in enabled_defaults:
            raise ValueError(
                f"semantic_scale.selected_scale_default={raw_value} is disabled. "
                "Enable that scale variant or choose an enabled selected_scale_default."
            )
        return value

    @property
    def max_optimized_shift_ratio(self):
        value = float(self._get_section("semantic_scale").get("max_optimized_shift_ratio", 0.03))
        if value < 0.0:
            raise ValueError("semantic_scale.max_optimized_shift_ratio must be >= 0.0.")
        return value

    @property
    def min_optimized_to_consistent_ratio(self):
        raw_value = self._get_section("semantic_scale").get("min_optimized_to_consistent_ratio")
        if raw_value in (None, "", "none", "None"):
            return None
        value = float(raw_value)
        if not 0.0 < value <= 1.0:
            raise ValueError("semantic_scale.min_optimized_to_consistent_ratio must be in (0, 1].")
        return value

    @property
    def min_valid_intervals_for_optimized(self):
        raw_value = self._get_section("semantic_scale").get("min_valid_intervals_for_optimized")
        if raw_value in (None, "", "none", "None"):
            return None
        value = int(raw_value)
        if value < 0:
            raise ValueError("semantic_scale.min_valid_intervals_for_optimized must be >= 0.")
        return value

    @property
    def min_consistent_count_for_optimized(self):
        raw_value = self._get_section("semantic_scale").get("min_consistent_count_for_optimized")
        if raw_value in (None, "", "none", "None"):
            return None
        value = int(raw_value)
        if value < 0:
            raise ValueError("semantic_scale.min_consistent_count_for_optimized must be >= 0.")
        return value

    @property
    def enabled_optimized_variants(self):
        variants = []
        if self.enable_optimized_scale_legacy:
            variants.append("optimized_scale_legacy")
        if self.enable_optimized_scale_original:
            variants.append("optimized_scale_original")
        if self.enable_optimized_scale_weighted_bi:
            variants.append("optimized_scale_weighted_bi")
        if self.enable_optimized_scale_weighted_ifpa:
            variants.append("optimized_scale_weighted_ifpa")

        default_variant = self.default_optimized_variant
        if default_variant not in variants:
            raise ValueError(
                f"semantic_scale.default_optimized_variant={default_variant} is disabled. "
                "Enable that variant or choose an enabled default_optimized_variant."
            )
        return tuple(variants)

    @property
    def enabled_metric_variants(self):
        variants = ["relative", "consistent"]
        if self.enable_optimized_scale_legacy:
            variants.append("optimized_legacy")
        if self.enable_optimized_scale_original:
            variants.append("optimized_original")
        if self.enable_optimized_scale_weighted_bi:
            variants.append("optimized_weighted_bi")
        if self.enable_optimized_scale_weighted_ifpa:
            variants.append("optimized_weighted_ifpa")
        if self.enable_selected_scale:
            variants.append("selected")
        return tuple(variants)

    @property
    def ifpa_num_bins(self):
        value = int(self._get_section("semantic_scale").get("ifpa_num_bins", 128))
        if value < 3:
            raise ValueError(
                f"Unsupported semantic_scale.ifpa_num_bins: {value}. "
                "Expected an integer >= 3."
            )
        return value

    @property
    def ifpa_use_weight(self):
        return bool(self._get_section("semantic_scale").get("ifpa_use_weight", True))

    @property
    def bi_keep_ratio(self):
        value = float(self._get_section("semantic_scale").get("bi_keep_ratio", 0.6))
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"Unsupported semantic_scale.bi_keep_ratio: {value}. "
                "Expected a float in (0.0, 1.0]."
            )
        return value

    @property
    def bi_min_segment_width(self):
        value = float(self._get_section("semantic_scale").get("bi_min_segment_width", 1e-6))
        if value <= 0.0:
            raise ValueError(
                f"Unsupported semantic_scale.bi_min_segment_width: {value}. "
                "Expected a float > 0.0."
            )
        return value

    @property
    def enable_visualization(self):
        return bool(self._get_section("runtime").get("enable_visualization", True))

    @property
    def cleanup_depth_outputs_after_metrics(self):
        return bool(self._get_section("runtime").get("cleanup_depth_outputs_after_metrics", False))

    @property
    def pi3_weights(self):
        return self._resolve_template(self._get_section("models").get("pi3_weights"))

    @property
    def vggt_weights(self):
        return self._resolve_template(self._get_section("models").get("vggt_weights"))

    @property
    def da3_model_name(self):
        return self._resolve_template(self._get_section("models").get("da3_model_name"))

    @property
    def moge_model_name(self):
        return self._resolve_template(self._get_section("models").get("moge_model_name"))

    def _active_depth_model_summary_lines(self) -> list[str]:
        model = str(self.depth_model)
        if model == "pi3_long":
            return [f"pi3_weights: {self.pi3_weights}"]
        if model == "vggt_long":
            return [f"vggt_weights: {self.vggt_weights}"]
        if model == "vggt_omega_long":
            return [
                f"vggt_omega_project_root: {self.vggt_omega_project_root}",
                f"vggt_omega_checkpoint: {self.vggt_omega_checkpoint}",
                f"vggt_omega_image_resolution: {self.vggt_omega_image_resolution}",
                f"vggt_omega_preprocess_mode: {self.vggt_omega_preprocess_mode}",
                f"vggt_omega_geometry_intrinsics_source: {self.vggt_omega_geometry_intrinsics_source}",
                f"vggt_omega_enable_alignment: {self.vggt_omega_enable_alignment}",
            ]
        if model == "dvlt_long":
            return [
                f"dvlt_project_root: {self.dvlt_project_root}",
                f"dvlt_checkpoint: {self.dvlt_checkpoint}",
            ]
        if model == "4rc_long":
            return [
                f"fourrc_project_root: {self.fourrc_project_root}",
                f"fourrc_checkpoint: {self.fourrc_checkpoint}",
                f"fourrc_image_size: {self.fourrc_image_size}",
                f"fourrc_patch_size: {self.fourrc_patch_size}",
                f"fourrc_dtype: {self.fourrc_dtype}",
                f"fourrc_use_center_as_anchor: {self.fourrc_use_center_as_anchor}",
                f"fourrc_preprocess_mode: {self.fourrc_preprocess_mode}",
            ]
        if model == "da3_long":
            return [f"da3_model_name: {self.da3_model_name}"]
        if model == "moge":
            return [f"moge_model_name: {self.moge_model_name}"]
        if model == "dust3r_long":
            return [f"dust3r_model_name: {self.dust3r_model_name}"]
        if model == "must3r_long":
            return [
                f"must3r_model_name: {self.must3r_model_name}",
                f"must3r_retrieval_ckpt_path: {self.must3r_retrieval_ckpt_path}",
            ]
        if model == "scal3r_long":
            return [
                f"scal3r_project_root: {self.scal3r_project_root}",
                f"scal3r_config_path: {self.scal3r_config_path}",
                f"scal3r_checkpoint: {self.scal3r_checkpoint}",
            ]
        if model == "lingbot_map_long":
            return [
                f"lingbot_map_project_root: {self.lingbot_map_project_root}",
                f"lingbot_map_checkpoint: {self.lingbot_map_checkpoint}",
                f"lingbot_map_mode: {self.lingbot_map_mode}",
                f"lingbot_map_image_size: {self.lingbot_map_image_size}",
                f"lingbot_map_patch_size: {self.lingbot_map_patch_size}",
                f"lingbot_map_num_scale_frames: {self.lingbot_map_num_scale_frames}",
                f"lingbot_map_keyframe_interval: {self.lingbot_map_keyframe_interval}",
                f"lingbot_map_max_frame_num: {self.lingbot_map_max_frame_num}",
                f"lingbot_map_kv_cache_sliding_window: {self.lingbot_map_kv_cache_sliding_window}",
                f"lingbot_map_camera_num_iterations: {self.lingbot_map_camera_num_iterations}",
                f"lingbot_map_use_sdpa: {self.lingbot_map_use_sdpa}",
                f"lingbot_map_offload_to_cpu: {self.lingbot_map_offload_to_cpu}",
                f"lingbot_map_window_size: {self.lingbot_map_window_size}",
                f"lingbot_map_overlap_size: {self.lingbot_map_overlap_size}",
                f"lingbot_map_overlap_keyframes: {self.lingbot_map_overlap_keyframes}",
            ]
        if model == "stream3r_long":
            return [
                f"stream3r_project_root: {self.stream3r_project_root}",
                f"stream3r_checkpoint: {self.stream3r_checkpoint}",
            ]
        return []

    def summary_lines(self):
        lines = [
            f"config: {os.path.abspath(self.args.config)}",
            f"data_path: {self.data_path}",
            f"dataset: {self.dataset}",
            f"depth_model: {self.depth_model}",
            f"sequence_name: {self.sequence_name or '<all>'}",
            f"sequence_names: {list(self.sequence_names)}",
            f"frame_count: {self.frame_count}",
            f"metric_depth_threshold: {self.metric_depth_threshold}",
            f"enable_metric_depth_bins: {self.enable_metric_depth_bins}",
            f"metric_depth_bins: {self.metric_depth_bins}",
            f"enable_sparse_reconstruction: {self.enable_sparse_reconstruction}",
            f"sparse_reconstruction_mode: {self.sparse_reconstruction_mode}",
            f"sparse_reconstruction_alignment: {self.sparse_reconstruction_alignment}",
            f"sparse_reconstruction_pose_source: {self.sparse_reconstruction_pose_source}",
            f"sparse_reconstruction_min_points: {self.sparse_reconstruction_min_points}",
            f"enable_pose_evaluation: {self.enable_pose_evaluation}",
            f"pose_evaluation_mode: {self.pose_evaluation_mode}",
            f"pose_eval_min_frames: {self.pose_eval_min_frames}",
            f"pose_eval_scale: {self.pose_eval_scale}",
            f"subsequence_size: {self.subsequence_size}",
            f"enable_visualization: {self.enable_visualization}",
            f"cleanup_depth_outputs_after_metrics: {self.cleanup_depth_outputs_after_metrics}",
            f"parallel: {self.parallel}",
            f"max_gpus: {self.max_gpus}",
            f"parallel_vlm_api: {self.parallel_vlm_api}",
            f"max_vlm_workers: {self.max_vlm_workers}",
            f"metric_workers: {self.metric_workers}",
            f"relative_height_mode: {self.relative_height_mode}",
            f"shared_plane_mode: {self.shared_plane_mode}",
            f"height_anchor_source: {self.height_anchor_source}",
            f"frame_parallel_workers: {self.frame_parallel_workers}",
            f"use_depth_confidence: {self.use_depth_confidence}",
            f"use_anchor_quality_filter: {self.use_anchor_quality_filter}",
            f"use_sam_anchor_bbox_intersection_filter: {self.use_sam_anchor_bbox_intersection_filter}",
            f"use_depth_geometry_filter: {self.use_depth_geometry_filter}",
            f"save_scale_intermediate_results: {self.save_scale_intermediate_results}",
            f"depth_geometry_min_anchor_points: {self.depth_geometry_min_anchor_points}",
            f"depth_geometry_min_relative_height: {self.depth_geometry_min_relative_height}",
            f"use_vlm_anchor_confidence_filter: {self.use_vlm_anchor_confidence_filter}",
            f"vlm_dynamic_anchor_confidence_threshold: {self.vlm_dynamic_anchor_confidence_threshold}",
            f"vlm_static_anchor_confidence_threshold: {self.vlm_static_anchor_confidence_threshold}",
            f"anchor_quality_min_percentile: {self.anchor_quality_min_percentile}",
            f"anchor_quality_max_percentile: {self.anchor_quality_max_percentile}",
            f"plane_conf_percentile: {self.plane_conf_percentile}",
            f"plane_normal_outlier_filter_enabled: {self.plane_normal_outlier_filter_enabled}",
            f"plane_normal_filter_min_frames: {self.plane_normal_filter_min_frames}",
            f"plane_normal_filter_max_angle_deg: {self.plane_normal_filter_max_angle_deg}",
            f"plane_normal_filter_max_points: {self.plane_normal_filter_max_points}",
            f"plane_normal_filter_ransac_iterations: {self.plane_normal_filter_ransac_iterations}",
            f"plane_normal_filter_ransac_threshold: {self.plane_normal_filter_ransac_threshold}",
            f"plane_normal_filter_ransac_confidence: {self.plane_normal_filter_ransac_confidence}",
            f"optimized_scale_priority_modulation_strength: {self.optimized_scale_priority_modulation_strength}",
            f"distance_weight_near: {self.distance_weight_near}",
            f"distance_weight_middle: {self.distance_weight_middle}",
            f"distance_weight_distant: {self.distance_weight_distant}",
            f"optimizer_dynamic_classes: {list(self.optimizer_dynamic_classes)}",
            f"optimizer_static_classes: {list(self.optimizer_static_classes)}",
            f"default_optimized_variant: {self.default_optimized_variant}",
            f"enable_optimized_scale_legacy: {self.enable_optimized_scale_legacy}",
            f"enable_optimized_scale_original: {self.enable_optimized_scale_original}",
            f"enable_optimized_scale_weighted_bi: {self.enable_optimized_scale_weighted_bi}",
            f"enable_optimized_scale_weighted_ifpa: {self.enable_optimized_scale_weighted_ifpa}",
            f"enable_selected_scale: {self.enable_selected_scale}",
            f"selected_scale_default: {self.selected_scale_default}",
            f"min_valid_intervals_for_optimized: {self.min_valid_intervals_for_optimized}",
            f"min_consistent_count_for_optimized: {self.min_consistent_count_for_optimized}",
            f"max_optimized_shift_ratio: {self.max_optimized_shift_ratio}",
            f"min_optimized_to_consistent_ratio: {self.min_optimized_to_consistent_ratio}",
            f"scale_refinement_solver: {self.scale_refinement_solver}",
            f"weighted_search_mode: {self.weighted_search_mode}",
            f"consensus_prior_mode: {self.consensus_prior_mode}",
            f"consensus_prior_strength: {self.consensus_prior_strength}",
            f"consensus_prior_interval_inside_ratio: {self.consensus_prior_interval_inside_ratio}",
            f"consensus_interval_solver: {self.consensus_interval_solver}",
            f"ifpa_num_bins: {self.ifpa_num_bins}",
            f"ifpa_use_weight: {self.ifpa_use_weight}",
            f"bi_keep_ratio: {self.bi_keep_ratio}",
            f"bi_min_segment_width: {self.bi_min_segment_width}",
            f"sequence_difficulty_enabled: {self.sequence_difficulty_enabled}",
            f"sequence_difficulty_mode: {self.sequence_difficulty_mode}",
            f"sequence_difficulty_dataset_summary_path: {self.sequence_difficulty_dataset_summary_path}",
        ]
        if self.sequence_difficulty_mode == "best_variant_vs_baseline":
            lines.extend(
                [
                    f"sequence_difficulty_metric_scope: {self.sequence_difficulty_metric_scope}",
                    f"sequence_difficulty_distance_bin: {self.sequence_difficulty_distance_bin}",
                    f"sequence_difficulty_baseline_variant: {self.sequence_difficulty_baseline_variant}",
                    f"sequence_difficulty_candidate_variants: {list(self.sequence_difficulty_candidate_variants)}",
                    f"sequence_difficulty_easy_d1_gain_threshold: {self.sequence_difficulty_easy_d1_gain_threshold}",
                ]
            )
        else:
            lines.extend(
                [
                    f"sequence_difficulty_metric_variant: {self.sequence_difficulty_metric_variant}",
                    f"sequence_difficulty_metric_scope: {self.sequence_difficulty_metric_scope}",
                    f"sequence_difficulty_metric_name: {self.sequence_difficulty_metric_name}",
                    f"sequence_difficulty_distance_bin: {self.sequence_difficulty_distance_bin}",
                    f"sequence_difficulty_metric_goal: {self.sequence_difficulty_metric_goal}",
                    f"sequence_difficulty_easy_threshold: {self.sequence_difficulty_easy_threshold}",
                    f"sequence_difficulty_mid_threshold: {self.sequence_difficulty_mid_threshold}",
                    f"sequence_difficulty_hard_threshold: {self.sequence_difficulty_hard_threshold}",
                ]
            )
        lines.extend(
            [
                f"sequence_difficulty_profiles: {self.sequence_difficulty_profiles}",
                f"enabled_optimized_variants: {self.enabled_optimized_variants}",
                f"vision_model_type: {self.vision_model_type}",
                f"vision_model_path: {self.vision_model_path}",
                f"reasoning_effort: {self.reasoning_effort}",
                f"thinking_type: {self.thinking_type}",
                f"thinking_budget_tokens: {self.thinking_budget_tokens}",
                f"sam_config: {self.sam_config_path}",
                f"sam_checkpoint: {self.sam_checkpoint_path}",
                f"depth_input_pixel_limit: {self.depth_input_pixel_limit}",
                f"depth_input_max_long_edge: {self.depth_input_max_long_edge}",
                f"depth_output_resolution: {self.depth_output_resolution}",
                f"depth_native_sampling_mode: {self.depth_native_sampling_mode}",
                *self._active_depth_model_summary_lines(),
                f"output_root: {self.get_path('output_root')}",
            ]
        )
        return lines

    def print_summary(self):
        print("\n=== Configuration Summary ===")
        for line in self.summary_lines():
            print(line)

    def get_path(self, path_name, *args):
        paths_cfg = self._get_section("paths")
        template = paths_cfg.get(path_name)
        if template is None:
            return None

        extra_context = {}
        if args:
            if len(args) == 1:
                extra_context["sequence_name"] = args[0]
            else:
                extra_context["args"] = args
        return self._resolve_template(template, extra_context=extra_context)

    def ensure_dirs(self, sequence_name):
        dirs_to_create = [
            "vlm_output",
            "vlm_results",
            "vlm_visualization",
            "sam_output",
            "sam_visualization",
            "da3_output",
            "pi3_output",
            "vggt_output",
            "dust3r_output",
            "must3r_output",
            "scal3r_output",
            "lingbot_map_output",
            "vggt_omega_output",
            "dvlt_output",
            "fourrc_output",
            "stream3r_output",
        ]

        for dir_name in dirs_to_create:
            dir_path = self.get_path(dir_name, sequence_name)
            if dir_path:
                ensure_dir(dir_path)


def ensure_dir(dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


class _SafeFormatDict(dict):
    def __init__(self, data, resolver):
        super().__init__(data)
        self._resolver = resolver

    def __missing__(self, key):
        value = self._resolver(key)
        self[key] = value
        return value


def _looks_like_path(value):
    return "/" in value or value.startswith(".")


config = Config()
