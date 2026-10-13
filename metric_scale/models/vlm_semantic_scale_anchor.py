import os
import re
import json
import importlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image

from metric_scale.config.config import config


class VLMSemanticScaleAnchor:
    """
    VLM-based semantic scale anchor extraction class.

    Key design:
    - VLM outputs bbox in normalized coordinates [0,1000] to avoid resolution dependence.
    - Strong anti-hallucination constraints to reduce over-completion (e.g., many fake cars).
    - Post-process robustly (json extraction, bbox sanity check, NMS, topK).
    """

    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.outdoor_dynamic_anchor_classes = {
            "car",
            "sedan",
            "minibus",
            "truck",
            "bus",
            "van",
            "heavy truck",
            "suv",
            "pickup truck",
            "cyclist",
            "motorcycle",
            "pedestrian",
            "person",
        }
        self.outdoor_static_anchor_classes = {
            "traffic barrier",
            "concrete barrier",
            "jersey barrier",
            "median barrier",
            "roadside barrier",
            "guardrail",
            "steel guardrail",
            "curbstone",
            "bollard",
            "delineator post",
            "warning post",
            "traffic cone",
        }
        self.indoor_dynamic_anchor_classes = {
            "pedestrian",
            "person",
            "cyclist",
        }
        self.indoor_static_anchor_classes = {
            "door",
            "counter",
            "cabinet",
            "drawer",
            "dresser",
            "table",
            "workbench",
            "toilet",
            "chair",
            "stool",
            "hamper",
            "trash can",
            "caution sign",
            "toaster",
            "box",
            "bottle",
            "cup",
            "book",
            "bowl",
            "brick",
        }

    @staticmethod
    def _normalize_dataset_module_name(dataset_name: str) -> str:
        return str(dataset_name).strip().lower().replace("-", "_")

    def _load_prompt_module(self, dataset_name: Optional[str] = None):
        normalized_dataset = self._normalize_dataset_module_name(dataset_name or config.dataset)
        module_candidates = [
            f"metric_scale.prompts.{normalized_dataset}",
            "metric_scale.prompts.default",
        ]

        for module_name in module_candidates:
            try:
                return importlib.import_module(module_name)
            except ModuleNotFoundError:
                continue

        raise ModuleNotFoundError(
            f"No prompt module found for dataset '{dataset_name or config.dataset}' and no default fallback available."
        )

    def _get_prompt(self, task_name: str, dataset_name: Optional[str] = None) -> str:
        prompt_module = self._load_prompt_module(dataset_name)
        if not hasattr(prompt_module, "get_prompt"):
            raise AttributeError(f"Prompt module {prompt_module.__name__} must define get_prompt(task_name)")
        try:
            return prompt_module.get_prompt(task_name, scene_domain=config.scene_domain)
        except TypeError:
            return prompt_module.get_prompt(task_name)

    def _humanoid_teacher(
        self,
        image_path: str,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        loader = getattr(self._load_prompt_module(dataset_name), "teacher_for_image", None)
        rec = loader(image_path) if callable(loader) else None
        return rec if isinstance(rec, dict) else {}

    @staticmethod
    def _normalize_anchor_class_name(object_class: Any) -> str:
        return str(object_class or "").strip().lower().replace("-", " ").replace("_", " ")

    @staticmethod
    def _normalize_scene_domain(scene_domain: Optional[str] = None) -> str:
        value = str(scene_domain or config.scene_domain or "outdoor").strip().lower()
        if value not in {"indoor", "outdoor"}:
            raise ValueError(
                f"Unsupported scene domain: {value}. Expected 'indoor' or 'outdoor'."
            )
        return value

    @staticmethod
    def _requires_upright_jrdb_dynamic_anchor_contract(
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> bool:
        dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
            dataset_name or config.dataset
        )
        return dataset == "jrdb" and task_name in {
            "dynamic_height_anchors",
            "height_anchors",
        }

    # Datasets whose static anchors come from a two-pass open-vocabulary flow:
    # pass 1 discovers candidate categories in the frame, pass 2 boxes them.
    # humanoid_everyday uses the DROID-style closed list instead.
    OPEN_VOCAB_STATIC_ANCHOR_DATASETS = frozenset({"droid"})

    @staticmethod
    def _uses_open_vocab_static_anchor_prompt(
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> bool:
        dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
            dataset_name or config.dataset
        )
        return (
            dataset in VLMSemanticScaleAnchor.OPEN_VOCAB_STATIC_ANCHOR_DATASETS
            and task_name == "static_height_anchors"
        )

    @staticmethod
    def _is_valid_upright_jrdb_dynamic_anchor(anchor: Dict[str, Any], object_class: str) -> bool:
        if object_class not in {"pedestrian", "person"}:
            return False
        if str(anchor.get("body_pose", "") or "").strip().lower() not in {"standing", "walking"}:
            return False
        if anchor.get("full_body_visible") is not True:
            return False
        if anchor.get("feet_visible") is not True:
            return False
        if anchor.get("floor_contact_clear") is not True:
            return False
        if anchor.get("is_occluded") is True:
            return False
        if anchor.get("is_truncated") is True:
            return False
        return True

    def _get_prompt_anchor_type_priors(
        self,
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        prompt_module = self._load_prompt_module(dataset_name)
        if not hasattr(prompt_module, "get_anchor_type_priors"):
            return None

        try:
            priors = prompt_module.get_anchor_type_priors(
                task_name, scene_domain=config.scene_domain
            )
        except TypeError:
            try:
                priors = prompt_module.get_anchor_type_priors(task_name)
            except (TypeError, ValueError):
                return None
        except ValueError:
            return None

        if not isinstance(priors, dict):
            return None
        return priors

    def _get_allowed_classes_for_task(
        self,
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> Optional[set]:
        if self._uses_open_vocab_static_anchor_prompt(task_name, dataset_name):
            return None
        priors = self._get_prompt_anchor_type_priors(task_name, dataset_name)
        if priors:
            return {self._normalize_anchor_class_name(name) for name in priors}
        scene_domain = self._normalize_scene_domain()
        if task_name == "dynamic_height_anchors":
            class_names = (
                self.indoor_dynamic_anchor_classes
                if scene_domain == "indoor"
                else self.outdoor_dynamic_anchor_classes
            )
            return {self._normalize_anchor_class_name(name) for name in class_names}
        if task_name == "static_height_anchors":
            class_names = (
                self.indoor_static_anchor_classes
                if scene_domain == "indoor"
                else self.outdoor_static_anchor_classes
            )
            return {self._normalize_anchor_class_name(name) for name in class_names}
        return None

    # --------------------------
    # Public API
    # --------------------------
    def process_image(self, image_path: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        dynamic_height_anchors = self.mine_dynamic_height_anchors(image_path, dataset_name)
        static_height_anchors = self.mine_static_height_anchors(image_path, dataset_name)
        scene_planes = self.mine_scene_semantic_planes(image_path, dataset_name)
        return {
            "dynamic_height_anchors": dynamic_height_anchors,
            "static_height_anchors": static_height_anchors,
            "scene_planes": scene_planes,
        }

    def visualize(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        dynamic_height_anchors: Optional[List[Dict[str, Any]]] = None,
        static_height_anchors: Optional[List[Dict[str, Any]]] = None,
        scene_planes: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(image_path))
        vis_dir = output_dir
        os.makedirs(vis_dir, exist_ok=True)

        dynamic_color = (255, 0, 0)  # BGR: blue
        static_color = (0, 165, 255)  # BGR: orange
        plane_color = (0, 255, 0)   # BGR: green

        for a in (dynamic_height_anchors or []):
            bbox = a.get("object_bbox_2d", None)
            if bbox:
                interval = a.get("height_interval", None)
                conf = a.get("confidence", 0.0)
                label = f"[D] {a.get('object_class','obj')} {interval}m {conf:.2f}"
                self._draw_bbox_with_label(img, bbox, label, dynamic_color)

        for a in (static_height_anchors or []):
            bbox = a.get("object_bbox_2d", None)
            if bbox:
                interval = a.get("height_interval", None)
                conf = a.get("confidence", 0.0)
                label = f"[S] {a.get('object_class','obj')} {interval}m {conf:.2f}"
                self._draw_bbox_with_label(img, bbox, label, static_color)

        if scene_planes is not None:
            for p in scene_planes:
                bbox = p.get("plane_bbox_2d", None)
                if bbox:
                    conf = p.get("confidence", 0.0)
                    label = f"[P] {p.get('plane_type','plane')} {conf:.2f}"
                    self._draw_bbox_with_label(img, bbox, label, plane_color)

        base = os.path.basename(image_path)
        stem, _ = os.path.splitext(base)
        output_path = os.path.join(vis_dir, f"{stem}_visualized.jpg")
        cv2.imwrite(output_path, img)
        return output_path

    # --------------------------
    # Core: Height Anchors
    # --------------------------
    def _discover_open_vocab_static_anchor_classes(
        self,
        image_path: str,
        dataset_name: Optional[str] = None,
    ) -> List[str]:
        prompt = self._get_prompt("static_anchor_category_discovery", dataset_name)
        resp = self._safe_vlm_json_call(prompt, image_path)
        if not isinstance(resp, dict):
            return []

        raw_classes = resp.get("candidate_static_classes", [])
        if not isinstance(raw_classes, list):
            return []

        classes = []
        seen = set()
        prompt_module = self._load_prompt_module(dataset_name)
        canonicalize = getattr(prompt_module, "canonicalize_anchor_class", None)
        for class_name in raw_classes:
            normalized = self._normalize_anchor_class_name(class_name)
            if callable(canonicalize):
                normalized = self._normalize_anchor_class_name(canonicalize(normalized))
            if not normalized or normalized in seen:
                continue
            if self._is_blocked_anchor_class(normalized, dataset_name):
                continue
            seen.add(normalized)
            classes.append(normalized)
        return classes[:12]

    def _get_height_anchor_prompt_for_task(
        self,
        image_path: str,
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> Tuple[str, Optional[set]]:
        """Return the mining prompt plus the class set the answer must stay inside.

        For the open-vocabulary flow the allowed set is whatever pass 1 found in
        this exact image, so pass 2 cannot rename a candidate or invent a class
        that was never proposed.
        """
        if not self._uses_open_vocab_static_anchor_prompt(task_name, dataset_name):
            prompt = self._get_prompt(task_name, dataset_name)
            dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
                dataset_name or config.dataset
            )
            builder = getattr(
                self._load_prompt_module(dataset_name),
                "build_static_height_anchors_prompt_closed",
                None,
            )
            if (
                dataset == "humanoid_everyday"
                and task_name == "static_height_anchors"
                and callable(builder)
            ):
                teacher = self._humanoid_teacher(image_path, dataset_name)
                prompt = builder(list(teacher.get("prefer_classes") or []))
            return prompt, self._get_allowed_classes_for_task(task_name, dataset_name)

        prompt_module = self._load_prompt_module(dataset_name)
        build_prompt = getattr(prompt_module, "build_static_height_anchors_prompt_v2", None)
        if build_prompt is None:
            raise AttributeError(
                f"Prompt module {prompt_module.__name__} must define "
                "build_static_height_anchors_prompt_v2 for open-vocabulary static anchors."
            )

        candidate_classes = self._discover_open_vocab_static_anchor_classes(
            image_path,
            dataset_name,
        )
        if not candidate_classes:
            return build_prompt(candidate_classes), set()
        return (
            build_prompt(candidate_classes),
            {self._normalize_anchor_class_name(name) for name in candidate_classes},
        )

    def _min_static_anchor_height_m(self, dataset_name: Optional[str] = None) -> Optional[float]:
        """Dataset-declared hard floor on a static anchor's real height.

        Datasets that manipulate small tabletop objects (droid) declare no floor
        and keep the previous behaviour. humanoid_everyday declares one because a
        few-centimetre object cannot carry a room-scale height cue.
        """
        prompt_module = self._load_prompt_module(dataset_name)
        value = getattr(prompt_module, "STATIC_ANCHOR_MIN_HEIGHT_M", None)
        return None if value is None else float(value)

    def _is_blocked_anchor_class(
        self,
        class_name: str,
        dataset_name: Optional[str] = None,
    ) -> bool:
        """Drop labels the dataset prompt module has banned.

        humanoid_everyday ships a word-boundary blocklist of classes that
        wrecked the 45-sequence v2 run. Datasets without that helper keep the
        previous structural ban: the sensor platform, people, and the room
        itself.
        """
        normalized = self._normalize_anchor_class_name(class_name)
        if not normalized:
            return True
        prompt_module = self._load_prompt_module(dataset_name)
        checker = getattr(prompt_module, "is_blocked_anchor_class", None)
        if callable(checker):
            return bool(checker(normalized))
        return normalized in {
            "robot",
            "robot arm",
            "robot gripper",
            "gripper",
            "arm",
            "hand",
            "finger",
            "person",
            "wall",
            "floor",
            "ground",
            "ceiling",
            "shadow",
            "reflection",
        }

    @staticmethod
    def _is_valid_height_interval(anchor: Dict[str, Any], min_height_m: Optional[float]) -> bool:
        interval = anchor.get("height_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return False
        try:
            low, high = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            return False
        if not (low > 0.0 and high >= low):
            return False
        if min_height_m is not None and low < float(min_height_m):
            return False
        return True

    def _mine_height_semantic_anchors_for_task(self, image_path: str, task_name: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt, allowed_classes = self._get_height_anchor_prompt_for_task(
            image_path,
            task_name,
            dataset_name,
        )
        min_height_m = (
            self._min_static_anchor_height_m(dataset_name)
            if task_name == "static_height_anchors"
            else None
        )

        resp = self._safe_vlm_json_call(prompt, image_path)
        if not isinstance(resp, list):
            return []

        # scale bbox to pixels + validate
        w, h = Image.open(image_path).size
        anchors = []
        require_upright_jrdb_dynamic_anchor = (
            self._requires_upright_jrdb_dynamic_anchor_contract(task_name, dataset_name)
        )
        dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
            dataset_name or config.dataset
        )
        for a in resp:
            if not isinstance(a, dict):
                continue
            if "object_bbox_2d" not in a:
                continue
            object_class = self._normalize_anchor_class_name(a.get("object_class"))
            canonicalize = getattr(
                self._load_prompt_module(dataset_name),
                "canonicalize_anchor_class",
                None,
            )
            if callable(canonicalize):
                object_class = self._normalize_anchor_class_name(canonicalize(object_class))
            if self._is_blocked_anchor_class(object_class, dataset_name):
                continue
            if allowed_classes is not None and object_class not in allowed_classes:
                continue
            a["object_class"] = object_class
            lookup_interval = getattr(
                self._load_prompt_module(dataset_name),
                "lookup_allowed_height_interval",
                None,
            )
            if callable(lookup_interval):
                table_interval = lookup_interval(object_class)
                if table_interval is not None:
                    a["height_interval"] = table_interval
            if not self._is_valid_height_interval(a, min_height_m):
                continue
            if require_upright_jrdb_dynamic_anchor and not self._is_valid_upright_jrdb_dynamic_anchor(
                a,
                object_class,
            ):
                continue
            
            # For non-JRDB prompts, lateral truncation can still be acceptable
            # when the top-to-bottom height cue remains reliable.
            if a.get("is_occluded", False) is True:
                continue
            if (
                task_name == "static_height_anchors"
                and dataset == "humanoid_everyday"
                and (
                    a.get("top_visible") is False
                    or a.get("base_visible") is False
                    or self._has_declared_occluder(a)
                )
            ):
                continue

            bbox_px = self._bbox_to_pixels(a["object_bbox_2d"], w, h, key_name="object_bbox_2d")
            if bbox_px is None:
                continue

            if (
                dataset == "humanoid_everyday"
                and task_name == "static_height_anchors"
                and (
                    self._is_unusable_humanoid_static_box(
                        a, bbox_px, (w, h), source=image_path
                    )
                    or self._is_loose_humanoid_static_box(a, bbox_px, (w, h))
                )
            ):
                continue

            a["object_bbox_2d"] = bbox_px
            anchors.append(a)

        # filtering + NMS + topK
        reject_vertical_truncation = not (
            dataset == "humanoid_everyday" and task_name == "static_height_anchors"
        )
        anchors = self._filter_anchors_by_quality(
            anchors,
            image_wh=(w, h),
            bbox_key="object_bbox_2d",
            min_area_ratio=0.003,   # less aggressive than 0.01
            min_conf=0.80,
            reject_vertical_truncation=reject_vertical_truncation,
        )
        anchors = self._nms_by_iou(anchors, bbox_key="object_bbox_2d", conf_key="confidence", iou_thr=0.5)
        anchors = sorted(anchors, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        if dataset == "humanoid_everyday" and task_name == "static_height_anchors":
            anchors = self._refine_humanoid_small_boxes(image_path, anchors, w, h)
            anchors = self._ground_empty_humanoid_static(image_path, anchors, w, h)
        return anchors

    def _refine_humanoid_small_boxes(
        self,
        image_path: str,
        anchors: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> List[Dict[str, Any]]:
        refine_prompt = getattr(
            self._load_prompt_module("humanoid_everyday"),
            "get_box_refine_prompt",
            None,
        )
        if not callable(refine_prompt):
            return anchors

        refined: List[Dict[str, Any]] = []
        for anchor in anchors:
            prompt = refine_prompt(anchor.get("object_class"))
            if not prompt:
                refined.append(anchor)
                continue
            box = self._ground_bbox_2d(prompt, image_path, w, h)
            if box is None:
                refined.append(anchor)
                continue
            if self._is_unusable_humanoid_static_box(
                anchor, box, (w, h), source=image_path
            ) or self._is_loose_humanoid_static_box(anchor, box, (w, h)):
                refined.append(anchor)
                continue
            anchor["object_bbox_2d"] = box
            refined.append(anchor)
        return refined

    def _ground_empty_humanoid_static(
        self,
        image_path: str,
        anchors: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> List[Dict[str, Any]]:
        if anchors:
            return anchors
        prompt_module = self._load_prompt_module("humanoid_everyday")
        refine_prompt = getattr(prompt_module, "get_box_refine_prompt", None)
        ground_classes = getattr(prompt_module, "get_empty_frame_ground_classes", None)
        lookup_interval = getattr(prompt_module, "lookup_allowed_height_interval", None)
        allowed = getattr(prompt_module, "HUMANOID_ALLOWED_STATIC_ANCHORS", {})
        if not callable(refine_prompt) or not callable(ground_classes):
            return anchors
        teacher = self._humanoid_teacher(image_path, "humanoid_everyday")
        found: List[Dict[str, Any]] = []
        for class_name in ground_classes(list(teacher.get("prefer_classes") or [])):
            prompt = refine_prompt(class_name)
            if not prompt:
                continue
            box = self._ground_bbox_2d(prompt, image_path, w, h)
            if box is None:
                continue
            spec = allowed.get(class_name) if isinstance(allowed, dict) else None
            interval = lookup_interval(class_name) if callable(lookup_interval) else None
            if spec is None or interval is None:
                continue
            anchor = {
                "object_class": class_name,
                "object_bbox_2d": box,
                "height_interval": interval,
                "distance_level": "near",
                "level": spec["level"],
                "type_weight": spec["weight"],
                "priority": spec["priority"],
                "confidence": 0.90,
                "reason": "Short bbox_2d grounding after the closed-list pass returned empty.",
                "is_occluded": False,
                "is_truncated": False,
            }
            if self._is_unusable_humanoid_static_box(
                anchor, box, (w, h), source=image_path
            ) or self._is_loose_humanoid_static_box(anchor, box, (w, h)):
                continue
            found.append(anchor)
            break
        return found

    def _ground_bbox_2d(self, prompt: str, image_path: str, w: int, h: int) -> Optional[List[int]]:
        resp = self._safe_vlm_json_call(prompt, image_path)
        items = resp if isinstance(resp, list) else [resp] if isinstance(resp, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = item.get("bbox_2d") or item.get("object_bbox_2d")
            box = self._bbox_to_pixels(raw, w, h, key_name="bbox_2d")
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            return box
        return None

    def mine_dynamic_height_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._mine_height_semantic_anchors_for_task(image_path, "dynamic_height_anchors", dataset_name)

    def mine_static_height_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._mine_height_semantic_anchors_for_task(image_path, "static_height_anchors", dataset_name)

    def mine_height_semantic_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.mine_dynamic_height_anchors(image_path, dataset_name) + self.mine_static_height_anchors(image_path, dataset_name)

    # --------------------------
    # Core: Scene Planes
    # --------------------------
    def _collect_scene_planes(
        self,
        image_path: str,
        prompt: str,
        dataset: str,
        allow_raised: bool = False,
    ) -> List[Dict[str, Any]]:
        resp = self._safe_vlm_json_call(prompt, image_path)
        if not isinstance(resp, list):
            return []

        w, h = Image.open(image_path).size
        planes = []
        for p in resp:
            if not isinstance(p, dict):
                continue
            if "plane_bbox_2d" not in p:
                continue
            bbox_px = self._bbox_to_pixels(p["plane_bbox_2d"], w, h, key_name="plane_bbox_2d")
            if bbox_px is None:
                continue
            if dataset == "humanoid_everyday" and self._is_unusable_humanoid_plane_box(
                p, bbox_px, (w, h), allow_raised=allow_raised
            ):
                continue
            p["plane_bbox_2d"] = bbox_px
            planes.append(p)

        planes = self._filter_anchors_by_quality(
            planes,
            image_wh=(w, h),
            bbox_key="plane_bbox_2d",
            min_area_ratio=0.02,
            min_conf=0.60,
            reject_vertical_truncation=False,
        )
        planes = self._nms_by_iou(planes, bbox_key="plane_bbox_2d", conf_key="confidence", iou_thr=0.6)
        planes = sorted(planes, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return planes[:1]

    def mine_scene_semantic_planes(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
            dataset_name or config.dataset
        )
        planes = self._collect_scene_planes(
            image_path,
            self._get_prompt("scene_planes", dataset_name),
            dataset,
            allow_raised=False,
        )
        if planes or dataset != "humanoid_everyday":
            return planes
        teacher = self._humanoid_teacher(image_path, dataset_name)
        raised_prompt = getattr(
            self._load_prompt_module(dataset_name),
            "SCENE_PLANES_PROMPT_RAISED",
            "",
        )
        if not teacher.get("allow_raised_plane") or not raised_prompt:
            return planes
        return self._collect_scene_planes(
            image_path,
            raised_prompt,
            dataset,
            allow_raised=True,
        )

    # --------------------------
    # Utils: VLM call + JSON parsing
    # --------------------------
    def _safe_vlm_json_call(self, prompt: str, image_path: str) -> Any:
        try:
            response = self.vision_model.generate(
                text=prompt,
                data_path=image_path,
                system_text="Return JSON only. Do not include markdown fences or extra text.",
            )
            return self._extract_json(response)
        except Exception as e:
            print(f"[VLM ERROR] {e}")
            message = str(e)
            if "API failed" in message or "401" in message or "Invalid proxy token" in message:
                raise
            return []

    def _extract_json(self, s: str) -> Any:
        if s is None:
            return []
        s = s.strip()

        # remove markdown fences if any
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()

        # try direct load
        try:
            return json.loads(s)
        except Exception:
            pass

        # fallback: find first JSON array/object substring
        m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                return []

        return []

    # --------------------------
    # Utils: bbox conversion / validation
    # --------------------------
    def _bbox_to_pixels(self, bbox: Any, w: int, h: int, key_name: str) -> Optional[List[int]]:
        """
        Accept bbox in either:
        - normalized [0,1000] ints
        - pixel coords (if values look like pixels)
        We convert to pixel coords safely.
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return None

        # Heuristic: if bbox values are within [0,1000] and image may be larger, treat as normalized.
        # If bbox values exceed 1200 or so, treat as pixel.
        is_normalized = max(x1, y1, x2, y2) <= 1000.0

        if is_normalized:
            px = [
                int(round(x1 / 1000.0 * w)),
                int(round(y1 / 1000.0 * h)),
                int(round(x2 / 1000.0 * w)),
                int(round(y2 / 1000.0 * h)),
            ]
        else:
            px = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

        return self._post_process_bbox(px, w, h)

    def _post_process_bbox(self, bbox: List[int], w: int, h: int) -> List[int]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        return [x1, y1, x2, y2]

    @staticmethod
    def _is_lateral_truncated_anchor(item: Dict[str, Any], bbox: List[int], image_wh: Tuple[int, int]) -> bool:
        if item.get("is_truncated", False) is not True:
            return False
        w, _ = image_wh
        x1, _, x2, _ = bbox
        margin = 5
        return x1 <= margin or x2 >= w - 1 - margin

    @staticmethod
    def _is_vertical_truncated_anchor(bbox: List[int], image_wh: Tuple[int, int]) -> bool:
        _, y1, _, y2 = bbox
        _, h = image_wh
        margin = 3
        return y1 <= margin or y2 >= h - 1 - margin

    # A static anchor's height is only readable from a side or mildly oblique
    # view. The prompt asks the VLM to name the viewpoint it judged, so the
    # unreadable ones can be dropped by their own admission.
    HUMANOID_MEASURABLE_VIEWPOINTS = frozenset({"side", "oblique"})
    HUMANOID_TOPDOWN_PART_CLASSES = frozenset({
        "bottle cap",
        "kettle lid",
        "plate",
        "kettle",
        "electric kettle",
        "boiling pot",
        "bottle",
        "cup",
        "mug",
        "basket",
        "bowl",
        "box",
        "file tray",
        "magnetic container",
        "eraser",
        "stapler",
        "light switch",
        "cube",
        "toy cube",
        "dumpling toy",
        "tennis ball",
        "american football",
        "toilet paper roll",
        "pencil case",
        "air pump",
        "elevator button",
    })

    @staticmethod
    def _has_declared_occluder(anchor: Dict[str, Any]) -> bool:
        """Drop an anchor whose own occluders_on_object list is non-empty.

        The VLM sometimes answers with a bare string instead of a list, so a
        non-empty string counts as a declared occluder too.
        """
        occluders = anchor.get("occluders_on_object")
        if isinstance(occluders, str):
            return bool(occluders.strip())
        if not isinstance(occluders, (list, tuple)):
            return False
        return any(str(entry or "").strip() for entry in occluders)

    @staticmethod
    def _is_unmeasurable_viewpoint(anchor: Dict[str, Any]) -> bool:
        """Drop top-down, bottom-up, and flat head-on views of a full object.

        A complete bottle cap or kettle lid is kept from above, because that is
        the usable height cue when the body is foreshortened.
        """
        viewpoint = str(anchor.get("viewpoint") or "").strip().lower().replace("-", "_")
        if not viewpoint:
            return False
        if viewpoint in VLMSemanticScaleAnchor.HUMANOID_MEASURABLE_VIEWPOINTS:
            return False
        object_class = str(anchor.get("object_class") or "").strip().lower()
        if (
            viewpoint in {"top_down", "topdown", "nadir", "overhead"}
            and object_class in VLMSemanticScaleAnchor.HUMANOID_TOPDOWN_PART_CLASSES
        ):
            return False
        return True

    @staticmethod
    def _is_unusable_humanoid_static_box(
        anchor: Dict[str, Any],
        bbox: List[int],
        image_wh: Tuple[int, int],
        source: Optional[str] = None,
    ) -> bool:
        """Drop the boxes that applied a full-object height to a clipped view.

        The outdoor chair/table run used 0.7–0.9 m table height on a downward
        view of the tabletop that filled the frame. A head camera often nicks
        the top of a still-usable chair, so a mere touch of the image edge is
        not enough to drop the box.
        """
        w, h = image_wh
        x1, y1, x2, y2 = bbox
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(max(w * h, 1))
        vertical = y1 <= 3 or y2 >= h - 4
        cue = str(anchor.get("height_cue") or "").lower()
        admitted = (
            anchor.get("is_truncated") is True
            or "inferred" in cue
            or "image border" in cue
            or "truncated" in cue
        )
        cls = str(anchor.get("object_class") or "").strip().lower()
        source_text = str(source or "")
        if cls in {"door handle", "door lock"}:
            # 3–15 cm hardware. peak_to_bottom is not a standing-object height.
            return True
        if (
            cls in {"kettle", "electric kettle"}
            and "pour_water_from_the_pot_to_a_cup" in source_text
        ):
            # Pouring tilts the kettle; the 18–28 cm interval no longer matches.
            return True
        top_only_nick = y1 <= 3 and y2 < h - 4
        furniture = any(
            name in cls
            for name in (
                "table",
                "desk",
                "chair",
                "stool",
                "cabinet",
                "locker",
                "door",
                "bin",
                "trash",
                "stanchion",
                "bollard",
                "microwave",
                "machine",
                "maker",
            )
        )
        tokens = set(cls.replace("-", " ").split())
        small = bool(
            tokens
            & {
                "kettle",
                "bottle",
                "cup",
                "plate",
                "ball",
                "football",
                "balloon",
                "cap",
                "lid",
            }
        )
        allow_top_nick = furniture or cls in {"bollard", "stanchion"}
        if admitted and vertical and not (allow_top_nick and top_only_nick):
            return True
        # True fill-frame crop: the box eats both the top and the bottom.
        if y1 <= 3 and y2 >= h - 4 and area_ratio >= 0.55:
            return True
        # A close-up kettle/bottle can fill a third of the frame. Only drop
        # a small-object box that ate most of the image.
        if small and area_ratio >= 0.70:
            return True
        if "trash" in cls and area_ratio >= 0.70:
            return True
        if cls == "elevator button":
            height_ratio = max(0, y2 - y1) / float(max(h, 1))
            # A round 2–3.5 cm button never fills the frame. The whole black
            # gate post does, and that height is not the button interval.
            if area_ratio >= 0.08 or height_ratio >= 0.35:
                return True
        if "duster" in tokens and y1 <= 3:
            # Head cut by the top of the image. The box then swallows the
            # cabinet face and peak_to_bottom becomes cabinet height.
            return True
        if "extinguisher" in tokens and y1 <= 3:
            # Same head cut. mix43v3 also tags the cabinet as the cylinder
            # and then applies 40-70 cm to a wall-height box.
            return True
        if cls in {"stool", "chair", "office chair"} and (y1 <= 3 or y2 < h - 8):
            # Stool/chair height is backrest-top to floor. A head-cut box
            # is seat-and-legs only. A box that does not reach the bottom
            # of the image is a close-up, not a standing floor view.
            return True
        if cls in {"shampoo", "shampoo bottle", "bottle"} and y1 <= 3:
            # Cap cut by the top of the image. peak_to_bottom is no longer
            # the 15–25 cm bottle interval.
            return True
        if cls in {"kettle", "electric kettle", "boiling pot"} and x1 <= 3:
            # Half the kettle has left the left edge. Pickup frames do this.
            return True
        return False

    @staticmethod
    def _is_loose_humanoid_static_box(
        anchor: Dict[str, Any],
        bbox: List[int],
        image_wh: Tuple[int, int],
    ) -> bool:
        """Drop a small object whose box ate most of the image."""
        w, h = image_wh
        x1, y1, x2, y2 = bbox
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(max(w * h, 1))
        cls = str(anchor.get("object_class") or "").strip().lower()
        tokens = set(cls.replace("-", " ").split())
        small = bool(tokens & {"kettle", "bottle", "cup", "plate", "cap", "lid"})
        if small and area_ratio >= 0.35:
            return True
        if "duster" in tokens and area_ratio >= 0.22:
            return True
        if cls == "elevator button" and area_ratio >= 0.08:
            return True
        if cls in {"kettle", "electric kettle", "boiling pot"} and y1 > 3 and area_ratio >= 0.22:
            # Close-up or pouring fill. The working kettle view nicks the
            # top of the image (y1==0) and keeps the body-to-base.
            return True
        return False

    HUMANOID_BOTTLE_HEIGHT_CLASSES = {"shampoo", "shampoo bottle", "bottle"}
    HUMANOID_DESK_ORGANIZER_CLASSES = {"basket", "file tray"}

    @staticmethod
    def rejected_humanoid_conflicting_class(
        anchor: Dict[str, Any],
        present_classes: Optional[set] = None,
    ) -> Optional[str]:
        """Drop a desk organizer when a bottle-height class is also in the batch.

        Shampoo is 15–25 cm. Basket/file tray is 5–12 cm. One shared scale
        cannot take both. The bottle-height class keeps the vote.
        """
        if not present_classes:
            return None
        cls = str(anchor.get("object_class") or "").strip().lower()
        if (
            cls in VLMSemanticScaleAnchor.HUMANOID_DESK_ORGANIZER_CLASSES
            and present_classes & VLMSemanticScaleAnchor.HUMANOID_BOTTLE_HEIGHT_CLASSES
        ):
            return "conflicting_desk_organizer"
        return None

    @staticmethod
    def rejected_humanoid_static_box_stage(
        anchor: Dict[str, Any],
        bbox: List[int],
        image_wh: Tuple[int, int],
        present_classes: Optional[set] = None,
        source: Optional[str] = None,
    ) -> Optional[str]:
        if VLMSemanticScaleAnchor._is_unusable_humanoid_static_box(
            anchor, bbox, image_wh, source=source
        ):
            return "unusable_humanoid_static_box"
        if VLMSemanticScaleAnchor._is_loose_humanoid_static_box(anchor, bbox, image_wh):
            return "loose_humanoid_static_box"
        return VLMSemanticScaleAnchor.rejected_humanoid_conflicting_class(anchor, present_classes)

    @staticmethod
    def _is_unusable_humanoid_plane_box(
        plane: Dict[str, Any],
        bbox: List[int],
        image_wh: Tuple[int, int],
        allow_raised: bool = False,
    ) -> bool:
        """Drop plane boxes that wrap a foreground object or guess occluded floor.

        A real floor strip in this head-cam view reaches the bottom of the
        image and does not start as a mid-frame box around a duster/gripper.
        Raised desk/counter tops are allowed only when the teacher says the
        better historical run needed them.
        """
        w, h = image_wh
        x1, y1, x2, y2 = bbox
        if h <= 0 or w <= 0:
            return True
        plane_type = str(plane.get("plane_type") or "").strip().lower()
        surface = str(plane.get("surface_kind") or plane.get("plane_type") or "").lower()
        reason = str(plane.get("reason") or "").lower()
        raised_types = {"counter top", "table top", "desk top"}
        is_raised = plane_type in raised_types
        if plane_type != "floor" and not (allow_raised and is_raised):
            return True
        raised_surface = any(
            token in f"{plane_type} {surface} {reason}"
            for token in ("table", "desk", "counter", "cabinet top")
        )
        height_ratio = max(0, y2 - y1) / float(h)
        if not is_raised:
            if y1 <= 0.25 * h and height_ratio >= 0.50:
                return True
            if raised_surface:
                return True

        # The VLM must enumerate what sits inside its own box. Anything listed
        # means the patch is not bare surface.
        occluders = plane.get("occluders_in_box")
        if isinstance(occluders, (list, tuple)) and len(occluders) > 0:
            return True
        if plane.get("box_is_clear") is False:
            return True
        declared_clear = isinstance(occluders, (list, tuple)) and not occluders

        # A box that spans most of the frame usually swallows legs, the arm, or
        # objects standing on the surface. A bare room-floor sweep to the bottom
        # edge is the exception. Desk / table / counter tops are not planes
        # unless the teacher explicitly allows a raised fallback.
        area_ratio = (max(0, x2 - x1) * max(0, y2 - y1)) / float(w * h)
        if is_raised:
            if area_ratio > 0.70:
                return True
        elif area_ratio > 0.35:
            is_ground_sweep = declared_clear and y1 >= 0.30 * h and y2 >= h - 8
            if not is_ground_sweep:
                return True
        # Wordings that admit the surface is only partly visible, for every
        # surface kind and not just the room floor.
        for phrase in (
            "behind the",
            "beneath the",
            "under the",
            "underneath",
            "partially",
            "partly",
            "obscur",
            "occluding",
            "occluded by",
            "covered by",
            "hidden by",
            "blocked by",
            "except for",
            "excluding the",
            "around the",
            "between the",
            "next to the",
            "sliver",
            "narrow strip",
        ):
            if phrase in reason:
                return True
        if plane_type == "floor" and y2 < h - 8:
            return True
        return False

    # --------------------------
    # Utils: filtering / NMS
    # --------------------------
    def _filter_anchors_by_quality(
        self,
        items: List[Dict[str, Any]],
        image_wh: Tuple[int, int],
        bbox_key: str,
        min_area_ratio: float,
        min_conf: float,
        reject_vertical_truncation: bool = True,
    ) -> List[Dict[str, Any]]:
        w, h = image_wh
        img_area = float(w * h)
        out = []
        for it in items:
            bbox = it.get(bbox_key, None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            area = float(bw * bh)
            if area <= 0:
                continue
            is_lateral_truncated = self._is_lateral_truncated_anchor(it, bbox, image_wh)
            is_vertical_truncated = self._is_vertical_truncated_anchor(bbox, image_wh)
            if reject_vertical_truncation and is_vertical_truncated and not is_lateral_truncated:
                continue
            if area / img_area < float(min_area_ratio) and not is_lateral_truncated:
                continue
            conf = float(it.get("confidence", 0.0))
            if conf < float(min_conf) and not is_lateral_truncated:
                continue
            out.append(it)
        return out

    def _iou(self, b1: List[int], b2: List[int]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return float(inter / union) if union > 0 else 0.0

    def _nms_by_iou(
        self,
        items: List[Dict[str, Any]],
        bbox_key: str,
        conf_key: str,
        iou_thr: float,
    ) -> List[Dict[str, Any]]:
        items = sorted(items, key=lambda x: float(x.get(conf_key, 0.0)), reverse=True)
        kept = []
        for it in items:
            bbox = it.get(bbox_key, None)
            if not bbox:
                continue
            keep = True
            for kt in kept:
                if self._iou(bbox, kt[bbox_key]) > float(iou_thr):
                    keep = False
                    break
            if keep:
                kept.append(it)
        return kept

    # --------------------------
    # Utils: drawing
    # --------------------------
    def _draw_bbox_with_label(self, img, bbox, label: str, color: Tuple[int, int, int]):
        text_color = (255, 255, 255)
        text_bg_color = (0, 0, 0)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        tx = x1
        ty = y1 - baseline - 5
        if ty < th:
            ty = y2 + th + 5

        cv2.rectangle(
            img,
            (tx, ty - th - baseline),
            (tx + tw, ty + baseline),
            text_bg_color,
            -1,
        )
        cv2.putText(img, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
