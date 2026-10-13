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

    @staticmethod
    def _uses_open_droid_static_anchor_prompt(
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> bool:
        dataset = VLMSemanticScaleAnchor._normalize_dataset_module_name(
            dataset_name or config.dataset
        )
        return dataset == "droid" and task_name == "static_height_anchors"

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
            priors = prompt_module.get_anchor_type_priors(task_name)
        except (TypeError, ValueError):
            return None

        if not isinstance(priors, dict):
            return None
        return priors

    def _get_allowed_classes_for_task(
        self,
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> Optional[set]:
        if self._uses_open_droid_static_anchor_prompt(task_name, dataset_name):
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
    def _discover_droid_static_anchor_classes(
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
        blocked = {
            "robot",
            "robot arm",
            "gripper",
            "hand",
            "arm",
            "cable",
            "cloth",
            "towel",
            "paper",
            "paper sheet",
            "plate",
            "lid",
            "mat",
            "shadow",
            "wall",
            "screen",
            "reflection",
        }
        for class_name in raw_classes:
            normalized = self._normalize_anchor_class_name(class_name)
            if not normalized or normalized in seen or normalized in blocked:
                continue
            seen.add(normalized)
            classes.append(normalized)
        return classes[:12]

    def _get_height_anchor_prompt_for_task(
        self,
        image_path: str,
        task_name: str,
        dataset_name: Optional[str] = None,
    ) -> str:
        if not self._uses_open_droid_static_anchor_prompt(task_name, dataset_name):
            return self._get_prompt(task_name, dataset_name)

        prompt_module = self._load_prompt_module(dataset_name)
        build_prompt = getattr(prompt_module, "build_static_height_anchors_prompt", None)
        if build_prompt is None:
            return self._get_prompt(task_name, dataset_name)

        candidate_classes = self._discover_droid_static_anchor_classes(image_path, dataset_name)
        return build_prompt(candidate_classes)

    def _mine_height_semantic_anchors_for_task(self, image_path: str, task_name: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt = self._get_height_anchor_prompt_for_task(image_path, task_name, dataset_name)
        allowed_classes = self._get_allowed_classes_for_task(task_name, dataset_name)

        resp = self._safe_vlm_json_call(prompt, image_path)
        if not isinstance(resp, list):
            return []

        # scale bbox to pixels + validate
        w, h = Image.open(image_path).size
        anchors = []
        require_upright_jrdb_dynamic_anchor = (
            self._requires_upright_jrdb_dynamic_anchor_contract(task_name, dataset_name)
        )
        for a in resp:
            if not isinstance(a, dict):
                continue
            if "object_bbox_2d" not in a:
                continue
            object_class = self._normalize_anchor_class_name(a.get("object_class"))
            if allowed_classes is not None and object_class not in allowed_classes:
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

            bbox_px = self._bbox_to_pixels(a["object_bbox_2d"], w, h, key_name="object_bbox_2d")
            if bbox_px is None:
                continue

            a["object_bbox_2d"] = bbox_px
            anchors.append(a)

        # filtering + NMS + topK
        anchors = self._filter_anchors_by_quality(
            anchors,
            image_wh=(w, h),
            bbox_key="object_bbox_2d",
            min_area_ratio=0.003,   # less aggressive than 0.01
            min_conf=0.70,
        )
        anchors = self._nms_by_iou(anchors, bbox_key="object_bbox_2d", conf_key="confidence", iou_thr=0.5)
        anchors = sorted(anchors, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return anchors

    def mine_dynamic_height_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._mine_height_semantic_anchors_for_task(image_path, "dynamic_height_anchors", dataset_name)

    def mine_static_height_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._mine_height_semantic_anchors_for_task(image_path, "static_height_anchors", dataset_name)

    def mine_height_semantic_anchors(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.mine_dynamic_height_anchors(image_path, dataset_name) + self.mine_static_height_anchors(image_path, dataset_name)

    # --------------------------
    # Core: Scene Planes
    # --------------------------
    def mine_scene_semantic_planes(self, image_path: str, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt = self._get_prompt("scene_planes", dataset_name)

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
            p["plane_bbox_2d"] = bbox_px
            planes.append(p)

        planes = self._filter_anchors_by_quality(
            planes,
            image_wh=(w, h),
            bbox_key="plane_bbox_2d",
            min_area_ratio=0.02,   # planes should be big
            min_conf=0.60,
            reject_vertical_truncation=False,
        )
        planes = self._nms_by_iou(planes, bbox_key="plane_bbox_2d", conf_key="confidence", iou_thr=0.6)
        planes = sorted(planes, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        # The downstream ground-plane estimator assumes one dominant semantic
        # ground proposal per frame. Enforce the prompt contract here instead of
        # trusting the VLM to always obey "at most 1 plane".
        return planes[:1]

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
