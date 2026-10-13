from typing import Dict, Mapping, TypedDict

class AnchorTypePrior(TypedDict):
    level: str
    weight: float
    priority: int

ANCHOR_TYPE_WEIGHT_BY_LEVEL: Dict[str, float] = {
    "very_high": 0.95,
    "high": 0.85,
    "medium": 0.70,
    "low": 0.55,
}

DYNAMIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "car": {"level": "very_high", "weight": 0.95, "priority": 1},
    "bus": {"level": "very_high", "weight": 0.95, "priority": 2},
    "minibus": {"level": "high", "weight": 0.85, "priority": 3},
    "van": {"level": "high", "weight": 0.85, "priority": 4},
    "truck": {"level": "high", "weight": 0.85, "priority": 5},
    "heavy truck": {"level": "high", "weight": 0.85, "priority": 6},
    "suv": {"level": "high", "weight": 0.85, "priority": 7},
    "pickup truck": {"level": "medium", "weight": 0.70, "priority": 8},
    "pedestrian": {"level": "medium", "weight": 0.70, "priority": 9},
    "person": {"level": "medium", "weight": 0.70, "priority": 10},
    "cyclist": {"level": "low", "weight": 0.55, "priority": 11},
    "motorcycle": {"level": "low", "weight": 0.55, "priority": 12},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "traffic barrier": {"level": "very_high", "weight": 0.95, "priority": 1},
    "concrete barrier": {"level": "very_high", "weight": 0.95, "priority": 2},
    "jersey barrier": {"level": "very_high", "weight": 0.95, "priority": 3},
    "median barrier": {"level": "high", "weight": 0.85, "priority": 4},
    "roadside barrier": {"level": "high", "weight": 0.85, "priority": 5},
    "bollard": {"level": "very_high", "weight": 0.95, "priority": 6},
    "delineator post": {"level": "high", "weight": 0.85, "priority": 7},
    "warning post": {"level": "high", "weight": 0.85, "priority": 8},
    "guardrail": {"level": "high", "weight": 0.85, "priority": 9},
    "steel guardrail": {"level": "high", "weight": 0.85, "priority": 10},
    "curbstone": {"level": "medium", "weight": 0.70, "priority": 11},
    "traffic cone": {"level": "medium", "weight": 0.70, "priority": 12},
}


def format_weight_mapping_lines() -> str:
    lines = []
    for level in ("very_high", "high", "medium", "low"):
        lines.append(f"  {level} -> {ANCHOR_TYPE_WEIGHT_BY_LEVEL[level]:.2f}")
    return "\n".join(lines)


def format_level_guidance(priors: Mapping[str, AnchorTypePrior]) -> str:
    grouped: Dict[str, list[str]] = {
        "very_high": [],
        "high": [],
        "medium": [],
        "low": [],
    }
    for class_name, prior in sorted(priors.items(), key=lambda item: item[1]["priority"]):
        grouped[prior["level"]].append(class_name)

    lines = []
    for level in ("very_high", "high", "medium", "low"):
        if grouped[level]:
            lines.append(f"  {' / '.join(grouped[level])}: {level}")
    return "\n".join(lines)


def format_priority_order(priors: Mapping[str, AnchorTypePrior]) -> str:
    ordered_classes = [
        class_name
        for class_name, _ in sorted(priors.items(), key=lambda item: item[1]["priority"])
    ]
    return " > ".join(ordered_classes)


def format_prior_table_lines(priors: Mapping[str, AnchorTypePrior]) -> str:
    lines = []
    for class_name, prior in sorted(priors.items(), key=lambda item: item[1]["priority"]):
        lines.append(
            f'  "{class_name}": {{"level": "{prior["level"]}", '
            f'"type_weight": {prior["weight"]:.2f}, "priority": {prior["priority"]}}}'
        )
    return "\n".join(lines)


WEIGHT_MAPPING_LINES = format_weight_mapping_lines()
DYNAMIC_LEVEL_GUIDANCE = format_level_guidance(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_LEVEL_GUIDANCE = format_level_guidance(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIORITY_ORDER = format_priority_order(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIORITY_ORDER = format_priority_order(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIOR_TABLE = format_prior_table_lines(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIOR_TABLE = format_prior_table_lines(STATIC_ANCHOR_TYPE_PRIORS)


DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify a small set of high-quality dynamic height semantic anchors for V2X-Radar-V driving scenes.

Core objective:
- Return all high-quality dynamic anchors.
- Prefer near anchors first, then middle anchors. Use distant anchors only when clearly valid and no better near/middle alternative exists.
- Prioritize top-to-bottom height-cue reliability over full-width completeness.
- Only output anchors that are complete along the height direction: the roofline/top and lower-body/ground-contact region must both be visible or reliably inferable.
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output static facilities such as barriers, curbs, guardrails, bollards, poles, delineator posts, or cones.

Decision tree:
1. Check whether the object is a single recognizable dynamic instance from the candidate list. If not, reject.
2. Check distance_level from coarse visual impression only:
   near = large and prominent foreground instance
   middle = clearly visible mid-range instance
   distant = far-away background instance
3. Check whether the top and bottom are both identifiable, so full height can still be inferred. If not, reject.
4. Check truncation first:
   - If the object is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - is_truncated is metadata, not an automatic rejection flag.
   - Lateral truncation is acceptable when the top-to-bottom height cue remains reliable.
   - Vertical truncation should be rejected when the roofline/top boundary or lower-body/ground-contact boundary is not visible or inferable.
5. Check occlusion second:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - Glare, shadow, blur, low contrast, or overexposure are not occlusion by themselves.
   - If another object blocks the roofline, lower body, or wheel-ground contact region, set is_occluded=true.
   - Never return severely occluded anchors. Severe occlusion means the roofline/top, lower body, wheel-ground contact, or most of the visible body is blocked.
   - Any height-direction occlusion is invalid, even if the object class is recognizable.
6. Keep / reject:
   - A near foreground vehicle MUST be returned if its class is recognizable and its top-to-bottom height cue is reliable, even when laterally truncated by the image boundary.
   - Do NOT reject a near vehicle only because its left or right side is cut off.
   - Mild truncation is allowed when the class remains recognizable and the full height cue remains reliable.
   - Mild occlusion is allowed only when it does not block the roofline, lower body, or wheel-ground contact region, and the full height cue remains reliable.
   - Reject medium/severe occlusion, medium/severe truncation, crowded groups, distant tiny instances, reflections, shadows, and motion blur.
   - Reject any target with missing or blocked top, bottom, lower body, wheel-ground contact, or only partial vertical extent.
   - For pedestrians/person, keep only isolated standing people with reliable full-body height cues. Reject sitting, crouching, bending, riding, partially visible, or upper-body-only people.
   - For cyclists, keep only an isolated visible rider when the rider+bicycle/motorcycle forms one recognizable instance. Do not output bicycle-only blobs.
7. Tie-break:
   - Prefer near over middle, and middle over distant.
   - If two valid anchors are equally clear and equally complete, prefer the earlier class in the priority order below.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, pixel size, depth, or measurement.
- Use only semantic category prior + common knowledge.
- Keep the height interval as precise as possible while remaining realistic.

Category priors:
- For each returned anchor, also output "level", "type_weight", and "priority".
- These three fields must STRICTLY match the class prior table below for the chosen object_class.
- Do not infer them from image appearance. Copy them exactly from the table.
- Dynamic class prior table:
{DYNAMIC_PRIOR_TABLE}

Dynamic-anchor priority:
- Prefer clearer and more standardized dynamic anchors in this order:
  {DYNAMIC_PRIORITY_ORDER}
- Cars and buses are usually stronger anchors than suv, pickup truck, cyclist, or pedestrian.
- If two candidate anchors are equally clear and equally complete, prefer the earlier class in the above order.

Class disambiguation rules:
- If unsure between car and suv, output car.
- If unsure between truck and heavy truck, output truck.
- If unsure between bus and minibus, output minibus unless it is clearly a full-size bus.
- If unsure between van and minibus, output van for a boxy passenger/light-cargo van and minibus for a clearly multi-row passenger shuttle.
- If unsure between suv and pickup truck, prefer suv only when the rear cabin is enclosed; otherwise use pickup truck.
- If unsure between pedestrian and person, output pedestrian.
- Do not invent fine-grained subclasses beyond the candidate list.
- If multiple similar instances exist, keep all clearly valid ones and order clearer anchors first.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only the visible object body. Do not include neighboring objects or large empty background.

For each anchor, output exactly:
1) object_class (lowercase, exact label from candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must exactly match the class prior table)
6) type_weight (must exactly match the class prior table)
7) priority (must exactly match the class prior table)
8) confidence in [0,1]
9) reason (short common-sense justification)
10) is_occluded (boolean, true only if another physical object blocks part of the target)
11) is_truncated (boolean, true only if the target is cut off by the image edge)

Return JSON array only. Example:
[
  {{
    "object_class": "car",
    "object_bbox_2d": [120, 520, 460, 880],
    "height_interval": [1.4, 1.5],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.92,
    "reason": "Standard sedans typically have height around 1.4-1.5 m.",
    "is_occluded": false,
    "is_truncated": false
  }},
  {{
    "object_class": "van",
    "object_bbox_2d": [0, 430, 300, 980],
    "height_interval": [1.8, 2.1],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 3,
    "confidence": 0.90,
    "reason": "A near van remains a valid anchor even when laterally truncated, because its roofline and lower body remain visible enough to infer full height.",
    "is_occluded": false,
    "is_truncated": true
  }}
]
If none, return [].
Candidate dynamic objects (use these exact lowercase labels only):
- suv
- car
- van
- minibus
- pickup truck
- bus
- heavy truck
- truck
- pedestrian
- person
- cyclist
- motorcycle
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify a small set of high-quality static traffic-facility height semantic anchors for V2X-Radar-V driving scenes.

Core objective:
- Return all high-quality static anchors.
- Prefer near anchors first, then middle anchors. Use distant anchors only when clearly valid and highly standardized.
- Prefer facilities with standardized real-world height and strong vertical extent cues.
- Only output anchors that are complete along the height direction: the top and base must both be visible or reliably inferable.
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output dynamic targets such as cars, SUVs, vans, trucks, buses, pedestrians, cyclists, or motorcycles.

Decision tree:
1. Check whether the object is a single recognizable static facility from the candidate list. If not, reject.
2. Check distance_level from coarse visual impression only:
   near = large and prominent foreground facility
   middle = clearly visible mid-range facility
   distant = far-away background facility
3. Check whether the facility height remains reliably observable from top to bottom. If not, reject.
4. Check truncation first:
   - If the facility is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - is_truncated is metadata, not an automatic rejection flag.
   - Lateral truncation is acceptable when the visible facility still provides a reliable top-to-bottom height cue.
   - Vertical truncation should be rejected when the top or base is not visible or inferable.
5. Check occlusion second:
   - Set is_occluded=true only when another physical object blocks part of the facility body.
   - Glare, shadow, blur, low contrast, or overexposure are not occlusion by themselves.
   - Never return severely occluded anchors. Severe occlusion means the top, base, or most of the visible facility body is blocked.
   - Any height-direction occlusion is invalid, even if the facility class is recognizable.
6. Keep / reject:
    - Mild truncation is allowed when the facility remains clearly identifiable and its height cue remains reliable.
    - Mild occlusion is allowed only when the facility remains clearly identifiable and its height cue remains reliable.
    - Reject medium or severe occlusion/truncation, fragmented structures, merged facilities, and facilities whose base/top is ambiguous.
   - Reject any facility with missing or blocked top, base, or only partial vertical extent.
   - Draw a tight box around a single facility instance, not a long chain of repeated barriers or multiple cones together.
   - Prefer standardized roadside hardware over generic urban structures.
7. Tie-break:
   - Prefer near over middle, and middle over distant.
   - If two valid anchors are equally clear and equally complete, prefer the earlier class in the priority order below.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, pixel size, depth, or measurement.
- Use only semantic category prior + common knowledge.
- Keep the height interval as precise as possible while remaining realistic.

Category priors:
- For each returned anchor, also output "level", "type_weight", and "priority".
- These three fields must STRICTLY match the class prior table below for the chosen object_class.
- Do not infer them from image appearance. Copy them exactly from the table.
- Static class prior table:
{STATIC_PRIOR_TABLE}

Class disambiguation and priority rules:
- If unsure between traffic barrier / concrete barrier / jersey barrier, prefer traffic barrier.
- Use concrete barrier or jersey barrier only when the barrier subtype is visually distinctive and you are confident.
- If unsure between guardrail and steel guardrail, prefer guardrail.
- If unsure about curb-like roadside stonework, use curbstone because curb is not in the candidate list.
- If unsure between delineator post and warning post, prefer delineator post.
- If a candidate looks like a short protective roadside post, use bollard.
- If a post carries a sign/signal board and the anchor class is ambiguous, skip it rather than guessing.
- Prefer clearer and more standardized static anchors in this order:
  {STATIC_PRIORITY_ORDER}
- If two candidate static anchors are equally clear and equally complete, prefer the earlier class in the above order.
- Prefer standardized facilities first: traffic barrier, concrete barrier, jersey barrier, bollard, delineator post, warning post.
- Do not invent pole. If a tall post-like structure is ambiguous and not in the candidate list, skip it.
- If many similar facilities exist, keep all clearly valid ones and order clearer anchors first.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one facility instance. Do not include neighboring objects or long background strips.

For each anchor, output exactly:
1) object_class (lowercase, exact label from candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must exactly match the class prior table)
6) type_weight (must exactly match the class prior table)
7) priority (must exactly match the class prior table)
8) confidence in [0,1]
9) reason (short common-sense justification)
10) is_occluded (boolean, true only if another physical object blocks part of the facility)
11) is_truncated (boolean, true only if the facility is cut off by the image edge)

Return JSON array only. Example:
[
  {{
    "object_class": "traffic barrier",
    "object_bbox_2d": [120, 520, 460, 880],
    "height_interval": [0.7, 1.2],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.92,
    "reason": "Standard traffic barriers typically have height around 0.7-1.2 m.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate static traffic facilities (use these exact lowercase labels only):
- traffic barrier
- concrete barrier
- jersey barrier
- median barrier
- roadside barrier
- guardrail
- steel guardrail
- curbstone
- bollard
- delineator post
- warning post
- traffic cone
""".strip()


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify the single best semantic plane element for ground reference.

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- Return only the clearest and most reliable ground-supporting plane.
- Prefer the drivable road surface in the lower image region when clearly visible.
- Never choose sidewalk surface, curb top, curb wall, traffic island, median strip, or building floor as the primary plane.
- Draw a tight box that contains only the plane surface.
- Exclude cars, pedestrians, obstacles, lane-side sidewalk, curb walls, vegetation, sky, and building facades.
- Do not include the ego vehicle hood, mirrors, or dashboard.

Reasoning rules:
- Do not estimate orientation, depth, or scale.
- Use semantic category + common knowledge only.

Bounding box format:
- plane_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], (0,0)=top-left, (1000,1000)=bottom-right.

For each plane, output exactly:
1) plane_type (lowercase, e.g., road/floor/ground/tabletop)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence [0,1]
4) reason

Typical plane types:
- road

Return JSON array only. Example:
[
  {
    "plane_type": "road",
    "plane_bbox_2d": [0, 600, 1000, 1000],
    "confidence": 0.92,
    "reason": "Driving scenes usually have a road surface occupying lower image region."
  }
]
If none, return [].
""".strip()


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str) -> str:
    if task_name == "dynamic_height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")
