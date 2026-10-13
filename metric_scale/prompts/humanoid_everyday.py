import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, TypedDict


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
    "pedestrian": {"level": "very_high", "weight": 0.95, "priority": 1},
    "person": {"level": "medium", "weight": 0.70, "priority": 2},
    "cyclist": {"level": "low", "weight": 0.55, "priority": 3},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "bollard": {"level": "high", "weight": 0.85, "priority": 2},
    "stanchion": {"level": "high", "weight": 0.85, "priority": 3},
    "trash can": {"level": "high", "weight": 0.85, "priority": 4},
    "signboard": {"level": "medium", "weight": 0.70, "priority": 5},
    "table": {"level": "low", "weight": 0.55, "priority": 6},
    "chair": {"level": "low", "weight": 0.55, "priority": 7},
}


STATIC_HEIGHT_INTERVALS: Dict[str, tuple] = {
    "door": (2.00, 2.10),
    "bollard": (0.80, 1.10),
    "stanchion": (0.90, 1.00),
    "trash can": (0.30, 0.70),
    "signboard": (1.20, 1.80),
    "table": (0.70, 0.80),
    "chair": (0.80, 1.05),
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


def format_height_interval_table_lines(intervals: Mapping[str, tuple]) -> str:
    lines = []
    for class_name, prior in sorted(
        STATIC_ANCHOR_TYPE_PRIORS.items(), key=lambda item: item[1]["priority"]
    ):
        lo, hi = intervals[class_name]
        lines.append(f'  "{class_name}": [{lo:.2f}, {hi:.2f}]')
    return "\n".join(lines)


WEIGHT_MAPPING_LINES = format_weight_mapping_lines()
DYNAMIC_LEVEL_GUIDANCE = format_level_guidance(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_LEVEL_GUIDANCE = format_level_guidance(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIORITY_ORDER = format_priority_order(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIORITY_ORDER = format_priority_order(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIOR_TABLE = format_prior_table_lines(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIOR_TABLE = format_prior_table_lines(STATIC_ANCHOR_TYPE_PRIORS)
STATIC_HEIGHT_INTERVAL_TABLE = format_height_interval_table_lines(STATIC_HEIGHT_INTERVALS)


DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify a small set of high-quality dynamic height semantic anchors for JRDB indoor/campus robot first-person scenes.

Core objective:
- Return only the best high-quality human-centered dynamic anchors. At most 3 dynamic anchors per frame.
- This is a JRDB human environment. Prefer people first.
- Prefer near anchors first, then clean middle anchors. Do not return distant anchors.
- A usable scale anchor must be visually large, crisp, and separable enough that SAM can mask the whole vertical body reliably.
- Prioritize head-to-feet height-cue reliability over full-width completeness.
- Only output anchors that are complete along the height direction: the head/top and feet/ground-contact region must both be visible or reliably inferable.
- JRDB frames often contain edge-cut half bodies. Reject image-edge partial people even when the class is recognizable.
- JRDB frames often contain far background people that are recognizable but too small for reliable scale. Reject them.
- JRDB frames often contain barricades, railings, columns, counters, tables, chairs, and glass walls. Reject people behind these structures when the legs, feet, or floor-contact region are not cleanly visible.
- JRDB campus frames may contain construction fences, orange safety mesh, black tarps, grass/dirt beds, curbs, ramps, stairs, and temporary barriers. Reject people in those regions unless both feet are clearly on the same stable flat walkable floor as the robot.
- The scale estimate assumes the person's feet lie on the same ground plane used for the scene floor reference. Reject any person visibly above, below, or separated from that ground plane.
- Reject people whose feet are behind a fence/barrier, on a different elevation, on a step, staircase, raised landing, platform, curb, ramp, escalator, bench, furniture, or grass/dirt instead of the main walkable floor.
- Treat truncation and occlusion as reject conditions for JRDB dynamic scale anchors. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the body is near the left/right image border, require clear empty margin outside the whole body; otherwise reject it as edge-truncated.
- If the target is ambiguous, small, partially visible, or the height cue is not reliable, return [] instead of guessing.
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output vehicles, robots, bags, backpacks, suitcases, strollers, wheelchairs without a clearly visible rider, or any static built-environment object.

Decision tree:
1. Check whether the target is a single recognizable dynamic human-centered instance from the candidate list. If not, reject.
2. Check scale-anchor visual reliability before assigning a class:
   - Reject tiny, far, blurry, low-contrast, or backlit people whose head/feet boundaries are only weakly visible.
   - Reject people embedded in a crowd when the body outline or legs merge with neighbors.
   - Reject people behind glass/reflections or in dark background regions when the full silhouette is uncertain.
   - Keep only people whose head, torso, two legs, and ground-contact region are all clear enough to support one clean mask.
   - Reject a person if SAM would likely mask only a thin silhouette, merge them with a neighbor, or miss the feet because the target is too small or dark.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground person
   middle = clearly visible mid-range person whose full body is still large and crisp
   distant = far-away background person, very small person, or weak silhouette
   - Reject every distant candidate. Do not output it even if no other anchor exists.
4. Check whether the top of the head and the feet/lower boundary are both identifiable with visible image evidence, so full height can still be inferred. If not, reject.
   - The feet/lower boundary must also be supported by the same flat ground plane as the robot. If the person stands on a stair tread, raised landing, curb, platform, ramp, furniture, or any elevated/offset surface, reject the target even when the full body is visible.
5. Check truncation first:
   - If the target is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - is_truncated is metadata, not a permission to keep a weak anchor.
   - Lateral truncation is acceptable only when the cut is minor, such as a small arm/shoulder/bag edge, and the torso, both legs, head, and ground-contact region remain reliable.
   - Reject any person whose body box lies against the left or right image edge unless the whole head, torso, both legs, and both feet are clearly inside the frame.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
   - Vertical truncation should be rejected when the head/top boundary or feet/lower boundary is not visible or inferable.
6. Check occlusion second:
   - Set is_occluded=true only when another physical object or person blocks part of the target body.
   - Glare, blur, shadow, motion blur, low contrast, or overexposure are not occlusion by themselves.
   - If another object blocks the head, torso, knees, feet, or most of the body contour, set is_occluded=true.
   - Never return severely occluded anchors. Severe occlusion means the head, torso, legs, feet, or most of the visible body is blocked.
   - Any height-direction occlusion is invalid, even if the class is recognizable.
   - Do not return occluded anchors. If is_occluded would be true because any object crosses the torso, legs, knees, feet, or floor contact, reject the target.
7. Keep / reject:
   - A clear near standing pedestrian should be returned only when the full head-to-feet height cue is reliable.
   - Do not reject a near standing person only because a small arm, bag, or shoulder edge is cut off.
   - Keep only upright standing or normally walking people whose feet touch a visible floor/ground region.
   - Mild truncation is allowed only when the body class remains recognizable, most of the body contour is visible, and the full vertical cue remains reliable.
   - Mild occlusion is allowed only when it does not block the head, torso, legs, or feet, and the full height cue remains reliable.
   - Reject medium/severe occlusion, medium/severe truncation, reflections, shadows, mannequins, posters, mirror images, and motion-blurred blobs.
   - Reject side-edge half bodies, cropped torsos, cropped legs, tiny blobs, partial limbs, upper-body-only silhouettes, lower-body-only silhouettes, and any person whose head/top or feet/bottom is cut by the image boundary.
   - Reject far courtyard/background pedestrians, people near windows/doorways whose feet blend into dark floor regions, and small people across a plaza or hallway.
   - Reject people standing behind fences, temporary barricades, railings, columns, door frames, counters, tables, chairs, or signs when those objects cross the body or hide floor contact.
   - Reject side-edge pedestrians in outdoor/campus scenes when part of the shoulder, arm, leg, or foot is clipped by the frame; these often create very small measured relative heights.
   - Reject a foreground person if only the side of the body is visible or the walking pose makes the two feet/ground contacts ambiguous.
   - Reject a candidate when the body is recognizable only because of context; the body outline itself must be clear.
   - In crowded groups, keep only people whose full body is visually separable from neighbors.
   - Prefer upright standing or walking people with reliable full-body cues. Reject sitting, crouching, kneeling, bending strongly, lying down, jumping, or upper-body-only people. Never infer standing height from a seated person.
   - If the target appears to be on stairs, a step, a raised landing/platform, a curb, a ramp, furniture, or any surface not coplanar with the robot's main walkable floor, reject it even when the full body is clearly visible from head to feet.
   - Use cyclist only when a visible rider and bicycle form one clear combined instance. Reject bicycle-only shapes, partial riders, and parked bicycles.
   - Do not invent fine-grained human subclasses such as child, student, staff, or person with backpack.
8. Tie-break:
   - Prefer near over middle, and middle over distant.
   - If two valid anchors are equally clear and equally complete, prefer the earlier class in the priority order below.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- You may use coarse visual prominence to reject small/distant weak anchors.
- Use only semantic category prior + common knowledge.
- Keep the height interval as precise as possible while remaining realistic.

Category priors:
- For each returned anchor, also output "level", "type_weight", and "priority".
- These three fields must STRICTLY match the class prior table below for the chosen object_class.
- Do not infer them from image appearance. Copy them exactly from the table.
- Dynamic class prior table:
{DYNAMIC_PRIOR_TABLE}

Dynamic-anchor priority:
- Prefer clearer and more standardized JRDB indoor dynamic anchors in this order:
  {DYNAMIC_PRIORITY_ORDER}
- Standing pedestrians are usually stronger anchors than generic person or cyclist.
- If two candidate anchors are equally clear and equally complete, prefer the earlier class in the above order.

Class disambiguation rules:
- If unsure between pedestrian and person, output pedestrian.
- Use person only when the image clearly shows a standing human but the more specific pedestrian label feels less certain.
- Use cyclist only when the rider and bicycle are both visible enough to form one recognizable instance.
- Reject cyclists or people at the left/right image edge when the rider/body is only partially inside the frame.
- If the person is small, dark, backlit, far away, or blended into a crowd, skip it unless the complete head, torso, legs, and feet are individually clear.
- Never use child as a reason for changing the height interval; if the human appears child-sized or age is uncertain, skip it unless the prompt's candidate classes and normal adult-height interval remain appropriate.
- Bags, backpacks, umbrellas, or carried items do not change the class.
- Do not invent subclasses beyond the candidate list.
- If multiple similar instances exist, keep only the clearest few valid full-body anchors and order clearer anchors first.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only the visible target body. Do not include neighboring people or large empty background.
- If a tight box would touch the top or bottom image boundary because the head or feet are missing, reject the anchor.

For each anchor, output exactly:
1) object_class (lowercase, exact label from candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must exactly match the class prior table)
6) type_weight (must exactly match the class prior table)
7) priority (must exactly match the class prior table)
8) confidence in [0,1]
9) body_pose (must be exactly one of: standing, walking, sitting, crouching, kneeling, bending, lying, jumping, cycling, unknown)
10) full_body_visible (boolean, true only when the full body from head to feet is visible enough for reliable height inference)
11) feet_visible (boolean, true only when both feet or the lower boundary / ground-contact region are clearly visible enough for reliable height inference)
12) floor_contact_clear (boolean, true only when the feet clearly touch the same stable flat walkable floor as the robot without fence/barrier/furniture/stair/step/raised-landing/platform/curb/ramp ambiguity)
13) reason (short common-sense justification)
14) is_occluded (boolean, true only if another physical object blocks part of the target)
15) is_truncated (boolean, true only if the target is cut off by the image edge)

Return JSON array only. Example:
[
  {{
    "object_class": "pedestrian",
    "object_bbox_2d": [310, 180, 470, 940],
    "height_interval": [1.60, 1.82],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.93,
    "body_pose": "standing",
    "full_body_visible": true,
    "feet_visible": true,
    "floor_contact_clear": true,
    "reason": "A clearly visible standing pedestrian usually has height around 1.6-1.8 m.",
    "is_occluded": false,
    "is_truncated": false
  }},
  {{
    "object_class": "cyclist",
    "object_bbox_2d": [620, 260, 820, 930],
    "height_interval": [1.45, 1.85],
    "distance_level": "middle",
    "level": "low",
    "type_weight": 0.55,
    "priority": 3,
    "confidence": 0.82,
    "body_pose": "cycling",
    "full_body_visible": true,
    "feet_visible": true,
    "floor_contact_clear": true,
    "reason": "A visible rider plus bicycle forms a recognizable cyclist anchor with a realistic combined height cue.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].
Candidate dynamic objects (use these exact lowercase labels only):
- pedestrian
- person
- cyclist
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify a small set of high-quality static height semantic anchors for Unitree G1 Locomanip / Humanoid-Everyday first-person scenes.

Scene context:
- Head-mounted D435, 640x480, moderate FOV. The robot often walks up to a fixture and then works at arm's length.
- Near desk work is common. Keep only freestanding tables, chairs, trash cans, stanchions/bollards, signboards, and real room doors.
- A black/silver robot gripper may enter from the bottom or side. A gripper touching the side of an object is not a reason to reject, if the object's own top and base remain visible.

Banned classes — never output these, even if they fill the frame:
- whiteboard, whiteboard stand
- counter, kitchen counter, sink, cabinet, locker, drawer
- fire extinguisher, fire extinguisher cabinet
- caution sign stand, A-frame warning sign
- window, elevator, elevator door, gate, push-to-open plate

Core objective:
- Return only the best high-quality static anchors. At most 2 static anchors per frame.
- Prefer objects with a standardized full height and a visible top plus base.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- height_interval must be copied exactly from the height interval table below. Never invent a number. Never assign a full-object interval to a fragment.
- If the visible crop is only a handle, button, tabletop, locker face, window glass, wall plate, or cabinet door, return [].
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output people, robot arms, hands, kettles, plates, bottles, footballs, toys, buttons, door handles, light switches, walls, stairs, or generic clutter.

Decision tree:
1. Check whether the object is a single recognizable static facility from the candidate list. If not, reject.
2. Check that the box covers that object's own top and base, not a fragment:
   - Door: keep only a real room door leaf whose top of the leaf/frame and the floor/threshold are both visible. Reject handle-only, lock-plate-only, wall-panel close-ups, locker doors, cabinet doors, elevator buttons, and PUSH TO OPEN plates. Do not label a wall or button panel as door.
   - Stanchion: keep a crowd-control post when post top and floor base are visible. The retractable belt may be present.
   - Table: keep when the tabletop and at least two legs/base on the floor are visible. Reject tabletop-only close-ups, legs-only floor crops, kitchen counters, and cabinets.
   - Chair: keep when the back top and legs on the floor are visible. Reject a cropped seat or armrest.
   - Trash can / bollard / signboard: keep only with visible top and floor base. Reject yellow A-frame caution signs; those are not signboard.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still large and crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate.
4. Check truncation:
   - If the object is cut by the image boundary, set is_truncated=true.
   - G1 close-ups may touch the image border. Keep the object when the missing sliver is small and the object's own top and base are still the box extremes.
   - Reject when the box is only a mid-body patch and the true top or base is clearly outside the frame.
5. Check occlusion:
   - Set is_occluded=true only when another object hides the target's top or base.
   - A robot gripper on the side or in front of a handle is not occlusion if top and base stay visible.
   - Reject when the gripper, arm, or another object hides the top or base.
6. Keep / reject:
   - Prefer door, stanchion, trash can, table, and chair when they meet the top-and-base test.
   - If nothing in the frame has a reliable top-to-base height, return [].
7. Tie-break:
   - Prefer near over middle.
   - If two valid anchors are equally clear, prefer the earlier class in the priority order below.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- Use only the class tables below plus common knowledge.
- Copy height_interval, level, type_weight, and priority from the tables. Do not infer them from appearance.

Category priors:
- Static class prior table:
{STATIC_PRIOR_TABLE}
- Static height interval table (copy exactly):
{STATIC_HEIGHT_INTERVAL_TABLE}

Class disambiguation and priority rules:
- If unsure between table and desk, output table.
- If the surface is a built-in kitchen/breakroom counter or cabinet, skip it. Do not output table.
- If unsure between bollard and stanchion, use stanchion for a queue post with a belt/cassette; use bollard for a permanent short protective post.
- If unsure between door and an empty wall opening, use door only when a real door leaf or door frame is visible.
- Prefer clearer anchors in this order:
  {STATIC_PRIORITY_ORDER}

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one object instance.
- The box must include that object's top and base. Do not box a handle, button, table edge, or locker label.

For each anchor, output exactly:
1) object_class (lowercase, exact label from candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats copied from the height interval table
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must exactly match the class prior table)
6) type_weight (must exactly match the class prior table)
7) priority (must exactly match the class prior table)
8) confidence in [0,1]
9) reason (short common-sense justification)
10) is_occluded (boolean)
11) is_truncated (boolean)

Return JSON array only. Example:
[
  {{
    "object_class": "door",
    "object_bbox_2d": [60, 40, 520, 990],
    "height_interval": [2.00, 2.10],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.90,
    "reason": "Near room door leaf with visible top and floor contact.",
    "is_occluded": false,
    "is_truncated": false
  }},
  {{
    "object_class": "trash can",
    "object_bbox_2d": [420, 450, 510, 860],
    "height_interval": [0.30, 0.70],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 4,
    "confidence": 0.86,
    "reason": "Upright trash can with visible top rim and floor base.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate static objects (use these exact lowercase labels only):
- door
- bollard
- stanchion
- trash can
- signboard
- table
- chair
""".strip()


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify the single best walkable plane element for JRDB indoor/campus robot first-person scenes.

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- JRDB includes indoor hallways and campus walkways. Prefer the broadest flat walkable floor, pavement, concrete, or tiled walkway near the robot.
- Use plane_type "floor" whenever the walkable flat surface is clearly visible, including outdoor campus pavement or walkway regions.
- Never choose stairs, stair treads, stair risers, handrails, tabletops, benches, counters, curbs, walls, grass, dirt, planters, or vegetation as the primary plane.
- If only a staircase is visible and there is no clear flat landing or floor region, return [].
- Draw a tight box that contains only the walkable plane surface.
- Exclude people, chairs, tables, signboards, obstacles, shadows, walls, and large non-plane objects.

Reasoning rules:
- Do not estimate orientation, depth, or scale.
- Use semantic category + common knowledge only.

Bounding box format:
- plane_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], (0,0)=top-left, (1000,1000)=bottom-right.

For each plane, output exactly:
1) plane_type (lowercase, e.g., floor)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence [0,1]
4) reason

Typical plane types:
- floor

Return JSON array only. Example:
[
  {
    "plane_type": "floor",
    "plane_bbox_2d": [0, 560, 1000, 1000],
    "confidence": 0.90,
    "reason": "The lower image region shows a broad flat indoor floor that is the main walkable surface."
  }
]
If none, return [].
""".strip()


def get_anchor_type_priors_v1(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt_v1(task_name: str, scene_domain: str = "indoor") -> str:
    scene_domain = str(scene_domain or "indoor").strip().lower()
    if scene_domain != "indoor":
        raise ValueError(
            f"JRDB prompt only supports scene_domain='indoor', but got: {scene_domain}"
        )
    if task_name == "dynamic_height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")


# ---------------------------------------------------------------------------
# Live static path: one closed list, same shape as DROID V2.
# Walk HUMANOID_ALLOWED_STATIC_ANCHORS one class at a time. No discovery pass.
# Scene plane is room floor only. Desk / table / counter / cabinet tops are
# not planes. Empty is a valid answer. v1 and the unused discovery helpers
# stay in this file for old runs.
# ---------------------------------------------------------------------------

# Bottle caps / kettle lids are a few centimetres. Peak-to-bottom uses that
# small interval, not the full bottle height.
STATIC_ANCHOR_MIN_HEIGHT_M = 0.015

HUMANOID_GRIPPER_HARDWARE_PRIOR = """
Humanoid-Everyday gripper prior (dataset-wide hardware, not a per-frame guess):
- The black/silver two-finger claw is the Unitree G1 gripper on the walking robot's wrist.
- It often enters from the bottom or side. Do not return the gripper, forearm, fingers, or a grasped object as a height anchor.
""".strip()


class HumanoidStaticAnchorSpec(TypedDict):
    level: str
    weight: float
    priority: int
    height_interval: tuple


# Classes that actually appear in Locomanip and can carry a standing height
# when the whole object is in view. Copy height_interval from this table.
HUMANOID_ALLOWED_STATIC_ANCHORS: Dict[str, HumanoidStaticAnchorSpec] = {
    # Large furniture and fixtures (very_high / high priority)
    "door": {"level": "very_high", "weight": 0.95, "priority": 1, "height_interval": (1.95, 2.15)},
    "stanchion": {"level": "high", "weight": 0.85, "priority": 2, "height_interval": (0.90, 1.00)},
    "bollard": {"level": "high", "weight": 0.85, "priority": 3, "height_interval": (0.80, 1.10)},
    "trash can": {"level": "high", "weight": 0.85, "priority": 4, "height_interval": (0.30, 0.70)},
    "cabinet": {"level": "high", "weight": 0.85, "priority": 7, "height_interval": (1.50, 2.10)},
    "locker": {"level": "high", "weight": 0.85, "priority": 8, "height_interval": (1.60, 2.00)},

    # Seating and medium furniture
    "chair": {"level": "medium", "weight": 0.70, "priority": 9, "height_interval": (0.75, 1.10)},
    "office chair": {"level": "medium", "weight": 0.70, "priority": 10, "height_interval": (0.90, 1.25)},
    "stool": {"level": "medium", "weight": 0.70, "priority": 11, "height_interval": (0.80, 1.00)},
    "armchair": {"level": "medium", "weight": 0.70, "priority": 12, "height_interval": (0.75, 1.10)},

    # Appliances and equipment
    "caution sign": {"level": "medium", "weight": 0.70, "priority": 13, "height_interval": (0.70, 0.95)},
    "coffee maker": {"level": "medium", "weight": 0.70, "priority": 14, "height_interval": (0.25, 0.40)},
    "coffee machine": {"level": "medium", "weight": 0.70, "priority": 15, "height_interval": (0.25, 0.40)},
    "microwave": {"level": "medium", "weight": 0.70, "priority": 16, "height_interval": (0.25, 0.40)},
    "fire extinguisher": {"level": "medium", "weight": 0.70, "priority": 17, "height_interval": (0.40, 0.70)},

    # Containers and bottles (medium priority)
    "kettle": {"level": "medium", "weight": 0.70, "priority": 18, "height_interval": (0.18, 0.28)},
    "electric kettle": {"level": "medium", "weight": 0.70, "priority": 19, "height_interval": (0.18, 0.28)},
    "boiling pot": {"level": "medium", "weight": 0.70, "priority": 20, "height_interval": (0.18, 0.28)},
    "shampoo": {"level": "medium", "weight": 0.70, "priority": 21, "height_interval": (0.15, 0.25)},
    "shampoo bottle": {"level": "medium", "weight": 0.70, "priority": 22, "height_interval": (0.15, 0.25)},

    # Small objects and tabletop items (low priority)
    "bottle": {"level": "low", "weight": 0.55, "priority": 23, "height_interval": (0.15, 0.25)},
    "cup": {"level": "low", "weight": 0.55, "priority": 24, "height_interval": (0.08, 0.12)},
    "mug": {"level": "low", "weight": 0.55, "priority": 25, "height_interval": (0.08, 0.12)},
    "plate": {"level": "low", "weight": 0.55, "priority": 26, "height_interval": (0.02, 0.04)},
    "american football": {"level": "low", "weight": 0.55, "priority": 27, "height_interval": (0.160, 0.180)},
    "tennis ball": {"level": "low", "weight": 0.55, "priority": 28, "height_interval": (0.065, 0.068)},
    "toilet paper roll": {"level": "low", "weight": 0.55, "priority": 29, "height_interval": (0.10, 0.12)},

    # Toys and small manipulation objects
    "dumpling toy": {"level": "low", "weight": 0.55, "priority": 30, "height_interval": (0.08, 0.12)},
    "egg toy": {"level": "low", "weight": 0.55, "priority": 31, "height_interval": (0.08, 0.12)},
    "cube": {"level": "low", "weight": 0.55, "priority": 32, "height_interval": (0.05, 0.10)},
    "toy cube": {"level": "low", "weight": 0.55, "priority": 33, "height_interval": (0.05, 0.10)},

    # Door hardware (when complete and visible)
    "door handle": {"level": "low", "weight": 0.55, "priority": 34, "height_interval": (0.10, 0.15)},
    "door lock": {"level": "low", "weight": 0.55, "priority": 35, "height_interval": (0.030, 0.045)},

    # Containers and organizers
    "magnetic container": {"level": "low", "weight": 0.55, "priority": 36, "height_interval": (0.06, 0.10)},
    "file tray": {"level": "low", "weight": 0.55, "priority": 37, "height_interval": (0.05, 0.08)},
    "pencil case": {"level": "low", "weight": 0.55, "priority": 38, "height_interval": (0.05, 0.08)},
    "basket": {"level": "low", "weight": 0.55, "priority": 39, "height_interval": (0.05, 0.12)},

    # Caps and lids (for top-view scenarios)
    "bottle cap": {"level": "low", "weight": 0.55, "priority": 40, "height_interval": (0.015, 0.030)},
    "kettle lid": {"level": "low", "weight": 0.55, "priority": 41, "height_interval": (0.015, 0.040)},
    "duster": {"level": "low", "weight": 0.55, "priority": 42, "height_interval": (0.050, 0.090)},
    "elevator button": {"level": "low", "weight": 0.55, "priority": 43, "height_interval": (0.020, 0.035)},
    "air pump": {"level": "low", "weight": 0.55, "priority": 44, "height_interval": (0.030, 0.055)},
    "caster wheel": {"level": "low", "weight": 0.55, "priority": 45, "height_interval": (0.040, 0.070)},
    "eraser": {"level": "low", "weight": 0.55, "priority": 46, "height_interval": (0.020, 0.045)},
    "stapler": {"level": "low", "weight": 0.55, "priority": 47, "height_interval": (0.035, 0.060)},
    "light switch": {"level": "low", "weight": 0.55, "priority": 48, "height_interval": (0.030, 0.055)},
}

# Spelling variants of allowed keys only. Unknown names stay unknown.
HUMANOID_ALLOWED_ALIASES: Dict[str, str] = {
    "trashcan": "trash can",
    "trashbin": "trash can",
    "trash bin": "trash can",
    "compost bin": "trash can",
    "recycling bin": "trash can",
    "coffeemaker": "coffee maker",
    "officechair": "office chair",
    "bottlecap": "bottle cap",
    "bottle top": "bottle cap",
    "kettlelid": "kettle lid",
    "elevator door": "door",
    "football": "american football",
    "gridiron football": "american football",
    "coffee cup": "cup",
    "water bottle": "bottle",
    "shampoo bottle": "shampoo",
    "toilet roll": "toilet paper roll",
    "tissue roll": "toilet paper roll",
    "dumpling": "dumpling toy",
    "bao": "dumpling toy",
    "xiaolongbao": "dumpling toy",
    "dumpling plush": "dumpling toy",
    "steamed bun": "dumpling toy",
    "steamed bun toy": "dumpling toy",
    "egg": "egg toy",
    "block": "cube",
    "toy block": "toy cube",
    "storage basket": "basket",
    "desk basket": "basket",
    "mesh basket": "basket",
    "organizer basket": "basket",
    "feather duster": "duster",
    "desk duster": "duster",
    "castor wheel": "caster wheel",
    "caster": "caster wheel",
    "castor": "caster wheel",
    "whiteboard caster": "caster wheel",
    "lift button": "elevator button",
    "elevator btn": "elevator button",
    "gate button": "elevator button",
    "access button": "elevator button",
    "accessibility button": "elevator button",
    "file organizer": "file tray",
    "paper tray": "file tray",
    "mesh tray": "file tray",
    "document tray": "file tray",
    "blackboard eraser": "eraser",
    "chalkboard eraser": "eraser",
    "desk eraser": "eraser",
    "desk stapler": "stapler",
    "desk switch": "light switch",
    "wall switch": "light switch",
    "ball pump": "air pump",
    "hand pump": "air pump",
    "inflation pump": "air pump",
    "keyhole": "door lock",
    "lock cylinder": "door lock",
    "cylinder lock": "door lock",
}


# Classes that should not be used as height anchors.
# These are known problematic categories from earlier runs.
HUMANOID_BAD_STATIC_ANCHORS: Dict[str, str] = {
    # Truly flat objects without meaningful height.
    # Do not list tray/container/button/elevator/gate here: those words sit
    # inside allowed names (file tray, magnetic container, elevator button).
    "paper": "flat sheet without standing height",

    # Soft/deformable objects
    "balloon": "soft and size-unstable",
    "bag": "soft and deformable",

    # Carried objects (whiteboard can be carried OR on stand)
    "whiteboard": "often carried or only shows flat face",

    # Tabletop furniture: head-cam almost never sees full top-to-floor height
    "table": "tabletop crop; 0.7-0.9 m standing height explodes scale",
    "desk": "tabletop crop; 0.7-0.9 m standing height explodes scale",

    # Structural elements without complete top-to-base
    "sink": "a recess, not a standing object",
    "window": "glass pane with incomplete frame",
    "drawer": "only a drawer face visible, not full height",

    # Handheld tools (when held, not standing)
    "shovel": "handheld tool, length ≠ standing height",

    # Soft toys (not standardized height)
    "doll": "soft toy, size not standardized",
    "plush": "soft toy",
    "stuffed toy": "soft toy",
}


HUMANOID_HALF_OBJECT_RULES = """
Half-object rule (applies to every class):
- Every object needs its own top AND its own base in the image. Furniture, appliances, containers, bottles, cups, plates, baskets, toys, and door hardware all follow this rule.
- A half chair, half table, half door, half kettle, half basket, or any other half object is not an anchor.
- Half-object means the object's own top or own base is out of the frame. The hidden back face does not make it half.
- A trash can whose lid/rim and floor contact are both visible is complete, even if only the front is seen.
- A thin nick of the lid or top rail at the image edge is not a half object when the floor/base is clearly visible. Keep that trash can or file tray.
- Reject when the base/floor contact is missing. Do not complete a missing half from memory.
""".strip()


def format_allowed_anchor_table() -> str:
    lines = []
    for class_name, spec in sorted(
        HUMANOID_ALLOWED_STATIC_ANCHORS.items(), key=lambda item: item[1]["priority"]
    ):
        lo, hi = spec["height_interval"]
        lines.append(
            f'  "{class_name}": {{"height_interval": [{lo:.3f}, {hi:.3f}], '
            f'"level": "{spec["level"]}", "type_weight": {spec["weight"]:.2f}, '
            f'"priority": {spec["priority"]}}}'
        )
    return "\n".join(lines)


def format_bad_anchor_table() -> str:
    return "\n".join(
        f'  "{class_name}": {reason}'
        for class_name, reason in HUMANOID_BAD_STATIC_ANCHORS.items()
    )


ALLOWED_STATIC_ANCHOR_TABLE = format_allowed_anchor_table()
BAD_STATIC_ANCHOR_TABLE = format_bad_anchor_table()
ALLOWED_STATIC_CANDIDATE_BULLETS = "\n".join(
    f"- {class_name}"
    for class_name, _ in sorted(
        HUMANOID_ALLOWED_STATIC_ANCHORS.items(),
        key=lambda item: item[1]["priority"],
    )
)
ALLOWED_STATIC_PRIORITY_ORDER = " > ".join(
    class_name
    for class_name, _ in sorted(
        HUMANOID_ALLOWED_STATIC_ANCHORS.items(),
        key=lambda item: item[1]["priority"],
    )
)


# One-pass closed list, same shape as DROID STATIC_HEIGHT_ANCHORS_PROMPT_V2.
# No discovery pass. The VLM walks this list and boxes at most two matches.
STATIC_HEIGHT_ANCHORS_PROMPT_CLOSED = f"""
You are a physical AI and computer vision expert.
Your task: identify high-quality static height semantic anchors for Humanoid-Everyday first-person scenes.

{HUMANOID_GRIPPER_HARDWARE_PRIOR}

Core objective:
- Walk the candidate list below, one class at a time. Keep a class only when that exact object is in the image and complete.
- Only use exact class names from the candidate list. Do not invent synonyms. Do not invent new classes.
- Return at most 2 static anchors per frame.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- A usable static anchor must be a single rigid physical object, isolated, and complete from top to base.
- Each returned anchor must represent exactly one physical object instance, never a group, stack, nested set, pair, or object-plus-contents region.
- Never return the same physical instance twice under different class names or with two different boxes.
- The only plane name is the walkable room floor. A desk, table, counter, or cabinet surface is not floor and is not a plane.
- Keep complete objects that stand on the room floor or sit on furniture. A kettle or basket on a desk is valid.
- Reject distant background room objects even if they are complete.
- Never return the gripper, robot arm, fingers, a person, or the floor itself.
- Never return an object the gripper is holding. A still-mounted door handle is not held just because the gripper is nearby.
- Treat truncation and occlusion as reject conditions. Do not return an anchor with is_truncated=true or is_occluded=true.
- If no listed class has a clean complete instance, return [].

{HUMANOID_HALF_OBJECT_RULES}

Known-bad classes. Never return these even if they look complete:
{BAD_STATIC_ANCHOR_TABLE}

Allowed height-anchor dictionary. Copy height_interval, level, type_weight, and priority exactly:
{ALLOWED_STATIC_ANCHOR_TABLE}

Decision tree:
1. Go through the candidate list in priority order. For each class, ask: is there one complete instance of this exact class in the image? If not, skip it.
2. Check scale-anchor visual reliability:
   - Reject extremely tiny objects where the top and base are not distinguishable.
   - Reject far, blurry, low-contrast, reflective, or backlit objects.
   - Reject the gripper, robot arm, fingers, and any grasped object.
   - Reject any object whose own top or own base is missing.
   - Reject distant background furniture or room objects even when their top-to-base extent is visible.
   - Keep objects whose top, sides/body, and base/bottom contact are visible enough for one clean SAM mask.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate.
4. Check whether the object height remains reliably observable from top to bottom. If not, reject.
5. Check truncation and half objects:
   - If the object is cut by the image boundary, set is_truncated=true and reject it.
   - If only the upper half or only the lower half is visible, reject it.
6. Check occlusion:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - If another object blocks the top, body, bottom, or base, reject.
   - A gripper touching the side of a still-grounded object is not occlusion if top and base stay clear.
7. Class-specific keep / reject:
   - door: keep a full door leaf with visible header/top and floor threshold.
   - stanchion / bollard: keep only when the post is complete from cap to floor.
   - trash can: keep a complete upright bin when the lid/rim and the floor contact are both visible while walking toward it. Keep it even if the lid nicks the top of the image. This is near or middle, not distant. Reject a close-up of only the opening.
   - table / desk: reject. A head-cam tabletop is not a full standing-height anchor.
   - cabinet / locker / chair / office chair / armchair: keep only a complete piece with top and legs/base.
   - stool: keep a standing bar/counter stool when the backrest top and the legs on the floor are both visible. Copy the stool interval (backrest top to floor, not seat-only height). Reject a top-down seat cushion. Reject it when the gripper is holding it. Reject it when the backrest is cut by the top of the image.
   - caution sign / coffee maker / microwave / fire extinguisher: keep only when the full body is visible from top to base.
   - kettle / electric kettle / boiling pot / bottle / shampoo / cup / mug / plate / basket: keep only a standalone instance with its own top and base. Reject a shampoo or bottle whose cap is cut by the top of the image. Reject a kettle cut by the left edge. Reject a close-up kettle that fills the frame without the lid nicking the top.
   - tennis ball / cube / toy cube / egg toy: keep only when the whole object is visible.
   - american football: keep a brown laced oval ball lying on the table with the whole ball visible. Copy the american-football interval (short-axis height on the table, not tip-to-tip length). Do not label it soccer ball. Reject the table. Reject it when the gripper is holding it.
   - dumpling toy: keep a white dumpling-shaped desk toy when the whole bun sits on the table. Copy the dumpling-toy interval. Do not label it plush, doll, egg toy, or ball. Reject the table. Reject it when the gripper is holding it.
   - toilet paper roll / pencil case: keep only when the whole object sits clear of clutter.
   - file tray: keep the mesh file tray when the tray floor and raised walls are visible. A thin nick of the back rail at the image edge is OK. Copy the file-tray interval. Reject the desk. Reject it when the gripper is holding it or when more than half the tray is out of frame.
   - magnetic container: keep the black magnetic box when it sits complete on the desk. A gripper nearby or touching a corner is not holding it. Reject it only when the gripper is closed on the box.
   - bottle cap / kettle lid: keep only when that small part is complete and not a crop of a larger object.
   - door handle: keep the metal lever when the full lever from the rose to the tip is visible. Copy the door-handle interval. Reject the door leaf, the cabinet, and a cropped stub. Reject it only when the gripper is closed on the lever. A gripper reaching toward the lever is not a reason to skip it.
   - door lock: keep the round lock cylinder / keyhole when the full circular face is visible. The box must sit on the metal cylinder, not on empty door, the lever, or the lock plate. Copy the door-lock interval (cylinder-face diameter, not the plate height). Reject the door leaf, the lever handle, a cropped hole, and empty door around the cylinder.
   - duster: keep a desk duster lying almost parallel on a surface when the fluffy head and handle are both visible inside the frame. Copy the duster interval (head thickness on the surface, not handle length). Reject the table and the cabinet body. Reject it when the gripper is holding it, when it is wiping a window, or when the head is cut by the top of the image.
   - caster wheel: keep one complete caster wheel plus its fork on a stand, cabinet, or locker. A caster under an open locker door counts. The box must sit on the black wheel, not on empty floor. Copy the caster-wheel interval. Reject the furniture body and a cropped wheel.
   - elevator button: keep one round wall or gate/access button when the full button face is visible. The round wheelchair button on a black gate post is elevator button. A gripper nearby is OK. Reject it only when the gripper covers the button. Reject a blank wall, a rectangular wall plate, the whole black post, or a whole elevator door.
   - eraser: keep the whole chalkboard eraser sitting on the desk. Copy the eraser interval (body thickness, not length). Reject the table. Reject it when the gripper is holding it.
   - stapler: keep the whole stapler sitting on the desk. Copy the stapler interval. Reject the table. Reject it when the gripper is holding it.
   - light switch: keep the complete switch assembly on the desk when the full body is visible. Copy the light-switch interval. Reject the table. Reject a wall plate cropped by the frame.
   - air pump: keep a white desk air pump lying as one complete object when the barrel, metal rod, and T-handle are all visible. Copy the air-pump interval (barrel diameter on the desk, not rod length). Reject the table. Reject it when the gripper is holding it or covering the barrel.
8. Tie-break:
   - Prefer the earlier class in the priority order below.
   - Prefer near over middle.
9. Final pairwise instance check:
   - Reject duplicate or nested boxes. If two boxes overlap, keep only the clearer one.
10. Box refine:
   - After the class is chosen, look at the image again and redraw object_bbox_2d.
   - The box must tightly wrap only that object's visible body.
   - If the first box sits on empty floor, empty desk, empty door, a drawer, or a neighboring object, discard it and draw a new tight box on the object.
   - If you cannot place a tight box that contains the object, drop that anchor.

Class priority order:
{ALLOWED_STATIC_PRIORITY_ORDER}

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- Use semantic category + common knowledge only.
- Copy height_interval, level, type_weight, and priority from the dictionary. Do not invent a number.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one complete object instance.
- After choosing the class, refine the box on the object pixels. Do not keep a first-guess box that missed the object.
- The box must cover only the chosen object's body. Exclude the support surface, neighboring clutter, and robot hardware.
- Two returned anchor boxes must not substantially overlap or contain one another.

For each anchor, output exactly:
1) object_class (lowercase, exact label from the candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must exactly match the class prior table)
6) type_weight (must exactly match the class prior table)
7) priority (must exactly match the class prior table)
8) confidence in [0,1]
9) reason (short common-sense justification)
10) is_occluded (boolean)
11) is_truncated (boolean)

Return JSON array only. Example:
[
  {{
    "object_class": "kettle",
    "object_bbox_2d": [410, 180, 560, 420],
    "height_interval": [0.18, 0.28],
    "distance_level": "near",
    "level": "medium",
    "type_weight": 0.70,
    "priority": 18,
    "confidence": 0.93,
    "reason": "Complete kettle on the desk with visible lid and base.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate static objects (use these exact lowercase labels only):
{ALLOWED_STATIC_CANDIDATE_BULLETS}
""".strip()

# Code-side safety net. The prompt does not dump this list; VLM answers that
# still use these words are dropped in VLMSemanticScaleAnchor.
# Note: Items in the allowed dictionary should NOT be in this blocklist.
STATIC_ANCHOR_BLOCKLIST_GROUPS: Dict[str, tuple] = {
    "recess": ("sink", "basin", "tub", "drain", "washbasin", "well"),
    "handheld tool when held": (
        "mop", "broom", "brush", "scoop", "trowel", "shovel",
        "syringe", "grease gun", "sprayer", "squeegee", "wand",
        "remote", "razor", "shaver",
        "caulk gun", "drill", "dryer", "plunger", "smartphone",
    ),
    "truly flat": (
        "mat", "rug", "carpet", "paper", "tissue", "towel",
        "cloth", "sheet", "board", "panel", "pad", "placemat",
        "tray",
    ),
    "soft deformable": (
        "cushion", "pillow", "bed", "pouch", "sponge", "fruit",
        "doll", "plush",
    ),
    "small parts": (
        "leg", "base", "frame", "door face", "rim", "edge", "pole",
        "post cap", "belt", "strap", "knob", "button", "switch", "outlet",
        "socket", "plate cover", "faucet", "tap", "spout", "hinge", "wheel",
    ),
    "structure": (
        "wall", "floor", "ceiling", "baseboard", "pavement", "tile", "column",
        "pillar", "partition", "window", "glass", "strip", "beam", "step",
        "stair", "curb", "ramp", "elevator", "gate",
    ),
    "too generic": (
        "fixture", "appliance", "device", "equipment", "object", "item",
        "unit", "thing", "stand", "holder", "organizer", "container",
        "accessory", "machine part", "tool", "rectangle", "reservoir",
    ),
    "robot or person": (
        "robot", "gripper", "finger", "arm", "hand", "person", "human",
        "operator", "microphone",
    ),
}

# Not hard-blocked. These are low-confidence anchors that may be named by VLM
# but should be used only when no better option exists.
STATIC_ANCHOR_LAST_RESORT_GROUPS: Dict[str, tuple] = {
    "last resort only": (
        "balloon",  # soft and unstable
    ),
}

STATIC_ANCHOR_BLOCKLIST: frozenset = frozenset(
    term for terms in STATIC_ANCHOR_BLOCKLIST_GROUPS.values() for term in terms
)
STATIC_ANCHOR_LAST_RESORT: frozenset = frozenset(
    term for terms in STATIC_ANCHOR_LAST_RESORT_GROUPS.values() for term in terms
)


def _normalized_class_name(class_name: str) -> str:
    return " ".join(
        str(class_name or "").strip().lower().replace("-", " ").replace("_", " ").split()
    )


def _label_hits_term(normalized: str, term: str) -> bool:
    term_n = " ".join(term.lower().replace("-", " ").split())
    if not term_n:
        return False
    if " " in term_n:
        return term_n in normalized
    return term_n in set(normalized.split())


def canonicalize_anchor_class(class_name: str) -> str:
    """Fold spelling variants onto the allowed dictionary key."""
    normalized = _normalized_class_name(class_name)
    return HUMANOID_ALLOWED_ALIASES.get(normalized, normalized)


def lookup_allowed_height_interval(class_name: str):
    spec = HUMANOID_ALLOWED_STATIC_ANCHORS.get(canonicalize_anchor_class(class_name))
    if spec is None:
        return None
    low, high = spec["height_interval"]
    return [float(low), float(high)]


def is_allowed_anchor_class(class_name: str) -> bool:
    return canonicalize_anchor_class(class_name) in HUMANOID_ALLOWED_STATIC_ANCHORS


def is_blocked_anchor_class(class_name: str) -> bool:
    """Ban known-bad and structural labels. Allowed dictionary keys pass."""
    canonical = canonicalize_anchor_class(class_name)
    if not canonical:
        return True
    if is_allowed_anchor_class(canonical):
        return False
    if canonical in HUMANOID_BAD_STATIC_ANCHORS:
        return True
    normalized = _normalized_class_name(class_name)
    if normalized in HUMANOID_BAD_STATIC_ANCHORS:
        return True
    return any(_label_hits_term(normalized, term) for term in STATIC_ANCHOR_BLOCKLIST)


def is_last_resort_anchor_class(class_name: str) -> bool:
    """Return True when the label is locker / extinguisher / ball-like."""
    normalized = _normalized_class_name(class_name)
    if not normalized:
        return False
    return any(_label_hits_term(normalized, term) for term in STATIC_ANCHOR_LAST_RESORT)


def format_candidate_list(class_names: Sequence[str]) -> str:
    cleaned = []
    seen = set()
    for class_name in class_names:
        normalized = canonicalize_anchor_class(class_name)
        if not normalized or normalized in seen or is_blocked_anchor_class(normalized):
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        return "- none"
    return "\n".join(f"- {class_name}" for class_name in cleaned)


STATIC_ANCHOR_CATEGORY_DISCOVERY_PROMPT_V2 = f"""
You are a physical AI and computer vision expert.
Your task: describe this Humanoid-Everyday frame and list object categories that could be usable static height anchors.

{HUMANOID_GRIPPER_HARDWARE_PRIOR}

Scene meaning:
- Head-mounted D435 on a walking Unitree G1. A sequence walks across a room toward one fixture, then works at arm's length.
- The only plane name is "floor": the walkable room floor under the robot's feet.
- A desk, table, counter, cabinet, or tray surface is not floor and is not a plane name.
- List complete objects whether they stand on the room floor or sit on furniture. A kettle or basket on a desk is valid.
- Reject distant background room objects that are only weakly visible.

Preferred height-anchor dictionary. Use an exact key when it fits:
{ALLOWED_STATIC_ANCHOR_TABLE}

Known-bad classes. Never list these:
{BAD_STATIC_ANCHOR_TABLE}

{HUMANOID_HALF_OBJECT_RULES}

You must discover obvious anchors yourself:
- Look at the image first. If a rigid object is complete and you can see its top and its base, list it.
- Do not wait for the dictionary. A complete kettle, bottle, cup, plate, basket, box, bowl, pot, or similar object in the foreground is mandatory.
- Example: a black electric kettle on a white counter with lid and base visible -> list "kettle". Returning [] is wrong.
- Example: a rectangular mesh/plastic basket sitting fully on a table -> list "basket". Returning [] is wrong.
- If the name is not in the dictionary, still list a short lowercase English name.
- Empty list is allowed only when no complete rigid object is visible. Walking toward empty floor/wall can be empty. A close-up desk with a clear complete object cannot.

Return categories when they could be useful for metric scale:
- Prefer a dictionary key when it fits. Then add other obvious complete objects.
- Reject robot arms, grippers, hands, people, walls, the floor itself, and known-bad classes.
- Reject soft toys, doll heads, flat paper, and handheld tools.
- List one object category per entry. Do not emit compound categories.
- Return at most 12 categories, ordered by usefulness as height anchors (larger/more stable first).

Return JSON object only:
{{
  "scene_description": "short description of what the robot is doing in this frame",
  "manipulation_plane": "floor|unknown",
  "candidate_static_classes": ["category", "..."]
}}
If no useful categories exist, return {{"scene_description": "...", "manipulation_plane": "unknown", "candidate_static_classes": []}}.
Use manipulation_plane "floor" only when the walkable room floor is visible. Otherwise use "unknown". Never output desk top, table top, counter top, or cabinet top.
""".strip()


def build_static_height_anchors_prompt_v2(candidate_static_classes: Sequence[str]) -> str:
    candidate_list = format_candidate_list(candidate_static_classes)
    return f"""
You are a physical AI and computer vision expert.
Your task: identify high-quality static height semantic anchors for Humanoid-Everyday first-person scenes.

{HUMANOID_GRIPPER_HARDWARE_PRIOR}

Open-vocabulary setup:
- A previous VLM pass inspected this exact image and found the candidate categories below, including objects it discovered in this frame.
- Output these exact lowercase labels when the instance is complete.
- If you see an obvious complete object that pass 1 missed, you may add that short lowercase name.
- Never return the gripper, robot arm, fingers, or a grasped object.
- Never return a class from the known-bad table.
- A complete kettle / bottle / basket / plate must be kept when its top and base are both visible.

Candidate static objects:
{candidate_list}

Allowed height-anchor dictionary. If the class is in this table, copy height_interval, level, type_weight, and priority exactly:
{ALLOWED_STATIC_ANCHOR_TABLE}

Known-bad classes from earlier runs. Never return these:
{BAD_STATIC_ANCHOR_TABLE}

{HUMANOID_HALF_OBJECT_RULES}

Core objective:
- Return stable, complete static objects whose full top-to-base vertical extent is visible and meaningful for metric scale.
- Return at most 2 static anchors per frame.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- The only plane name is the walkable room floor. A desk, table, counter, or cabinet surface is not floor and is not a plane.
- Keep complete objects that stand on the room floor or sit on furniture. A kettle or basket on a desk is valid.
- Reject distant background room objects even if they are complete.
- A usable static anchor must be a single rigid physical object, isolated, and complete from top to base.
- Each returned anchor must represent exactly one physical object instance, never a group, stack, nested set, pair, or object-plus-contents region.
- Never return the same physical instance twice under different class names or with two different boxes.
- The half-object rule applies to every class: keep only when both the object's own top and its own base are visible.
- Furniture needs its top and its legs/base. A kettle, bottle, cup, plate, basket, toy, or door handle also needs its own top and base. No class is exempt.
- A gripper in the same frame is not occlusion unless it covers the chosen object's top or rim.
- Treat truncation and occlusion as reject conditions. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the object is ambiguous, distant, floating, wall-mounted without base, only a front panel, or only a half object, return [] instead of guessing.

Decision tree:
1. Check whether the object is one recognizable static object from the candidate list. If not, reject.
2. Check scale-anchor visual reliability:
   - Reject extremely tiny objects where the top and base are not distinguishable.
   - Reject far, blurry, low-contrast, reflective, or backlit objects.
   - Reject the gripper, robot arm, fingers, and any grasped object.
   - Reject any object whose own top or own base is missing.
   - Reject distant background furniture or room objects even when their top-to-base extent is visible.
   - Keep objects whose top, sides/body, and base/bottom contact are visible enough for one clean SAM mask.
   - Small objects (5-25cm) are acceptable if they are complete, upright, and both top and base are visible.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate.
4. Check whether the object height remains reliably observable from top to bottom. If not, reject.
5. Check truncation and half objects:
   - If the object is cut by the image boundary, set is_truncated=true.
   - If only the upper half or only the lower half is visible, reject it.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
6. Check occlusion:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - If another object blocks the top, body, bottom, or base, reject.
   - A gripper touching the side of a still-grounded object is not occlusion if top and base stay clear.
7. Tie-break:
   - When multiple valid anchors exist, prefer larger, more rigid, more standardized objects.
   - Prefer objects with a clearly visible top and base.
8. Final pairwise instance check before returning JSON:
   - Compare the two proposed anchor boxes if two anchors are selected.
   - Reject duplicate or nested proposals: one box must not substantially contain the other box.
   - If two objects overlap because they are stacked, nested, or visually merged, keep only the single clearer anchor.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- Use semantic category + common knowledge only.
- If the class is an exact key in the allowed dictionary, copy height_interval, level, type_weight, and priority from that table. Do not invent a number.
- If the class is not in the table, keep a realistic interval from common knowledge. Set level=low, type_weight=0.55, priority=99.

Category weights:
- For each returned anchor, output "level", "type_weight", and "priority".
- If the class is in the dictionary, copy those three fields. Otherwise use low / 0.55 / 99.
- type_weight must match level:
{WEIGHT_MAPPING_LINES}

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one complete object instance.
- The box must cover only the chosen object's body. Exclude the support surface, neighboring clutter, and robot hardware.
- Do not stretch the box to the image border if the object itself does not touch that border.
- Two returned anchor boxes must not substantially overlap or contain one another. If clean non-overlapping boxes are impossible, return only the best single anchor.

For each anchor, output exactly:
1) object_class (lowercase, exact label from candidate list)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) distance_level (must be exactly one of: near, middle, distant)
5) level (must be exactly one of: very_high, high, medium, low)
6) type_weight
7) priority
8) confidence in [0,1]
9) reason (short common-sense justification)
10) is_occluded (boolean, true only if another physical object blocks part of the object)
11) is_truncated (boolean, true only if the object is cut off by the image edge)

Return JSON array only. If none, return [].
""".strip()


SCENE_PLANES_PROMPT_V2 = f"""
You are a physical AI and computer vision expert.
Your task: identify the single best horizontal reference plane for Humanoid-Everyday first-person scenes.

{HUMANOID_GRIPPER_HARDWARE_PRIOR}

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- plane_type must be exactly "floor". The only valid plane is the walkable room floor.
- A desk, table, counter, cabinet, tray, or any raised furniture surface is not a plane. Do not output those names.
- If the walkable room floor is not visible, return []. Close-up desk/table work with no room floor must be [].
- If the walkable room floor is visible, it usually occupies the lower image and reaches the bottom edge.
- Never choose vertical cabinet fronts, walls, doors, robot arms, grippers, hands, chairs, object tops of a single kettle/bottle, stairs, or clutter as the primary plane.
- Reject a surface if it is too small, heavily covered by objects, strongly tilted, curved, transparent, reflective, or visually ambiguous.
- If the gripper, a tool, or a manipulated object covers the only visible floor patch, return []. Do not guess the surface behind it.
- If only a desk/table/counter or other raised surface is visible, return [].
- Draw a tight box that contains only the visible walkable floor.
- Exclude robot arms, grippers, task objects, furniture bodies, people, walls, and large non-plane objects from the box.

Reasoning rules:
- Do not estimate orientation, depth, or scale.
- Use semantic category + common knowledge only.

Bounding box format:
- plane_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], (0,0)=top-left, (1000,1000)=bottom-right.

For each plane, output exactly:
1) plane_type (must be exactly: floor)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence [0,1]
4) reason

Typical plane types:
- floor

Return JSON array only. Example when the room floor is visible:
[
  {{
    "plane_type": "floor",
    "plane_bbox_2d": [80, 520, 940, 980],
    "confidence": 0.90,
    "reason": "The lower image region shows the walkable room floor down to the bottom edge."
  }}
]
If only a desk/table/counter is visible, or the room floor is absent, return [].
If none, return [].
""".strip()


# Fallback only when the better historical run used a raised plane and the
# floor-only pass returned []. Never used as the default plane prompt.
SCENE_PLANES_PROMPT_RAISED = f"""
You are a physical AI and computer vision expert.
Your task: identify the single best horizontal reference plane for Humanoid-Everyday first-person scenes.

{HUMANOID_GRIPPER_HARDWARE_PRIOR}

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- Prefer the walkable room floor when it is visible. Then plane_type must be exactly "floor".
- If the walkable room floor is not visible, you may return one raised work surface.
  Then plane_type must be exactly one of: "counter top", "table top", "desk top".
- Never return a tray, cabinet door, kettle/bottle top, wall, chair seat, or object lid as a plane.
- Reject a surface if it is too small, heavily covered by objects, strongly tilted, curved, transparent, reflective, or visually ambiguous.
- If the gripper, a tool, or a manipulated object covers the only visible surface, return [].
- Draw a tight box that contains only the visible surface.
- Exclude robot arms, grippers, task objects, furniture bodies, people, and walls from the box.

Bounding box format:
- plane_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], (0,0)=top-left, (1000,1000)=bottom-right.

For each plane, output exactly:
1) plane_type (exactly one of: floor, counter top, table top, desk top)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence [0,1]
4) reason

Return JSON array only. If none, return [].
""".strip()


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return {
            class_name: {
                "level": spec["level"],
                "weight": spec["weight"],
                "priority": spec["priority"],
            }
            for class_name, spec in HUMANOID_ALLOWED_STATIC_ANCHORS.items()
        }
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


# Long closed-set prompts make Qwen miss or over-box. A second short
# grounding call with the native bbox_2d schema verifies the box.
# If this call returns nothing usable, the first-pass box is kept.
HUMANOID_BOX_REFINE_SPECS: Dict[str, str] = {
    "door": (
        "one complete door leaf from the header to the floor threshold. "
        "Do not box only a handle, lock, or a cropped door fragment."
    ),
    "stanchion": (
        "one complete stanchion post from the cap to the floor base. "
        "Do not box the belt, nearby posts, or empty floor."
    ),
    "bollard": (
        "one complete bollard from the cap to the floor base. "
        "Do not box empty floor or nearby posts."
    ),
    "trash can": (
        "the standing trash or recycling bin from lid to floor. Box it even "
        "if the lid nicks the top of the image. Do not box only the opening, "
        "the wall, or empty floor. Return [] only when the floor base is missing."
    ),
    "cabinet": (
        "one complete cabinet from the top to the floor base. "
        "Do not box only a door face, handle, or a cropped cabinet."
    ),
    "locker": (
        "one complete locker from the top to the floor base. "
        "Do not box only a door face or a cropped locker."
    ),
    "chair": (
        "one complete standing chair from the backrest top to the legs on "
        "the floor. Do not box only the seat, a cropped back, or the table."
    ),
    "office chair": (
        "one complete office chair from the backrest top to the base on "
        "the floor. Do not box only the seat or empty floor."
    ),
    "stool": (
        "one standing bar/counter stool from the backrest top to the legs "
        "on the floor. Do not box only the seat cushion or the table. "
        "Return [] when the backrest is cut by the top of the image."
    ),
    "armchair": (
        "one complete armchair from the backrest top to the legs on the "
        "floor. Do not box only the seat or a cropped arm."
    ),
    "caution sign": (
        "one complete standing caution sign from the top to the floor base. "
        "Do not box the wall or empty floor."
    ),
    "coffee maker": (
        "the complete coffee maker from its own top to its own base. "
        "Do not box the table, kettle, or empty desk."
    ),
    "coffee machine": (
        "the complete coffee machine from its own top to its own base. "
        "Do not box the table or nearby objects."
    ),
    "microwave": (
        "the complete microwave from its own top to its own base. "
        "Do not box the cabinet, table, or empty wall."
    ),
    "fire extinguisher": (
        "the complete fire extinguisher cylinder from the valve to the "
        "base. Do not box the cabinet door or the wall."
    ),
    "kettle": (
        "the complete kettle from the lid/handle top to its own base. "
        "Do not box the table, base plate, or empty desk. Return [] "
        "when the kettle is cut by the left edge or is a close-up fill."
    ),
    "electric kettle": (
        "the complete electric kettle from the lid/handle top to its own "
        "base. Do not box the table or the base plate. Return [] when "
        "the kettle is cut by the left edge or is a close-up fill."
    ),
    "boiling pot": (
        "the complete boiling pot from the lid top to its own base. "
        "Do not box the table, cup, or empty desk. Return [] when the "
        "pot is cut by the left edge or is a close-up fill."
    ),
    "shampoo": (
        "the complete shampoo bottle from the cap to its own base. "
        "Do not box the table, tray, or empty desk. Return [] when "
        "the cap is cut by the top of the image."
    ),
    "shampoo bottle": (
        "the complete shampoo bottle from the cap to its own base. "
        "Do not box the table, tray, or empty desk. Return [] when "
        "the cap is cut by the top of the image."
    ),
    "bottle": (
        "the complete bottle from the cap to its own base. "
        "Do not box the table, tray, or empty desk. Return [] when "
        "the cap is cut by the top of the image."
    ),
    "cup": (
        "the complete cup from the rim to its own base. "
        "Do not box the table, pot, or empty desk."
    ),
    "mug": (
        "the complete mug from the rim to its own base. "
        "Do not box the table or empty desk."
    ),
    "plate": (
        "the complete plate as one disk. Tightly wrap only the plate, "
        "not the table and not empty desk."
    ),
    "american football": (
        "the whole brown laced oval football on the table. Tightly wrap "
        "only the ball, not the table."
    ),
    "tennis ball": (
        "the whole tennis ball. Tightly wrap only the ball, not the "
        "table, tray, or empty desk."
    ),
    "toilet paper roll": (
        "the whole toilet paper roll standing or lying as one cylinder. "
        "Do not box the table or tray."
    ),
    "dumpling toy": (
        "the whole white dumpling-shaped desk toy. Tightly wrap only the bun, "
        "not the table and not empty desk."
    ),
    "egg toy": (
        "the whole egg-shaped desk toy. Tightly wrap only the toy, not "
        "the table and not empty desk."
    ),
    "cube": (
        "the whole cube. Tightly wrap only the cube, not the table or tray."
    ),
    "toy cube": (
        "the whole toy cube. Tightly wrap only the cube, not the table or tray."
    ),
    "door lock": (
        "the round metal lock cylinder / keyhole. Tightly wrap only the "
        "circular metal face, not the lever and not the lock plate."
    ),
    "door handle": (
        "the metal door lever on the cabinet or door. Tightly wrap the handle "
        "from the rose to the tip. Do not box the door leaf, the cabinet, "
        "empty wall, or the gripper."
    ),
    "magnetic container": (
        "a small solid black rectangular magnetic box with a white label, "
        "sitting on a desk. Return [] for a mesh tray, crumpled paper, a "
        "gripper, a locker, or an empty table."
    ),
    "file tray": (
        "a black wire-mesh document tray with visible mesh holes and raised "
        "walls. Return [] for a solid plastic/metal box, an empty desk, or a "
        "locker door. A thin nick of the back rail at the image edge is OK."
    ),
    "pencil case": (
        "the whole pencil case. Tightly wrap only the case, not the "
        "table, tray, or empty desk."
    ),
    "basket": (
        "the whole basket from the rim to its own base. Do not box the "
        "table, tray, or empty desk."
    ),
    "bottle cap": (
        "the complete bottle cap only. Tightly wrap the cap, not the "
        "bottle body and not the table."
    ),
    "kettle lid": (
        "the complete kettle lid only. Tightly wrap the lid, not the "
        "kettle body and not the table."
    ),
    "duster": (
        "the whole duster / feather duster as one object. Tightly wrap "
        "only the duster, not the table, cabinet, container, or gripper. "
        "Return [] if the fluffy head is cut by the top of the image."
    ),
    "elevator button": (
        "one round metal elevator or gate/access button, including the "
        "round wheelchair button on a black gate post. Return [] for a "
        "locker handle, a rectangular wall plate, the whole black post, "
        "a light switch, empty wall, or the gripper."
    ),
    "air pump": (
        "the white desk air pump lying as one object. Tightly wrap the barrel, "
        "metal rod, and T-handle, not the desk drawer and not empty table."
    ),
    "caster wheel": (
        "one black caster wheel plus its fork under a stand, cabinet, or locker. "
        "Return [] for a desk leg, gripper, or empty floor. The box must sit on the wheel."
    ),
    "eraser": (
        "the whole chalkboard eraser sitting on the desk. Tightly wrap only "
        "the eraser, not the table, basket, or gripper."
    ),
    "stapler": (
        "the whole stapler sitting on the desk. Tightly wrap only the "
        "stapler, not the table, file tray, or gripper."
    ),
    "light switch": (
        "the complete desk light-switch assembly. Tightly wrap only the "
        "switch body, not the table and not the gripper."
    ),
}


# Closed-list first pass often returns [] on a complete trash can / file tray
# because the long prompt treats a lid nick as half-object. These short
# grounding classes are tried only when that pass is empty. Never include
# door handle here.
HUMANOID_EMPTY_FRAME_GROUND_CLASSES: tuple = (
    "trash can",
    "magnetic container",
    "elevator button",
    "duster",
    "caster wheel",
    "file tray",
)


def get_empty_frame_ground_classes(prefer_classes: Optional[Sequence[str]] = None) -> tuple:
    seen = []
    for name in list(prefer_classes or []) + list(HUMANOID_EMPTY_FRAME_GROUND_CLASSES):
        canonical = canonicalize_anchor_class(name)
        if not is_allowed_anchor_class(canonical) or canonical in seen:
            continue
        seen.append(canonical)
    return tuple(seen)


BESTMIX_TEACHER_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "humanoid_everyday"
    / "legacy"
    / "bestmix_teacher.json"
)
_TEACHER_CACHE = None


def load_bestmix_teacher() -> dict:
    global _TEACHER_CACHE
    if _TEACHER_CACHE is None:
        if BESTMIX_TEACHER_PATH.is_file():
            _TEACHER_CACHE = json.loads(BESTMIX_TEACHER_PATH.read_text())
        else:
            _TEACHER_CACHE = {}
    return _TEACHER_CACHE


def teacher_for_image(image_path: str) -> dict:
    seq = Path(str(image_path or "")).parent.name
    rec = load_bestmix_teacher().get(seq)
    return rec if isinstance(rec, dict) else {}


def build_static_height_anchors_prompt_closed(
    prefer_classes: Optional[Sequence[str]] = None,
) -> str:
    prefer = []
    for name in prefer_classes or []:
        canonical = canonicalize_anchor_class(name)
        if is_allowed_anchor_class(canonical) and canonical not in prefer:
            prefer.append(canonical)
    if not prefer:
        return STATIC_HEIGHT_ANCHORS_PROMPT_CLOSED
    bullets = "\n".join(f"- {class_name}" for class_name in prefer)
    prefix = (
        "This sequence historically scaled better when these classes were used "
        "as height anchors, if they are actually visible and complete:\n"
        f"{bullets}\n"
        "Prefer a listed class over other allowed classes when both are complete. "
        "Still reject table/desk as height anchors. Still reject half objects. "
        "If none of these are visible, walk the full candidate list as usual."
    )
    return prefix + "\n\n" + STATIC_HEIGHT_ANCHORS_PROMPT_CLOSED


def get_box_refine_prompt(object_class: str) -> Optional[str]:
    raw = str(object_class or "").strip().lower()
    canonical = canonicalize_anchor_class(raw)
    if not canonical or is_blocked_anchor_class(canonical):
        return None
    if canonical not in HUMANOID_ALLOWED_STATIC_ANCHORS:
        return None
    description = HUMANOID_BOX_REFINE_SPECS.get(canonical)
    if not description:
        description = (
            f"the complete {canonical}. Tightly wrap only this one object "
            "from its own top to its own base. Do not box the table, desk, "
            "floor, wall, door leaf, cabinet body, or the gripper."
        )
    return (
        f"Locate {description} in this image.\n"
        "Return JSON only:\n"
        f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{canonical}"}}]\n'
        "Coordinates are integers in [0,1000]. (0,0) is top-left, "
        "(1000,1000) is bottom-right.\n"
        "If the target is missing or cut in half, return []. "
        "A thin nick at the image edge is OK when the object body is otherwise complete."
    )


def get_prompt(task_name: str, scene_domain: str = "indoor") -> str:
    scene_domain = str(scene_domain or "indoor").strip().lower()
    if scene_domain != "indoor":
        raise ValueError(
            f"humanoid_everyday prompt only supports scene_domain='indoor', "
            f"but got: {scene_domain}"
        )
    if task_name == "dynamic_height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT_CLOSED
    if task_name == "static_anchor_category_discovery":
        return STATIC_ANCHOR_CATEGORY_DISCOVERY_PROMPT_V2
    if task_name == "height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT_V2
    raise ValueError(f"Unsupported prompt task: {task_name}")


if __name__ == "__main__":
    static = get_prompt("static_height_anchors")
    plane = get_prompt("scene_planes")
    assert "Walk the candidate list below, one class at a time" in static
    assert "Only use exact class names from the candidate list" in static
    assert "Do not invent new classes" in static
    assert "You must discover obvious anchors yourself" not in static
    assert "If the class is not in the table, keep a realistic interval" not in static
    assert "The only plane name is the walkable room floor" in static
    assert "Half-object rule (applies to every class)" in static
    assert "Return at most 2 static anchors" in static
    assert "10. Box refine:" in static
    assert "- door" in static
    assert "- kettle" in static
    assert "- bottle cap" in static
    assert "- duster" in static
    assert "- chair" in static
    assert "- eraser" in static
    assert "- stapler" in static
    assert "- light switch" in static
    assert "- file tray" in static
    assert "- magnetic container" in static
    assert "- elevator button" in static
    assert "whole black post" in static
    assert "cut by the top of the image" in static
    assert "- gripper" not in static
    assert "desk top" not in static
    assert "table top" not in static
    assert "counter top" not in static
    assert "slightly downward" not in static
    assert "from above" not in static
    assert "looks down" not in static
    assert "[0.015, 0.030]" in static
    assert "If none, return []." in static
    assert "gripper" in plane.lower()
    assert "must be exactly: floor" in plane
    assert "desk top" not in plane
    assert "table top" not in plane
    assert "counter top" not in plane
    assert "If only a desk/table/counter is visible" in plane

    # Test blocklist - these should be blocked
    assert is_blocked_anchor_class("kitchen sink")
    assert not is_blocked_anchor_class("duster")
    assert is_blocked_anchor_class("tray")  # generic flat tray, not file tray
    assert is_blocked_anchor_class("container")  # generic, not magnetic container
    assert is_blocked_anchor_class("button")  # generic, not elevator button
    assert is_blocked_anchor_class("elevator")
    assert is_blocked_anchor_class("gate")
    assert is_blocked_anchor_class("doll")
    assert is_blocked_anchor_class("gripper")
    assert is_blocked_anchor_class("table")
    assert is_blocked_anchor_class("desk")

    # Allowed dictionary keys and aliases should NOT be blocked
    assert not is_blocked_anchor_class("cabinet")
    assert not is_blocked_anchor_class("fire extinguisher")
    assert not is_blocked_anchor_class("chair")
    assert not is_blocked_anchor_class("kettle")
    assert not is_blocked_anchor_class("bottle cap")
    assert not is_blocked_anchor_class("trashcan")  # maps to "trash can"
    assert not is_blocked_anchor_class("football")
    assert not is_blocked_anchor_class("american football")
    assert not is_blocked_anchor_class("plate")
    assert not is_blocked_anchor_class("cup")
    assert not is_blocked_anchor_class("door handle")
    assert not is_blocked_anchor_class("file tray")
    assert not is_blocked_anchor_class("magnetic container")
    assert not is_blocked_anchor_class("elevator button")
    assert not is_blocked_anchor_class("eraser")
    assert not is_blocked_anchor_class("stapler")
    assert not is_blocked_anchor_class("light switch")
    assert not is_blocked_anchor_class("cube")
    assert not is_blocked_anchor_class("shampoo")
    assert not is_blocked_anchor_class("basket")
    assert not is_blocked_anchor_class("duster")
    assert not is_blocked_anchor_class("feather duster")

    # Test aliases
    assert canonicalize_anchor_class("trashcan") == "trash can"
    assert canonicalize_anchor_class("bottlecap") == "bottle cap"
    assert canonicalize_anchor_class("recycling bin") == "trash can"
    assert canonicalize_anchor_class("football") == "american football"
    assert canonicalize_anchor_class("gridiron football") == "american football"
    assert canonicalize_anchor_class("dumpling") == "dumpling toy"
    assert canonicalize_anchor_class("feather duster") == "duster"
    assert canonicalize_anchor_class("gate button") == "elevator button"
    assert canonicalize_anchor_class("mesh tray") == "file tray"
    assert canonicalize_anchor_class("chalkboard eraser") == "eraser"
    assert "bbox_2d" in get_box_refine_prompt("door lock")
    assert "bbox_2d" in get_box_refine_prompt("caster wheel")
    assert "bbox_2d" in get_box_refine_prompt("dumpling toy")
    assert "bbox_2d" in get_box_refine_prompt("air pump")
    assert "bbox_2d" in get_box_refine_prompt("kettle")
    assert "bbox_2d" in get_box_refine_prompt("chair")
    assert "bbox_2d" in get_box_refine_prompt("file tray")
    assert "bbox_2d" in get_box_refine_prompt("eraser")
    assert "bbox_2d" in get_box_refine_prompt("stapler")
    assert "bbox_2d" in get_box_refine_prompt("light switch")
    assert get_box_refine_prompt("table") is None
    assert get_box_refine_prompt("desk") is None
    assert "trash can" in get_empty_frame_ground_classes()
    assert "duster" in get_empty_frame_ground_classes()
    assert "door handle" not in get_empty_frame_ground_classes()
    assert lookup_allowed_height_interval("bottle cap") == [0.015, 0.030]
    assert build_static_height_anchors_prompt_closed([]) == STATIC_HEIGHT_ANCHORS_PROMPT_CLOSED
    prefer_prompt = build_static_height_anchors_prompt_closed(["plate", "desk"])
    assert prefer_prompt.startswith("This sequence historically")
    assert "- plate" in prefer_prompt
    assert "- desk" not in prefer_prompt.split("walk the full candidate list")[0]
    grounded = get_empty_frame_ground_classes(["kettle", "desk"])
    assert grounded[0] == "kettle"
    assert "trash can" in grounded
    assert "desk" not in grounded
    plate_teacher = teacher_for_image("/tmp/Locomanip_clean_the_plate/000.jpg")
    assert "plate" in (plate_teacher.get("prefer_classes") or [])
    assert plate_teacher.get("allow_raised_plane") is True
    assert "counter top" in SCENE_PLANES_PROMPT_RAISED
    assert "desk top" in SCENE_PLANES_PROMPT_RAISED
    assert "counter top" not in plane

    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from metric_scale.models.vlm_semantic_scale_anchor import VLMSemanticScaleAnchor as _VLMBox

    _wh = (640, 480)
    _duster = {"object_class": "duster"}
    _button = {"object_class": "elevator button"}
    _stool = {"object_class": "stool"}
    _shampoo = {"object_class": "shampoo"}
    _kettle = {"object_class": "electric kettle"}
    _basket = {"object_class": "basket"}
    _handle = {"object_class": "door handle"}
    _lock = {"object_class": "door lock"}
    _ext = {"object_class": "fire extinguisher"}
    _cup = {"object_class": "cup"}
    _pour = (
        "data/official-data/humanoid_everyday/"
        "Locomanip_walk_towards_a_desk_and_pour_water_from_the_pot_to_a_cup/"
        "000050_camera1.jpg"
    )
    _on_base = (
        "data/official-data/humanoid_everyday/"
        "Locomanip_pick_up_a_kettle_place_it_on_base_and_turn_on/"
        "000050_camera1.jpg"
    )
    assert _VLMBox._is_unusable_humanoid_static_box(_duster, [228, 0, 518, 343], _wh)
    assert not _VLMBox._is_unusable_humanoid_static_box(_duster, [119, 120, 478, 257], _wh)
    assert _VLMBox._is_loose_humanoid_static_box(_duster, [115, 120, 639, 262], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(_button, [266, 0, 506, 470], _wh)
    assert _VLMBox.rejected_humanoid_static_box_stage(_button, [342, 173, 381, 211], _wh) is None
    assert _VLMBox._is_unusable_humanoid_static_box(_stool, [183, 0, 436, 352], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(_stool, [190, 55, 489, 429], _wh)
    assert _VLMBox.rejected_humanoid_static_box_stage(_stool, [196, 170, 570, 479], _wh) is None
    assert _VLMBox._is_unusable_humanoid_static_box(_shampoo, [101, 0, 157, 94], _wh)
    assert _VLMBox.rejected_humanoid_static_box_stage(_shampoo, [108, 41, 154, 208], _wh) is None
    assert _VLMBox._is_unusable_humanoid_static_box(_kettle, [0, 69, 236, 364], _wh)
    assert not _VLMBox._is_unusable_humanoid_static_box(_kettle, [292, 0, 588, 353], _wh)
    assert _VLMBox._is_loose_humanoid_static_box(_kettle, [112, 125, 467, 427], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(_handle, [395, 115, 438, 158], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(_lock, [395, 115, 438, 158], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(_ext, [221, 0, 554, 341], _wh)
    assert not _VLMBox._is_unusable_humanoid_static_box(_ext, [278, 40, 362, 400], _wh)
    assert _VLMBox._is_unusable_humanoid_static_box(
        _kettle, [200, 80, 400, 280], _wh, source=_pour
    )
    assert not _VLMBox._is_unusable_humanoid_static_box(
        _kettle, [200, 80, 400, 280], _wh, source=_on_base
    )
    assert not _VLMBox._is_unusable_humanoid_static_box(_kettle, [200, 80, 400, 280], _wh)
    assert not _VLMBox._is_unusable_humanoid_static_box(
        _cup, [200, 80, 280, 180], _wh, source=_pour
    )
    assert _VLMBox.rejected_humanoid_static_box_stage(
        _kettle, [200, 80, 400, 280], _wh, source=_pour
    ) == "unusable_humanoid_static_box"
    assert _VLMBox.rejected_humanoid_conflicting_class(_basket, {"shampoo", "basket"}) == "conflicting_desk_organizer"
    assert _VLMBox.rejected_humanoid_conflicting_class(_basket, {"basket"}) is None

    print("humanoid_everyday closed-list prompt self-check ok")
