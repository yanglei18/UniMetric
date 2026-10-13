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


# CEAR indoor contains a fixed collection of rooms and repeated illumination runs.
# Keep the taxonomy closed to objects that recur in those sequences.
DYNAMIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "pedestrian": {"level": "very_high", "weight": 0.95, "priority": 1},
    "person": {"level": "high", "weight": 0.85, "priority": 2},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "table": {"level": "high", "weight": 0.85, "priority": 2},
    "chair": {"level": "high", "weight": 0.85, "priority": 3},
    "cabinet": {"level": "medium", "weight": 0.70, "priority": 4},
    "traffic cone": {"level": "medium", "weight": 0.70, "priority": 5},
    "caution sign": {"level": "medium", "weight": 0.70, "priority": 6},
    "trash can": {"level": "low", "weight": 0.55, "priority": 7},
}


def _format_prior_table(priors: Mapping[str, AnchorTypePrior]) -> str:
    lines = []
    for class_name, prior in sorted(priors.items(), key=lambda item: item[1]["priority"]):
        lines.append(
            f'  "{class_name}": {{"level": "{prior["level"]}", '
            f'"type_weight": {prior["weight"]:.2f}, "priority": {prior["priority"]}}}'
        )
    return "\n".join(lines)


def _format_priority_order(priors: Mapping[str, AnchorTypePrior]) -> str:
    return " > ".join(
        class_name
        for class_name, _ in sorted(priors.items(), key=lambda item: item[1]["priority"])
    )


DYNAMIC_PRIOR_TABLE = _format_prior_table(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIOR_TABLE = _format_prior_table(STATIC_ANCHOR_TYPE_PRIORS)
DYNAMIC_PRIORITY_ORDER = _format_priority_order(DYNAMIC_ANCHOR_TYPE_PRIORS)
STATIC_PRIORITY_ORDER = _format_priority_order(STATIC_ANCHOR_TYPE_PRIORS)


DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality dynamic height anchors only for CEAR indoor sequences.

Dataset scope:
- CEAR indoor contains corridors, classrooms, a dining hall, a kitchen/lounge,
  laboratories, a learning center, and repeated mocap-room runs under well-lit,
  blinking, HDR, and dark illumination.
- People occur in some classroom, laboratory, corridor, and mocap frames, but many
  frames contain no dynamic anchor. Returning [] is expected.
- The dynamic taxonomy is fixed. Do not discover or invent another class.

Selection rules:
- Return at most 2 anchors. Prefer one excellent full-body person over several weak people.
- Priority when equally clear: {DYNAMIC_PRIORITY_ORDER}.
- Use pedestrian for a clearly standing or normally walking adult. Use person only when the human is clear but pedestrian is uncertain.
- Require a complete head-to-feet height cue, both feet or the lower ground-contact boundary visible, and a clean silhouette suitable for one SAM mask.
- Keep only near or clean middle-distance people on the same visible floor as the camera platform.
- Reject sitting people, people using chairs, crouching or bending people, and upper-body-only people behind desks, counters, shelves, lab equipment, or room dividers.
- Reject image-edge truncation, missing head or feet, occlusion across the body, reflections through glass, mannequins, posters, blur, glare, and merged groups.
- Treat dark, blinking, and HDR sequences strictly. If lighting hides the head, legs, feet, or floor contact, return [] rather than inferring them.
- Never output chairs, robots, carts, boxes, traffic cones, signs, or other static objects as dynamic anchors.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate height. Copy the fixed interval below.

Exact height intervals:
- pedestrian: [1.60, 1.82]
- person: [1.55, 1.85]

Class priors (copy level, type_weight, and priority exactly):
{DYNAMIC_PRIOR_TABLE}

Bounding boxes:
- object_bbox_2d uses normalized integer coordinates [0,1000] as [x1,y1,x2,y2].
- Draw one tight box around one complete person. Do not include nearby furniture or empty regions.

For each anchor output exactly:
1) object_class
2) object_bbox_2d
3) height_interval
4) distance_level: near or middle
5) level
6) type_weight
7) priority
8) confidence in [0,1]
9) body_pose: standing, walking, sitting, crouching, kneeling, bending, lying, jumping, or unknown
10) full_body_visible
11) feet_visible
12) floor_contact_clear
13) reason
14) is_occluded
15) is_truncated

Every returned anchor must have body_pose standing or walking, full_body_visible=true,
feet_visible=true, floor_contact_clear=true, is_occluded=false, and is_truncated=false.

Return a JSON array only. Example:
[
  {{
    "object_class": "pedestrian",
    "object_bbox_2d": [350, 150, 500, 940],
    "height_interval": [1.60, 1.82],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.94,
    "body_pose": "standing",
    "full_body_visible": true,
    "feet_visible": true,
    "floor_contact_clear": true,
    "reason": "Complete standing adult with clear head, feet, and floor contact.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate dynamic classes: pedestrian, person.
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality static height anchors only for CEAR indoor sequences.

Dataset-specific candidate policy:
- Corridors repeatedly contain standard full-height doors and occasional trash cans.
- Classrooms repeatedly contain student tables/desks and chairs.
- Dining-hall, kitchen/lounge, and laboratory sequences repeatedly contain tables,
  chairs, floor-standing cabinets, and trash cans.
- Mocap sequences repeatedly contain tables, chairs, cabinets, and room doors.
- Mocap3 illumination runs repeatedly contain tall traffic cones and folding caution signs.
- Candidate classes are fixed from these sequences. Do not discover or invent another category.

Selection rules:
- Return at most 2 anchors. Prefer one complete standardized object over several partial objects.
- Priority when equally clear: {STATIC_PRIORITY_ORDER}.
- Keep only a near or clean middle-distance single object with a reliable full top-to-base height and one clean SAM mask.
- The top, bottom, supports, and floor-contact base must be visible. Returned anchors must have is_occluded=false and is_truncated=false.
- Reject distant, tiny, dark, blurred, backlit, overexposed, reflected, cropped, or partly hidden candidates.
- In blinking, HDR, and dark runs, reject an object whenever illumination hides its top, base, legs, or boundary. Do not infer missing parts.
- door: require one normal door leaf or frame with visible header and bottom threshold. Reject open wall gaps, glass-only openings, cropped door strips, and distant corridor doors.
- table: require one standard table or student desk with tabletop and legs/pedestal visible. Reject long counters, lab benches merged with equipment, rows of touching desks, and tabletops with hidden bases.
- chair: require one empty isolated chair with back, seat, legs/base, and floor contact visible. Reject occupied, nested, stacked, overlapping, or cropped chairs.
- cabinet: require one floor-standing cabinet with top and base visible. Reject wall cupboards, shelves full of objects, tool-cart fragments, and long merged cabinet rows.
- traffic cone: use only the tall pink/white floor cones recurring in mocap3 when tip and full base are visible. Reject tiny blue foam markers, toy blocks, partial cones, and cone-like reflections.
- caution sign: use only one complete freestanding folding yellow caution/wet-floor sign with top, two panels/legs, and floor contact visible. Reject flat signboards, whiteboards, posters, and sign faces without a base.
- trash can: require one rigid upright bin with top rim/lid and base visible. Reject boxes, carts, hampers, and soft bags.
- Never select cardboard boxes, foam blocks, bricks, shelves, couches, stools, computer towers, monitors, lab rigs, carts, room dividers, columns, fire equipment, or arbitrary machinery. Their dimensions or masks are not stable enough here.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate height. Copy the fixed interval for the chosen class.

Exact height intervals:
- door: [2.00, 2.15]
- table: [0.72, 0.78]
- chair: [0.80, 1.00]
- cabinet: [0.85, 1.10]
- traffic cone: [0.85, 0.95]
- caution sign: [0.58, 0.65]
- trash can: [0.60, 0.85]

Class priors (copy level, type_weight, and priority exactly):
{STATIC_PRIOR_TABLE}

Bounding boxes:
- object_bbox_2d uses normalized integer coordinates [0,1000] as [x1,y1,x2,y2].
- Draw one tight box around one complete physical instance.

For each anchor output exactly:
1) object_class
2) object_bbox_2d
3) height_interval
4) distance_level: near or middle
5) level
6) type_weight
7) priority
8) confidence in [0,1]
9) reason
10) is_occluded
11) is_truncated

Return a JSON array only. Example:
[
  {{
    "object_class": "caution sign",
    "object_bbox_2d": [350, 390, 470, 900],
    "height_interval": [0.58, 0.65],
    "distance_level": "near",
    "level": "medium",
    "type_weight": 0.70,
    "priority": 6,
    "confidence": 0.93,
    "reason": "Complete freestanding folding caution sign with visible top and feet.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate static classes: door, table, chair, cabinet, traffic cone, caution sign, trash can.
""".strip()


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Identify the single best nearby walkable floor plane in a CEAR indoor frame.

- Return at most 1 plane and use plane_type floor only.
- Prefer the broad coherent floor directly supporting the camera platform.
- Reject raised classroom platforms, dark carpets whose boundary is not visible,
  ramps, stairs, walls, tabletops, shelves, and reflections.
- Dark, blinking, or HDR lighting is not permission to guess. Return [] if the
  floor region cannot be separated reliably.
- The box must mostly contain floor and exclude people, furniture, cones, signs,
  boxes, glare, and large obstacles.
- Do not estimate orientation, depth, or scale.
- plane_bbox_2d is normalized [0,1000] as [x1,y1,x2,y2].

Output exactly: plane_type, plane_bbox_2d, confidence, reason.
Return a JSON array only. Example:
[
  {
    "plane_type": "floor",
    "plane_bbox_2d": [0, 560, 1000, 1000],
    "confidence": 0.92,
    "reason": "Broad unobstructed indoor floor supporting the camera route."
  }
]
If no reliable floor exists, return [].
Allowed plane type: floor.
""".strip()


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str, scene_domain: str = "indoor") -> str:
    normalized_domain = str(scene_domain or "indoor").strip().lower()
    if normalized_domain != "indoor":
        raise ValueError(
            f"CEAR indoor prompt requires scene_domain='indoor', got: {scene_domain}"
        )
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")
