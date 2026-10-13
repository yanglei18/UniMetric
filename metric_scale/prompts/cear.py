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


# CEAR outdoor is quadruped campus/downtown: sidewalks, parking lots, building
# perimeters, and night downtown streets. Keep the taxonomy closed.
OUTDOOR_DYNAMIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "pedestrian": {"level": "very_high", "weight": 0.95, "priority": 1},
    "person": {"level": "high", "weight": 0.85, "priority": 2},
    "sedan": {"level": "high", "weight": 0.85, "priority": 3},
    "suv": {"level": "medium", "weight": 0.70, "priority": 4},
}


OUTDOOR_STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "traffic cone": {"level": "high", "weight": 0.85, "priority": 2},
    "bollard": {"level": "high", "weight": 0.85, "priority": 3},
    "trash can": {"level": "medium", "weight": 0.70, "priority": 4},
    "caution sign": {"level": "medium", "weight": 0.70, "priority": 5},
}


OUTDOOR_DYNAMIC_PRIOR_TABLE = _format_prior_table(OUTDOOR_DYNAMIC_ANCHOR_TYPE_PRIORS)
OUTDOOR_STATIC_PRIOR_TABLE = _format_prior_table(OUTDOOR_STATIC_ANCHOR_TYPE_PRIORS)
OUTDOOR_DYNAMIC_PRIORITY_ORDER = _format_priority_order(OUTDOOR_DYNAMIC_ANCHOR_TYPE_PRIORS)
OUTDOOR_STATIC_PRIORITY_ORDER = _format_priority_order(OUTDOOR_STATIC_ANCHOR_TYPE_PRIORS)


OUTDOOR_DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality dynamic height anchors only for CEAR outdoor sequences.

Dataset scope:
- CEAR outdoor is a quadruped walking campus sidewalks, building perimeters,
  parking lots, and downtown streets in day and night.
- Sequences: around_building, between_buildings, sidewalk1, parking_lot1/2,
  downtown1/2. Night runs are common. Many frames contain no usable person or vehicle.
- The dynamic taxonomy is fixed. Do not discover or invent another class.

Selection rules:
- Return at most 2 anchors. Prefer one excellent full-height instance over several weak ones.
- Priority when equally clear: {OUTDOOR_DYNAMIC_PRIORITY_ORDER}.
- Use pedestrian for a clearly standing or normally walking adult. Use person only when the human is clear but pedestrian is uncertain.
- Use sedan or suv only when that exact body type is visually clear. If vehicle type is uncertain, reject it; never output generic car, van, truck, or bus.
- A person must have a complete head-to-feet height cue, both feet or the lower ground-contact boundary visible, and a clean silhouette suitable for one SAM mask.
- A vehicle must have its roofline and wheel-ground lower boundary visible, with one clean body mask.
- Keep only near or clean middle-distance instances on the same visible ground as the camera platform.
- Reject distant tiny people across a plaza, distant parked rows, image-edge truncation, missing head/roof or feet/wheels, occlusion across the body, reflections through glass, posters, blur, glare, and merged groups.
- Treat night sequences strictly. If lighting hides the head, legs, feet, wheels, or ground contact, return [] rather than inferring them.
- Reject sitting people, crouching or bending people, people on bicycles/scooters, and upper-body-only people behind cars or building corners.
- A parked vehicle may still be a semantic height anchor; dynamic means movable object class, not that motion must be visible.
- Never output traffic cones, bollards, signs, trees, lamp posts, buildings, bicycles without riders, or the quadruped robot as dynamic anchors.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate height. Copy the fixed interval below.

Exact height intervals:
- pedestrian: [1.60, 1.82]
- person: [1.55, 1.85]
- sedan: [1.40, 1.55]
- suv: [1.60, 1.80]

Class priors (copy level, type_weight, and priority exactly):
{OUTDOOR_DYNAMIC_PRIOR_TABLE}

Bounding boxes:
- object_bbox_2d uses normalized integer coordinates [0,1000] as [x1,y1,x2,y2].
- Draw one tight box around one complete instance. Do not include neighbors or large empty regions.

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

For vehicles, set body_pose to unknown. full_body_visible means the full vehicle height is reliable;
feet_visible and floor_contact_clear mean the wheel/lower-body ground contact is reliable.
Every returned person must have body_pose standing or walking.
Returned anchors must have full_body_visible=true, feet_visible=true,
floor_contact_clear=true, is_occluded=false, and is_truncated=false.

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
    "reason": "Complete standing adult with clear head, feet, and ground contact.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate dynamic classes: pedestrian, person, sedan, suv.
""".strip()


OUTDOOR_STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Identify high-quality static height anchors only for CEAR outdoor sequences.

Dataset-specific candidate policy:
- Building-perimeter and downtown sequences repeatedly contain standard full-height exterior doors.
- Parking lots and sidewalks repeatedly contain traffic cones, short bollards, and upright trash cans.
- Folding caution signs appear occasionally near construction or wet pavement.
- Candidate classes are fixed from these sequences. Do not discover or invent another category.
- Empty sidewalk, grass verge, or distant parking frames should return [].

Selection rules:
- Return at most 2 anchors. Prefer one complete standardized object over several partial objects.
- Priority when equally clear: {OUTDOOR_STATIC_PRIORITY_ORDER}.
- Keep only a near or clean middle-distance single object with a reliable full top-to-base height and one clean SAM mask.
- The top, bottom, and ground-contact base must be visible. Returned anchors must have is_occluded=false and is_truncated=false.
- Reject distant, tiny, dark, blurred, backlit, overexposed, reflected, cropped, or partly hidden candidates.
- In night runs, reject an object whenever illumination hides its top, base, or boundary. Do not infer missing parts.
- door: require one normal exterior door leaf or frame with visible header and bottom threshold. Reject open wall gaps, glass-only storefronts, garage doors, cropped door strips, and distant facade doors.
- traffic cone: require one standard cone with tip and full base visible on pavement. Reject tiny distant cones, painted road markings, and cone-like reflections.
- bollard: require one permanent short protective post with top and base visible. Reject lamp posts, sign poles, tree supports, railing posts, queue stanchions, and hydrants.
- trash can: require one rigid upright outdoor bin with top rim/lid and base visible. Reject dumpsters, cardboard boxes, and soft bags.
- caution sign: require one complete freestanding folding yellow caution/wet-floor sign with top, two panels/legs, and ground contact visible. Reject flat signboards, posters, and traffic regulatory signs.
- Never select trees, lamp posts, benches, buildings, cars, curbs, fire hydrants, mailboxes, or arbitrary street furniture. Their dimensions or masks are not stable enough here.
- Do not use perspective, depth, pixels, camera height, or scene geometry to estimate height. Copy the fixed interval for the chosen class.

Exact height intervals:
- door: [2.00, 2.15]
- traffic cone: [0.65, 0.75]
- bollard: [0.75, 1.00]
- trash can: [0.65, 0.90]
- caution sign: [0.58, 0.65]

Class priors (copy level, type_weight, and priority exactly):
{OUTDOOR_STATIC_PRIOR_TABLE}

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
    "object_class": "traffic cone",
    "object_bbox_2d": [350, 390, 470, 900],
    "height_interval": [0.65, 0.75],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 2,
    "confidence": 0.93,
    "reason": "Complete traffic cone with visible tip and base on pavement.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If no candidate satisfies every rule, return [].
Candidate static classes: door, traffic cone, bollard, trash can, caution sign.
""".strip()


OUTDOOR_SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Identify the single best nearby walkable ground plane in a CEAR outdoor frame.

- Return at most 1 plane and use plane_type floor only.
- Prefer the broad coherent sidewalk, paved walkway, parking surface, or road
  directly supporting the camera platform.
- Reject curbs, stairs, ramps, grass clumps, walls, vehicle hoods, and distant surfaces.
- Night lighting is not permission to guess. Return [] if the ground region
  cannot be separated reliably.
- The box must mostly contain ground and exclude people, vehicles, cones, bollards,
  trash cans, glare, and large obstacles.
- Do not estimate orientation, depth, or scale.
- plane_bbox_2d is normalized [0,1000] as [x1,y1,x2,y2].

Output exactly: plane_type, plane_bbox_2d, confidence, reason.
Return a JSON array only. Example:
[
  {
    "plane_type": "floor",
    "plane_bbox_2d": [0, 560, 1000, 1000],
    "confidence": 0.92,
    "reason": "Broad unobstructed sidewalk supporting the camera route."
  }
]
If no reliable ground exists, return [].
Allowed plane type: floor.
""".strip()


def _normalize_scene_domain(scene_domain: str) -> str:
    normalized = str(scene_domain or "indoor").strip().lower()
    if normalized not in {"indoor", "outdoor"}:
        raise ValueError(
            f"CEAR prompt requires scene_domain 'indoor' or 'outdoor', got: {scene_domain}"
        )
    return normalized


def get_anchor_type_priors(
    task_name: str, scene_domain: str = "indoor"
) -> Dict[str, AnchorTypePrior]:
    domain = _normalize_scene_domain(scene_domain)
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        if domain == "outdoor":
            return dict(OUTDOOR_DYNAMIC_ANCHOR_TYPE_PRIORS)
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        if domain == "outdoor":
            return dict(OUTDOOR_STATIC_ANCHOR_TYPE_PRIORS)
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str, scene_domain: str = "indoor") -> str:
    domain = _normalize_scene_domain(scene_domain)
    if domain == "outdoor":
        if task_name in {"dynamic_height_anchors", "height_anchors"}:
            return OUTDOOR_DYNAMIC_HEIGHT_ANCHORS_PROMPT
        if task_name == "static_height_anchors":
            return OUTDOOR_STATIC_HEIGHT_ANCHORS_PROMPT
        if task_name == "scene_planes":
            return OUTDOOR_SCENE_PLANES_PROMPT
        raise ValueError(f"Unsupported prompt task: {task_name}")
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")


if __name__ == "__main__":
    indoor_dyn = get_anchor_type_priors("dynamic_height_anchors", "indoor")
    outdoor_dyn = get_anchor_type_priors("dynamic_height_anchors", "outdoor")
    assert "sedan" not in indoor_dyn
    assert outdoor_dyn["sedan"]["priority"] == 3
    assert "bollard" in get_anchor_type_priors("static_height_anchors", "outdoor")
    assert "CEAR outdoor" in get_prompt("dynamic_height_anchors", "outdoor")
    assert "plane_type floor only" in get_prompt("scene_planes", "outdoor")
    try:
        get_prompt("scene_planes", "mixed")
    except ValueError:
        pass
    else:
        raise SystemExit("expected ValueError for invalid scene_domain")
    print("cear prompt self-check ok")
