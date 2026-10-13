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
    "pedestrian": {"level": "very_high", "weight": 0.95, "priority": 1},
    "person": {"level": "medium", "weight": 0.70, "priority": 2},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "bollard": {"level": "high", "weight": 0.85, "priority": 2},
    "trash can": {"level": "high", "weight": 0.85, "priority": 3},
    "signboard": {"level": "medium", "weight": 0.70, "priority": 4},
    "table": {"level": "low", "weight": 0.55, "priority": 5},
    "chair": {"level": "low", "weight": 0.55, "priority": 6},
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
Your task: identify a small set of high-quality dynamic height semantic anchors for JRDB indoor/campus robot first-person scenes.

Core objective:
- Return only the best high-quality upright human dynamic anchors. At most 3 dynamic anchors per frame.
- This is a JRDB human environment. Use only complete standing or normally walking people for scale.
- Sitting, crouching, kneeling, bending, lying, jumping, and cycling people may be recognized in the image, but must not be returned as scale anchors.
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
- Never output cyclists, riders on bicycles, bicycle-only shapes, vehicles, robots, bags, backpacks, suitcases, strollers, wheelchairs, seated people, or any static built-environment object.

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
   - A clipped bag or carried item edge does not count as body truncation. If any body part is cut by the image boundary, reject the target.
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
   - Do not treat a small bag or carried item edge as part of the required body extent.
   - Keep only upright standing or normally walking people whose feet touch a visible floor/ground region.
   - Reject any person whose body_pose would be sitting, crouching, kneeling, bending, lying, jumping, cycling, or unknown.
   - Reject all body truncation. Do not return an anchor with is_truncated=true.
   - Reject all body occlusion. Do not return an anchor with is_occluded=true.
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
   - Reject cyclists and bicycle riders even when the rider and bicycle form one clear combined instance.
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
- Standing pedestrians are usually stronger anchors than generic person.
- If two candidate anchors are equally clear and equally complete, prefer the earlier class in the above order.

Class disambiguation rules:
- If unsure between pedestrian and person, output pedestrian.
- Use person only when the image clearly shows a standing human but the more specific pedestrian label feels less certain.
- Reject cyclists, bicycle riders, and people at the left/right image edge when the body is only partially inside the frame.
- Reject seated people even if their full seated body is visible. Do not label a seated person as pedestrian/person for scale.
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
  }}
]
If none, return [].
Candidate dynamic objects (use these exact lowercase labels only):
- pedestrian
- person
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify a small set of high-quality static built-environment height semantic anchors for JRDB indoor/campus robot first-person scenes.

Core objective:
- Return only the best high-quality static anchors. At most 2 static anchors per frame.
- Prefer built-environment objects with relatively standardized full height and a visible base on the floor.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- A usable static scale anchor must be a single object, visually large, isolated, grounded, and complete from top to base.
- The object must sit on the same stable walkable floor as the robot. Reject objects mounted in or behind construction fencing, grass/dirt beds, curbs, ramps, stairs, temporary barricades, or raised landings.
- Prefer objects whose full vertical extent is meaningful for scale: top and base should both be visible or reliably inferable.
- Prefer full interior doors and upright trash cans over furniture. Use bollards, signboards, tables, and chairs only when they are clearly standardized, isolated, complete, and grounded.
- Treat truncation and occlusion as reject conditions for JRDB static scale anchors. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the object is ambiguous, small, distant, floating, hanging, wall-mounted, or lacks a reliable top-to-base height cue, return [] instead of guessing.
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output people, bicycles, vehicles, posters without visible support/base, walls, stairs, railings, temporary barricades, queue stanchions, traffic cones, thin sign poles, or generic clutter.

Decision tree:
1. Check whether the object is a single recognizable static facility from the candidate list. If not, reject.
2. Check scale-anchor visual reliability before assigning a class:
   - Reject tiny, far, blurry, low-contrast, reflective, or backlit objects.
   - Reject objects merged with furniture clusters, crowds, walls, glass reflections, or clutter.
   - Reject objects whose base/floor contact is hidden by people, chairs, tables, shadows, glare, or image crop.
   - Reject temporary construction signs, warning boards, plastic barricades, fence posts, safety mesh, tarp supports, and sign stands in planted or dirt areas; their dimensions and bases are not stable JRDB scale anchors.
   - Keep only objects whose top, sides/supports, and floor-contact base are visible enough for one clean mask.
   - Reject a static object if a tight mask would include a wall panel, poster surface, rail, rope, or neighboring object rather than one physical object body.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still large and crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate. Do not output it even if no other anchor exists.
4. Check whether the object height remains reliably observable from top to bottom. If not, reject.
5. Check truncation first:
   - If the object is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - is_truncated is metadata, not a permission to keep a weak anchor.
   - Lateral truncation is acceptable only for a tiny side sliver when the complete top-to-base height is still visible.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
   - Vertical truncation should be rejected when the top or base is not visible or inferable.
6. Check occlusion second:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - Glare, blur, shadow, low contrast, or overexposure are not occlusion by themselves.
   - Never return severely occluded anchors. Severe occlusion means the top, base, or most of the visible object body is blocked.
   - Any height-direction occlusion is invalid, even if the object class is recognizable.
   - Do not return occluded anchors. If is_occluded would be true because another object blocks the top, support, body, or base, reject the target.
7. Keep / reject:
   - Prefer doors, bollards, trash cans, and rare freestanding signboards with clear top and floor-contact base.
   - For door, keep only a near or clean middle single standard door leaf or tight door-frame region whose top header and bottom threshold/floor contact are visible and not blocked by people, chairs, tables, counters, railings, or clutter. Reject far hallway doors, narrow background door slivers, glass reflections, partial door edges, open wall gaps, and doorways whose bottom is hidden.
   - For signboard, keep only a permanent freestanding floor sign/map/notice board whose panel, support, and base touch the same walkable floor as the robot. Reject wall-mounted plaques, menu posters, wall display panels, large wall-mounted advertisement boards, small warning/no-smoking signs, cropped sign panels, hanging signs, sign faces without support/base, thin sign poles, portable sign stands, temporary construction signs, warning boards, A-frame barricade signs, and far signs behind crowds.
   - For table, keep only a near single small standard table or desk whose tabletop and multiple legs/base are visible and isolated. Reject large conference tables, counters, front table edges, tabletops without legs, rows of cafe tables, far small tables, and desks merged with clutter.
   - For chair, keep only one near isolated empty chair with visible top/back, seat, legs, and floor contact. Reject occupied chairs, rows of chairs, stacked chairs, overlapping chairs, far small chairs, chair-like seat regions, backs, and any chair whose legs/base are hidden or cropped.
   - For trash can, keep only upright rigid bins with visible top rim/lid and base.
   - For bollard, keep only a permanent short rigid protective post with visible top and base on the same walkable floor as the robot; reject queue stanchions, rope/tape posts, portable sign stands, traffic cones, construction barriers, fence/barrier posts, railing posts, sign support poles, cropped posts, far tiny posts, and posts attached to clutter or railings.
   - Reject floating, hanging, or wall-mounted objects unless the full top-to-base height is clearly visible and physically grounded.
   - Mild truncation is allowed only for tiny side cuts that do not affect the top, base, legs, support, or floor contact.
   - Mild occlusion is allowed only when the object remains clearly identifiable and its height cue remains reliable.
   - Reject any static object whose top, base, legs, support, or floor contact is cropped by the image boundary, or whose visible appearance is only a partial seat/top surface without reliable height cue.
   - Reject medium or severe occlusion/truncation, merged multi-object clusters, repeated chair/table clusters, transparent reflections, stair structures, long handrails, counters without visible base, shelves fused into walls, plants, and decorative structures.
   - If the only available static candidates are far doors, far furniture, chair/table groups, wall signs, or uncertain short posts, return [].
   - If the object is a queue barrier/stanchion with a belt, rope, tape, or crowd-control rail, return [] even if the top and base are visible.
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
- Static class prior table:
{STATIC_PRIOR_TABLE}

Class disambiguation and priority rules:
- If unsure between table and desk, output table.
- If unsure between signboard and poster, use signboard only when a freestanding support and base are visible; otherwise skip it.
- If unsure between door and an empty wall opening, use door only when a real door leaf or full door-frame structure is visible.
- If unsure between bollard, queue stanchion, railing post, cone, or sign support post, skip it unless it is clearly a permanent short protective bollard.
- If unsure whether a table/chair is complete, isolated, and grounded, skip it.
- Prefer clearer and more standardized JRDB indoor static anchors in this order:
  {STATIC_PRIORITY_ORDER}
- If two candidate static anchors are equally clear and equally complete, prefer the earlier class in the above order.
- If many similar objects exist, keep only the clearest one or two valid anchors and order clearer anchors first.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one object instance. Do not include neighboring clutter, large empty background, or partial fragments.
- Do not box a table edge, chair back, wall sign panel, poster, or any fragment that lacks the object's full top-to-base extent.

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
    "object_class": "door",
    "object_bbox_2d": [60, 90, 230, 980],
    "height_interval": [2.00, 2.15],
    "distance_level": "near",
    "level": "very_high",
    "type_weight": 0.95,
    "priority": 1,
    "confidence": 0.91,
    "reason": "A standard interior door usually has height close to 2.0-2.1 m.",
    "is_occluded": false,
    "is_truncated": false
  }},
  {{
    "object_class": "trash can",
    "object_bbox_2d": [420, 450, 510, 860],
    "height_interval": [0.70, 1.00],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 3,
    "confidence": 0.86,
    "reason": "An upright trash can with visible top rim and base gives a more reliable static height cue than partial indoor furniture.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate static built-environment objects (use these exact lowercase labels only):
- door
- signboard
- bollard
- table
- chair
- trash can
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


def get_anchor_type_priors(task_name: str) -> Dict[str, AnchorTypePrior]:
    if task_name in {"dynamic_height_anchors", "height_anchors"}:
        return dict(DYNAMIC_ANCHOR_TYPE_PRIORS)
    if task_name == "static_height_anchors":
        return dict(STATIC_ANCHOR_TYPE_PRIORS)
    raise ValueError(f"Unsupported anchor prior task: {task_name}")


def get_prompt(task_name: str, scene_domain: str = "indoor") -> str:
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
