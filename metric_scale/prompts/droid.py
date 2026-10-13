from typing import Dict, Mapping, Sequence, TypedDict


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
    "person": {"level": "medium", "weight": 0.70, "priority": 1},
}


STATIC_ANCHOR_TYPE_PRIORS: Dict[str, AnchorTypePrior] = {
    "door": {"level": "very_high", "weight": 0.95, "priority": 1},
    "counter": {"level": "high", "weight": 0.85, "priority": 2},
    "cabinet": {"level": "high", "weight": 0.85, "priority": 3},
    "drawer": {"level": "high", "weight": 0.85, "priority": 4},
    "dresser": {"level": "high", "weight": 0.85, "priority": 5},
    "table": {"level": "high", "weight": 0.85, "priority": 6},
    "workbench": {"level": "high", "weight": 0.85, "priority": 7},
    "toilet": {"level": "high", "weight": 0.85, "priority": 8},
    "chair": {"level": "medium", "weight": 0.70, "priority": 9},
    "stool": {"level": "medium", "weight": 0.70, "priority": 10},
    "hamper": {"level": "low", "weight": 0.55, "priority": 11},
    "trash can": {"level": "low", "weight": 0.55, "priority": 12},
    "toaster": {"level": "low", "weight": 0.55, "priority": 13},
    "box": {"level": "low", "weight": 0.55, "priority": 14},
    "bottle": {"level": "low", "weight": 0.55, "priority": 15},
    "cup": {"level": "low", "weight": 0.55, "priority": 16},
    "book": {"level": "low", "weight": 0.55, "priority": 17},
    "bowl": {"level": "low", "weight": 0.55, "priority": 18},
    "brick": {"level": "low", "weight": 0.55, "priority": 19},
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


def format_candidate_list(class_names: Sequence[str]) -> str:
    cleaned = []
    seen = set()
    for class_name in class_names:
        normalized = str(class_name or "").strip().lower().replace("-", " ").replace("_", " ")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        return "- none"
    return "\n".join(f"- {class_name}" for class_name in cleaned)


DYNAMIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify high-quality dynamic human height anchors for DROID indoor robot manipulation scenes.

Core objective:
- DROID is an indoor robot manipulation dataset from external/robot cameras in labs, offices, kitchens, and home-like rooms.
- DROID frames are usually dominated by robot arms, grippers, tables, workbenches, counters, cabinets, drawers, cups, bowls, blocks, cloths, cables, tools, and containers.
- People are usually not the manipulated object and are often absent, partial, seated, in the background, or occluded by furniture and robot hardware.
- Return [] unless a real complete standing or normally walking adult person is clearly visible.
- Return at most 1 dynamic anchor per frame.
- Prefer a clear near person over a middle person. Reject distant people.
- A usable dynamic scale anchor must be visually large, crisp, separable, and complete from head to feet.
- The head/top and both feet or the lower ground-contact boundary must be visible enough for one clean SAM mask.
- The person's feet must touch the same room floor used as the scene reference. Reject people standing on chairs, stools, steps, platforms, furniture, or behind counters/tables when the floor contact is hidden.
- Treat truncation and occlusion as reject conditions. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the target is ambiguous, small, partially visible, behind furniture, or height cue is unreliable, return [] instead of guessing.
- Only use exact class names from the candidate list below. Do not invent synonyms.
- Never output robot arms, grippers, hands/forearms, mannequins, posters, reflections, vehicles, bicycles, bags, carts, or any static object.

Decision tree:
1. Check whether the target is one real human instance from the candidate list. If not, reject.
2. Check visual reliability before assigning a class:
   - Reject tiny, far, blurry, backlit, low-contrast, or motion-blurred people.
   - Reject people whose outline merges with a robot arm, chair, table, cabinet, doorway, glass reflection, or another person.
   - Reject partial operators, visible hands/arms, upper-body-only people, seated people, crouching people, or people cropped by the frame.
   - Keep only a person whose head, torso, legs, and floor-contact region are clear enough for one clean mask.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground person
   middle = clearly visible mid-range person whose full body is still large and crisp
   distant = far-away background person, very small person, or weak silhouette
   - Reject every distant candidate.
4. Check full vertical extent:
   - Both top of head and lower feet/floor-contact boundary must be identifiable with visible image evidence.
   - If legs, feet, or floor contact are hidden by a table, workbench, counter, chair, cabinet, robot base, box, bin, or image crop, reject.
5. Check truncation:
   - If the target is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
6. Check occlusion:
   - Set is_occluded=true only when another physical object blocks part of the target.
   - If another object blocks the head, torso, legs, feet, or floor contact, reject.
   - Glare, shadow, blur, or low contrast are not occlusion by themselves, but they are reject conditions when height cue is weak.
7. Keep / reject:
   - Keep only upright standing or normally walking people with reliable full-body height cues.
   - Reject sitting, crouching, kneeling, bending strongly, lying, jumping, cycling, partial, reflected, poster, toy, or mannequin-like human shapes.
   - Reject people behind desks, counters, benches, cabinets, shelves, robot arms, boxes, trash cans, or clutter when full body or floor contact is not visible.
   - If multiple people exist, keep only the clearest complete one.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- You may use coarse visual prominence to reject small/distant weak anchors.
- Use only semantic category prior + common knowledge.
- Keep the height interval realistic for a visible adult person. Use a broad interval such as [1.55, 1.90] unless visual evidence supports a tighter ordinary-adult range.

Category priors:
- For each returned anchor, also output "level", "type_weight", and "priority".
- These three fields must STRICTLY match the class prior table below for the chosen object_class.
- Do not infer them from image appearance. Copy them exactly from the table.
- Dynamic class prior table:
{DYNAMIC_PRIOR_TABLE}

Dynamic-anchor priority:
- Prefer clearer and more standardized DROID dynamic anchors in this order:
  {DYNAMIC_PRIORITY_ORDER}

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only the visible target body. Do not include neighboring furniture, robot hardware, other people, or empty background.
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
12) floor_contact_clear (boolean, true only when the feet clearly touch the same stable flat room floor without furniture/platform/step ambiguity)
13) reason (short common-sense justification)
14) is_occluded (boolean, true only if another physical object blocks part of the target)
15) is_truncated (boolean, true only if the target is cut off by the image edge)

Return JSON array only. Example:
[
  {{
    "object_class": "person",
    "object_bbox_2d": [360, 120, 520, 950],
    "height_interval": [1.55, 1.90],
    "distance_level": "near",
    "level": "medium",
    "type_weight": 0.70,
    "priority": 1,
    "confidence": 0.91,
    "body_pose": "standing",
    "full_body_visible": true,
    "feet_visible": true,
    "floor_contact_clear": true,
    "reason": "A complete standing adult person with clear head-to-feet visibility gives a usable height cue.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate dynamic objects (use these exact lowercase labels only):
- person
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = f"""
You are a physical AI and computer vision expert.
Your task: identify high-quality static height semantic anchors for DROID indoor robot manipulation scenes.

Core objective:
- DROID is dominated by indoor manipulation around tables, workbenches, counters, drawers, dressers, toilets, hampers, cabinets, and small task objects.
- Return only large, stable, complete static objects whose full top-to-base vertical extent is visible and meaningful for metric scale.
- For DROID, prefer objects resting on the selected manipulation plane. The downstream "floor" label usually means the tabletop/workbench/counter support surface, not the room floor.
- Reject background room furniture that is not on the selected manipulation plane.
- Return at most 2 static anchors per frame.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- Prefer rigid task objects such as cups, bottles, bowls, boxes, toasters, books, and bricks when they are on the selected manipulation plane and complete enough to give a clean height cue.
- A usable static anchor must be a single physical object, visually large, isolated, grounded, and complete from top to base.
- Each returned anchor must represent exactly one physical object instance, never a group, stack, nested set, pair, or object-plus-contents region.
- Never return the same physical instance twice under different class names or with two different boxes.
- The object must stand on the room floor or be a full furniture/counter unit whose bottom/base and top are visible. A tabletop surface alone is a plane, not a height anchor.
- Reject soft or flat tabletop manipulation objects: cloths, towels, cables, tools, utensils, erasers, plates, lids, and other clutter. Small rigid objects like cups, bottles, bowls, boxes, toasters, books, and bricks are allowed only when they are standalone, upright, isolated, and fully visible enough for a clean height cue.
- Reject robot arms, grippers, robot bases, cameras, tripods, monitors, laptops, cables, fixtures, shelves fused into clutter, and generic lab equipment.
- Prefer objects whose full vertical extent is standardized enough for common-sense height: top and base should both be visible or reliably inferable from a complete body.
- Treat truncation and occlusion as reject conditions. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the object is ambiguous, tiny, distant, floating, wall-mounted without base, only a top surface, only a front panel, or lacks a reliable top-to-base cue, return [] instead of guessing.
- Only use exact class names from the candidate list below. Do not invent synonyms.

Decision tree:
1. Check whether the object is one recognizable static object from the candidate list. If not, reject.
2. Check scale-anchor visual reliability before assigning a class:
   - Reject tiny, far, blurry, low-contrast, reflective, or backlit objects.
   - Reject objects merged with robot hardware, shelves, drawers, clutter, other furniture, people, walls, or glass reflections.
   - Reject objects whose base/floor contact or bottom edge is hidden by tables, chairs, robot arms, bins, boxes, shadows, image crop, or clutter.
   - Keep only objects whose top, sides/supports/body, and base are visible enough for one clean SAM mask.
   - Reject a mask target if it would include a wall, tabletop-only surface, neighboring object, or robot hardware rather than one complete object body.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still large and crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate.
4. Check whether the object height remains reliably observable from top to bottom. If not, reject.
5. Check truncation:
   - If the object is cut by the image boundary, set is_truncated=true.
   - Image-boundary cut-off is truncation, not occlusion.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
   - Reject any candidate whose top, base, legs, supports, or floor-contact region are cropped.
6. Check occlusion:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - If another object blocks the top, body, supports, legs, bottom, or base/floor contact, reject.
   - Glare, blur, shadow, low contrast, or overexposure are not occlusion by themselves, but reject when the height cue becomes weak.
7. Class-specific keep / reject:
   - door: keep a full standard door leaf or full door-frame region with visible header/top and bottom threshold/floor contact. Reject open wall gaps, cropped door edges, far door slivers, glass reflections, or doors hidden by furniture.
   - counter: keep a complete kitchen/lab counter or counter cabinet with visible top and lower base/front reaching the floor. Reject countertop-only surfaces, sinks, appliances, partial front panels, and counters whose bottom is hidden.
   - cabinet: keep a complete cabinet, drawer unit, cart cabinet, or storage cabinet with visible top and bottom/base. Use a height interval matching the visible subtype: low drawer units are much shorter than tall cabinets. Reject a single drawer face, open drawer only, cropped cabinet fronts, shelves without a clear full body, or cluttered racks.
   - table: keep a single complete table/desk with tabletop and multiple legs/base visible. Reject tabletop-only surfaces, cropped table edges, large foreground surfaces without legs, and tables merged with clutter.
   - workbench: keep a complete lab/work bench with top and legs/base visible. Reject only the benchtop plane or bench surfaces whose bottom/supports are hidden.
   - chair: reject background chairs in DROID manipulation scenes unless the chair itself is the manipulated tabletop object, which is rare.
   - stool: reject background stools in DROID manipulation scenes unless the stool itself is the manipulated tabletop object, which is rare.
   - drawer: keep a full drawer unit or drawer stack with visible top and bottom/base. Reject a single drawer face, open drawer only, or cropped drawer fronts.
   - dresser: keep a complete dresser with the full body visible from top to base. Reject partial dressers, open drawers only, or cropped fronts.
   - toilet: keep one complete toilet with visible tank/top, seat/body, and base/floor contact. Reject cropped toilets or toilets hidden by furniture.
   - hamper: keep one rigid upright hamper or clothes hamper with visible rim/top and base. Reject soft bags, baskets without a clear base, or cluttered containers.
   - trash can: keep one upright rigid bin/trash can with visible top rim/lid and bottom/base. Reject bags, buckets, boxes, cups, small containers, or bins whose base is hidden.
   - toaster / box / bottle / cup / book / bowl / brick: keep only standalone rigid instances with visible top and base or rim; reject tiny, flattened, tilted, or clutter-merged instances.
   - If the only available candidates are cups, bowls, blocks, cloths, cables, robot parts, tabletop-only planes, partial furniture, cluttered shelves, or far background objects, return [].
8. Tie-break:
   - Prefer near over middle, and middle over distant.
   - If two valid anchors are equally clear and equally complete, prefer the earlier class in the priority order below.
9. Final pairwise instance check before returning JSON:
   - Compare every pair of proposed anchor boxes.
   - Reject duplicate or nested proposals: one box must not substantially contain another box.
   - Reject proposals with clear area overlap. Touching edges or small contact at object boundaries is acceptable; overlapping object bodies are not.
   - If two objects overlap because they are stacked, nested, held together, or visually merged, keep only the single clearer and more complete anchor.
   - This rule applies between object anchors. Normal contact between an object and its supporting tabletop/floor plane is expected and is not anchor-anchor overlap.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- You may use coarse visual prominence to reject small/distant weak anchors.
- Use only semantic category prior + common knowledge.
- Keep the height interval as precise as possible while remaining realistic for the visible object subtype.
- Typical height intervals:
  door: [1.95, 2.15]
  counter: [0.85, 1.05]
  low cabinet/drawer unit: [0.60, 1.10]
  tall cabinet: [1.50, 2.20]
  drawer unit: [0.60, 1.20]
  dresser: [0.80, 1.80]
  table or desk: [0.70, 0.90]
  workbench: [0.75, 1.05]
  toilet: [0.55, 0.90]
  chair full back height: [0.75, 1.10]
  stool: [0.45, 0.80]
  hamper: [0.45, 0.90]
  trash can: [0.45, 0.95]

Category priors:
- For each returned anchor, also output "level", "type_weight", and "priority".
- These three fields must STRICTLY match the class prior table below for the chosen object_class.
- Do not infer them from image appearance. Copy them exactly from the table.
- Static class prior table:
{STATIC_PRIOR_TABLE}

Class disambiguation and priority rules:
- If unsure between counter and tabletop, use counter only when a complete counter body/front and bottom/base are visible; otherwise skip it as a height anchor.
- If unsure between cabinet and drawer, use cabinet only when the complete cabinet/drawer unit body is visible from top to bottom; otherwise skip it.
- If unsure between table and workbench, use table for ordinary desks/tables and workbench for lab/industrial benches with visible supports/base.
- If unsure between chair and stool, use stool only when there is no backrest and the seat plus legs/base are complete; otherwise use chair.
- If unsure whether an object is complete, isolated, grounded, and height-standard enough, skip it.
- Prefer clearer and more standardized DROID indoor static anchors in this order:
  {STATIC_PRIORITY_ORDER}
- If two candidate static anchors are equally clear and equally complete, prefer the earlier class in the above order.
- If many similar objects exist, keep only the clearest one or two valid anchors and order clearer anchors first.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one complete object instance. Do not include neighboring clutter, robot arms, large empty background, or partial fragments.
- The box must cover only the chosen object's body. Exclude the support surface, objects inside a container, objects stacked on top, and nearby items.
- Two returned anchor boxes must not substantially overlap or contain one another. If clean non-overlapping boxes are impossible, return only the best single anchor.
- Do not box a tabletop-only surface, drawer face, chair back fragment, wall panel, poster, shelf clutter, or any fragment that lacks full top-to-base extent.

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
10) is_occluded (boolean, true only if another physical object blocks part of the object)
11) is_truncated (boolean, true only if the object is cut off by the image edge)

Return JSON array only. Example:
[
  {{
    "object_class": "counter",
    "object_bbox_2d": [70, 330, 520, 910],
    "height_interval": [0.85, 1.05],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 2,
    "confidence": 0.88,
    "reason": "A complete counter body with visible top and base gives a standard indoor furniture height cue.",
    "is_occluded": false,
    "is_truncated": false
  }},
  {{
    "object_class": "table",
    "object_bbox_2d": [220, 430, 760, 900],
    "height_interval": [0.70, 0.90],
    "distance_level": "near",
    "level": "high",
    "type_weight": 0.85,
    "priority": 4,
    "confidence": 0.84,
    "reason": "A complete table with visible top and legs/base has a common desk-height scale.",
    "is_occluded": false,
    "is_truncated": false
  }}
]
If none, return [].

Candidate static built-environment objects (use these exact lowercase labels only):
- door
- counter
- cabinet
- drawer
- dresser
- table
- workbench
- toilet
- chair
- stool
- hamper
- trash can
- toaster
- box
- bottle
- cup
- book
- bowl
- brick
""".strip()


STATIC_ANCHOR_CATEGORY_DISCOVERY_PROMPT = """
You are a physical AI and computer vision expert.
Your task: describe the DROID robot manipulation scene and list object categories that could be usable static height anchors.

Scene meaning:
- DROID images usually show a robot arm working on a tabletop, desk, workbench, counter, tray, cabinet top, or lab bench.
- Treat this support surface as the local manipulation plane. It may not be the room floor.
- In this project the downstream label "floor" means this selected floor-like manipulation plane, usually the tabletop/workbench/counter surface.
- Only list object categories for physical objects resting on, attached to, or immediately supported by this selected manipulation plane.
- Reject background room objects that are not on the manipulation plane, even if they are complete and easy to recognize.

Return categories only when they could be useful for metric scale:
- Keep rigid tabletop/work-surface objects with a visible top-to-base vertical extent: box, bottle, cup, mug, bowl, can, jar, block, brick, container, tray with raised walls, bin, basket, tool holder, small appliance, tabletop stand, clamp, fixture, vise, holder.
- Reject chairs, stools, desks, tables, workbenches, counters, cabinets, drawers, doors, wall boards, monitors, and room furniture unless the object itself is a small item physically sitting on the selected manipulation plane.
- Prefer common category names, lowercase singular nouns.
- List one object category per entry. Do not emit compound categories such as "cup and bowl", "box with blocks", "stacked containers", or "assorted objects".
- Do not list robot arms, grippers, hands, cables, cloths, towels, paper sheets, plates, lids, flat mats, shadows, walls, screens, reflections, labels, or amorphous clutter.
- Do not list categories that are only tiny, heavily occluded, cropped, or visually ambiguous in this image.
- Return at most 12 categories, ordered by usefulness as height anchors.

Return JSON object only:
{
  "scene_description": "short description of the manipulation scene",
  "manipulation_plane": "tabletop|workbench|counter|cabinet top|floor|unknown",
  "candidate_static_classes": ["category", "..."]
}
If no useful categories exist, return {"scene_description": "...", "manipulation_plane": "unknown", "candidate_static_classes": []}.
""".strip()


def build_static_height_anchors_prompt(candidate_static_classes: Sequence[str]) -> str:
    candidate_list = format_candidate_list(candidate_static_classes)
    return f"""
You are a physical AI and computer vision expert.
Your task: identify high-quality static height semantic anchors for DROID robot manipulation scenes.

Open-vocabulary setup:
- A previous VLM pass inspected this exact image and found the candidate categories below.
- You may output only these exact lowercase labels. Do not invent new labels in this second pass.
- If none of these categories has a clean complete instance, return [].

Candidate static objects:
{candidate_list}

Core objective:
- Return only large, stable, complete static objects whose full top-to-base vertical extent is visible and meaningful for metric scale.
- Return at most 2 static anchors per frame.
- Prefer near anchors first, then clean middle anchors. Do not return distant static anchors.
- DROID frames are centered on a manipulation plane: tabletop, desk surface, workbench top, counter top, cabinet top, tray surface, or visible room floor.
- The downstream "floor" label means this selected floor-like manipulation plane. For tabletop manipulation data, it usually means the tabletop/workbench/counter surface, not the room floor.
- Keep only objects physically resting on, attached to, or immediately supported by this selected manipulation plane.
- Reject objects that are on the room floor, behind the table, beside the table, or part of the background room, even if they are complete.
- A usable static anchor must be a single rigid physical object, visually large, isolated, and complete from top to base.
- Each returned anchor must represent exactly one physical object instance, never a group, stack, nested set, pair, or object-plus-contents region.
- Never return the same physical instance twice under different class names or with two different boxes.
- Reject soft, flat, flexible, or thickness-poor objects: cloths, towels, cables, paper sheets, flat plates, lids, mats, and amorphous clutter.
- Reject robot arms, grippers, robot bases, hands, cameras, tripods, monitors, laptops, cables, wall surfaces, reflections, background chairs, background desks, background tables, background cabinets, room doors, whiteboards, and room furniture.
- Treat truncation and occlusion as reject conditions. Do not return an anchor with is_truncated=true or is_occluded=true.
- If the object is ambiguous, tiny, distant, floating, wall-mounted without base, only a top surface, only a front panel, or lacks a reliable top-to-base cue, return [] instead of guessing.

Decision tree:
1. Check whether the object is one recognizable static object from the candidate list. If not, reject.
2. Check scale-anchor visual reliability before assigning a class:
   - Reject tiny, far, blurry, low-contrast, reflective, or backlit objects.
   - Reject objects merged with robot hardware, shelves, drawers, clutter, other furniture, people, walls, or glass reflections.
   - Reject any object whose bottom/contact point is not on the selected manipulation plane.
   - Reject background furniture or room objects even when their top-to-base extent is visible.
   - Keep only objects whose top/rim, sides/body, and base/bottom contact are visible enough for one clean SAM mask.
   - Reject a mask target if it would include the support plane, neighboring clutter, robot hardware, or multiple objects rather than one complete object body.
3. Check distance_level from coarse visual impression only:
   near = large and prominent foreground object
   middle = clearly visible mid-range object whose full top-to-base extent is still large and crisp
   distant = far-away background object, very small object, or weak silhouette
   - Reject every distant candidate.
4. Check whether the object height remains reliably observable from top to bottom. If not, reject.
5. Check truncation:
   - If the object is cut by the image boundary, set is_truncated=true.
   - Do not return truncated anchors. If is_truncated would be true, reject the target.
6. Check occlusion:
   - Set is_occluded=true only when another physical object blocks part of the target body.
   - If another object blocks the top, body, bottom, base, or contact with the support plane, reject.
7. Tie-break:
   - Prefer larger, more rigid, more complete, more standardized objects.
   - Prefer objects with clear contact on the selected manipulation plane.
8. Final pairwise instance check before returning JSON:
   - Compare the two proposed anchor boxes if two anchors are selected.
   - Reject duplicate or nested proposals: one box must not substantially contain the other box.
   - Reject proposals with clear area overlap. Touching edges or small contact at object boundaries is acceptable; overlapping object bodies are not.
   - If two objects overlap because they are stacked, nested, inside one another, held together, or visually merged, keep only the single clearer and more complete anchor.
   - This rule applies between object anchors. Normal contact between an object and its supporting tabletop/floor plane is expected and is not anchor-anchor overlap.

Reasoning rules:
- Do not estimate metric distance or use geometry, perspective, exact pixel size, depth, or measurement.
- Use semantic category + common knowledge only.
- Keep the height_interval realistic for the visible subtype. Use broad intervals when category size varies.
- Examples: cup/mug/can [0.08,0.18], bottle [0.12,0.35], bowl [0.06,0.18], box/container [0.05,0.45], block/brick [0.03,0.15], tray with raised walls [0.03,0.12], jar [0.08,0.25], holder/stand/fixture [0.05,0.40], small appliance [0.10,0.35].

Category weights:
- For each returned anchor, output "level", "type_weight", and "priority".
- Assign level by anchor usefulness:
  very_high: large rigid object on the selected manipulation plane with standardized height
  high: complete rigid container, appliance, stand, holder, fixture, or bin on the selected manipulation plane with clear height
  medium: complete standalone bottle, mug, cup, can, bowl, jar, box, block, brick, tray with raised walls
  low: valid but size-variable rigid small objects
- type_weight must match level:
{WEIGHT_MAPPING_LINES}
- priority is 1 for the best returned anchor and 2 for the second returned anchor.

Bounding box format:
- object_bbox_2d must be normalized to [0,1000]:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.
- Draw a tight box around only one complete object instance. Do not include neighboring clutter, robot arms, support plane, or empty background.
- The box must cover only the chosen object's body. Exclude the support surface, objects inside a container, objects stacked on top, and nearby items.
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


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify the single best horizontal reference plane for DROID indoor robot manipulation scenes.

CRITICAL rules:
- Do not hallucinate planes. Output at most 1 plane.
- DROID frames often show the robot manipulating objects on a table, desk, workbench, lab bench, counter, or cabinet top, and the room floor may be absent.
- Prefer the broadest flat horizontal support surface near the robot that can define the local scene plane: tabletop, desk surface, workbench top, counter top, cabinet top, or visible room floor.
- Use plane_type "floor" for the selected primary horizontal reference plane. In DROID this label may represent the visible tabletop/workbench/counter support plane, because the downstream estimator expects a single floor-like horizontal reference.
- Never choose vertical cabinet fronts, walls, doors, robot arms, grippers, hands, object boxes, cups, bowls, cloths, cables, trays, shelves, monitor screens, chair seats, small tools, or clutter as the primary plane.
- Reject a surface if it is too small, heavily covered by objects, strongly tilted, curved, transparent, reflective, or visually ambiguous.
- If only small objects or vertical surfaces are visible and there is no broad flat support plane, return [].
- Draw a tight box that contains only the visible horizontal plane surface.
- Exclude robot arms, grippers, task objects, cups, bowls, blocks, cloths, tools, containers, chairs, cabinets fronts, shadows, walls, and large non-plane objects from the box.

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

Return JSON array only. Example:
[
  {
    "plane_type": "floor",
    "plane_bbox_2d": [80, 520, 940, 980],
    "confidence": 0.90,
    "reason": "The lower image region shows a broad flat tabletop/workbench support surface used as the DROID reference plane."
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
            f"DROID prompt only supports scene_domain='indoor', but got: {scene_domain}"
        )
    if task_name == "dynamic_height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_height_anchors":
        return STATIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "static_anchor_category_discovery":
        return STATIC_ANCHOR_CATEGORY_DISCOVERY_PROMPT
    if task_name == "height_anchors":
        return DYNAMIC_HEIGHT_ANCHORS_PROMPT
    if task_name == "scene_planes":
        return SCENE_PLANES_PROMPT
    raise ValueError(f"Unsupported prompt task: {task_name}")
