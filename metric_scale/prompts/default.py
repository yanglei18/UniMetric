DYNAMIC_HEIGHT_ANCHORS_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify high-quality dynamic height semantic anchors with stable real-world height ranges.

CRITICAL rules (anti-hallucination):
- Only select CLEAR, SINGLE, INDIVIDUAL physical instances (not groups, shadows, ambiguous blobs).
- Do NOT infer objects that are not clearly visible as a distinct instance.
- Prefer the LARGEST and CLOSEST instances with clean boundaries.
- If unsure, SKIP (return fewer objects rather than guessing).
- Return all high-quality dynamic anchors.
- Only output classes from the dynamic candidate list below.
- Never output static traffic facilities such as traffic barriers, cones, curbs, guardrails, bollards, or poles.

Height completeness rules:
- Only output anchors that are complete along the height direction: the roofline/top and lower-body/ground-contact region must both be visible or reliably inferable.
- Treat image-boundary truncation as metadata, not an automatic rejection flag.
- Lateral truncation is acceptable when the top-to-bottom height cue remains reliable.
- Reject vertical truncation when the roofline/top boundary or lower-body/ground-contact boundary is not visible or inferable.
- Set is_occluded=true only when another physical object blocks part of the target body.
- Never return severely occluded anchors. Severe occlusion means the roofline/top, lower body, wheel-ground contact, or most of the visible body is blocked.
- Any height-direction occlusion is invalid, even if the object class is recognizable.
- Reject medium/severe occlusion, medium/severe truncation, crowded groups, distant tiny instances, reflections, shadows, and motion blur.
- Reject any target with missing or blocked top, bottom, lower body, wheel-ground contact, or only partial vertical extent.
- For pedestrians, keep only isolated standing people with reliable full-body height cues. Reject sitting, crouching, bending, riding, partially visible, or upper-body-only people.
- If multiple similar instances exist, keep all clearly valid ones and order clearer anchors first.

No geometry:
- Do NOT do geometric reasoning, measurement, perspective, or pixel-size-based scale estimation.
- Only use semantic category + common knowledge/standards for height interval.

Height interval precision:
- Use the MOST PRECISE possible height intervals based on common knowledge.
- Minimize the range while maintaining accuracy.

Bounding box format:
- object_bbox_2d must be NORMALIZED to [0,1000] range:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.

For each anchor, output exactly:
1) object_class (lowercase)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) confidence in [0,1]
5) reason (short common-sense justification)
6) is_occluded (boolean, true only if another physical object blocks part of the target)
7) is_truncated (boolean, true only if the target is cut off by the image edge)

Return JSON array only. Example:
[
  {
    "object_class": "sedan",
    "object_bbox_2d": [120, 520, 460, 880],
    "height_interval": [1.4, 1.5],
    "confidence": 0.92,
    "reason": "Standard sedans typically have height around 1.4-1.5 m.",
    "is_occluded": false,
    "is_truncated": false
  }
]
If none, return [].
Candidate dynamic objects:
- sedan
- suv
- van
- pickup truck
- bus
- heavy truck
- pedestrian
- cyclist
- motorcycle
""".strip()


STATIC_HEIGHT_ANCHORS_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify high-quality static traffic-facility height semantic anchors with stable real-world height ranges.

CRITICAL rules (anti-hallucination):
- Only select CLEAR, SINGLE, INDIVIDUAL physical instances (not groups, shadows, ambiguous blobs).
- Do NOT infer objects that are not clearly visible as a distinct instance.
- Prefer the LARGEST and CLOSEST instances with clean boundaries.
- If unsure, SKIP (return fewer objects rather than guessing).
- Return all high-quality static anchors.
- Only output classes from the static traffic-facility candidate list below.
- Never output dynamic targets such as sedans, SUVs, vans, trucks, buses, pedestrians, cyclists, or motorcycles.

Height completeness rules:
- Only output anchors that are complete along the height direction: the top and base must both be visible or reliably inferable.
- Treat image-boundary truncation as metadata, not an automatic rejection flag.
- Lateral truncation is acceptable when the visible facility still provides a reliable top-to-bottom height cue.
- Reject vertical truncation when the top or base is not visible or inferable.
- Set is_occluded=true only when another physical object blocks part of the facility body.
- Never return severely occluded anchors. Severe occlusion means the top, base, or most of the visible facility body is blocked.
- Any height-direction occlusion is invalid, even if the facility class is recognizable.
- Reject medium or severe occlusion/truncation, fragmented structures, merged facilities, and facilities whose base/top is ambiguous.
- Reject any facility with missing or blocked top, base, or only partial vertical extent.
- If many similar facilities exist, keep all clearly valid ones and order clearer anchors first.

No geometry:
- Do NOT do geometric reasoning, measurement, perspective, or pixel-size-based scale estimation.
- Only use semantic category + common knowledge/standards for height interval.

Height interval precision:
- Use the MOST PRECISE possible height intervals based on common knowledge.
- Minimize the range while maintaining accuracy.

Bounding box format:
- object_bbox_2d must be NORMALIZED to [0,1000] range:
  [x1, y1, x2, y2], where (0,0) is top-left and (1000,1000) is bottom-right.

For each anchor, output exactly:
1) object_class (lowercase)
2) object_bbox_2d (normalized [0,1000], integers)
3) height_interval (meters) as [min,max], floats
4) confidence in [0,1]
5) reason (short common-sense justification)
6) is_occluded (boolean, true only if another physical object blocks part of the facility)
7) is_truncated (boolean, true only if the facility is cut off by the image edge)

Return JSON array only. Example:
[
  {
    "object_class": "traffic barrier",
    "object_bbox_2d": [120, 520, 460, 880],
    "height_interval": [0.7, 1.2],
    "confidence": 0.92,
    "reason": "Standard traffic barriers typically have height around 0.7-1.2 m.",
    "is_occluded": false,
    "is_truncated": false
  }
]
If none, return [].

Candidate static traffic facilities:
- traffic cone
- traffic barrier
- curb
- guardrail
- bollard
- pole
""".strip()


SCENE_PLANES_PROMPT = """
You are a physical AI and computer vision expert.
Your task: identify potential semantic plane elements (few only).

CRITICAL rules:
- Do NOT hallucinate planes. Only output if the plane is plausibly present and visually coherent.
- Output at most 2 planes.
- Draw TIGHT bounding boxes that enclose ONLY the plane surface.
- EXCLUDE non-plane elements (e.g., cars, pedestrians, obstacles, vegetation) from the bounding box.
- Ensure the plane is the DOMINANT content within the bounding box.

No geometry:
- Do NOT estimate orientation, depth, or scale.
- Only semantic category + common knowledge.

Bounding box format:
- plane_bbox_2d must be NORMALIZED to [0,1000]:
  [x1, y1, x2, y2], (0,0)=top-left, (1000,1000)=bottom-right.

For each plane, output exactly:
1) plane_type (lowercase, e.g., road/floor/ground/tabletop)
2) plane_bbox_2d (normalized [0,1000], integers)
3) confidence [0,1]
4) reason

Typical plane types:
- road
- ground

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
