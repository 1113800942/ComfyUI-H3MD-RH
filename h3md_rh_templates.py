"""Plan templates for the RunningHub apps (H3MD RH).

Each template returns an H3MD plan v1 dict (see h3md_rh_core/h3md_director.py).  Users only fill a few fields
(who, scene, music start, shot lines); everything else is fixed here, tested on RH / locally (2026-11-06).

  mv            photo + song -> hard-cut shots (medium close-ups / close-ups) that fill the length cap (30 s: 3 shots
                of 226 frames = 9.4 s), the song locked as soundtrack, each shot at start + shot length * i seconds
  motion_copy   guide video + face photo + full-body photo -> one long shot that copies the guide's motion
                (TARA: global prompt "motion_transfer:", guide = <Video 1> encoded at short edge 480, its soundtrack
                locked).  The Director
                rounds the length down to 5+17n so it never runs past the clip end; autosplit cuts it into parts.
  story         photo + 1..8 shot lines -> one hard-cut shot per line, all shots shortened together to fit the
                length cap (at least 4.5 s each; lines that do not fit are dropped and named in the summary)

0.4.3-rh1: every template fits in MAX_SECONDS (30 s) and MAX_MEGAPIXELS (1.0) - the RunningHub limits.
"""
from __future__ import annotations

import math
import re

FPS = 24
SHOT = 243                      # 10.125 s, 5 + 17 * 14
SHOT_SEC = SHOT / FPS
HIRES_TRIGGER = "r64, r34l1sm"  # high-res LoRA trigger words used by the RH workflows
MAX_SECONDS = 30.0              # 0.4.3-rh1: whole video at most 30 s (RunningHub task time / cost)
MAX_MEGAPIXELS = 1.0            # 0.4.3-rh1: at most 1 MP per frame (1344x768 at 16:9)
MIN_SHOT = 107                  # 5 + 17 * 6 = 4.46 s: shortest shot a template makes when it shortens shots
GUIDE_HEIGHT = "480"            # 0.4.3-rh1: motion copy encodes the guide video at short edge 480 - the motion is
                                # set in the low-res pass; a 1080p guide filled a 32 GB card in the high-res pass
                                # (11-09 test: 1920x1088 guide at 1344x760 -> 106k fixed rows per strip, 12 strips)

TEMPLATES = ["mv", "motion_copy", "story", "custom"]
NOTES_KEY = "_rhNotes"          # template notes for the summary; removed by the Director before the plan is built

_CLEAN = re.compile(r'[\r\n\\"{}]+')


def clean(text: str, fallback: str = "") -> str:
    """User text -> one safe line (no newlines, quotes or braces that would break the prompt sections)."""
    t = _CLEAN.sub(" ", str(text or "")).strip()
    t = re.sub(r"\s{2,}", " ", t)
    return t or fallback


def frames_5_17n_down(frames: float) -> int:
    n = max(0, math.floor((float(frames) - 5) / 17))
    return 5 + 17 * n


def cap_seconds(max_seconds: float = 0.0) -> float:
    """User value -> seconds actually allowed: 0 (or empty) = MAX_SECONDS; anything above MAX_SECONDS is cut to it;
    below 5 s is raised to 5 s."""
    try:
        v = float(max_seconds or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    if v <= 0:
        return MAX_SECONDS
    return max(5.0, min(v, MAX_SECONDS))


def cap_frames(max_seconds: float = 0.0) -> int:
    """Frame budget (5+17n) for the whole video."""
    return frames_5_17n_down(cap_seconds(max_seconds) * FPS)


def shot_length(budget: int, count: int) -> int:
    """Length of each of `count` equal hard-cut shots that fit in `budget` frames (never longer than SHOT)."""
    return min(SHOT, frames_5_17n_down(budget / max(1, int(count))))


# ---------------------------------------------------------------------------------------------------------- MV
MV_STYLE = ("Photorealistic live-action music-video footage with natural skin texture, true-to-life colors, clear, "
            "even and bright natural light and crisp detail, in a clean frame with no subtitles, captions, overlay "
            "text, watermarks or logos. ")
MV_KEEP = ("The camera is locked off on a tripod and does not move at all; only <Subject 1>, the clothes and the hair "
           "move. One single continuous shot: the camera never cuts to another angle, and the distance between "
           "<Subject 1> and the camera stays the same. The face of <Subject 1> is large, sharp and clear in the frame "
           "and keeps the exact facial identity of <Picture 1> from every angle, with natural skin texture and a "
           "natural matte finish. <Subject 1> never laughs or grins widely and does not sing; the lips stay softly "
           "closed with a calm, soft closed-lip smile. The setting and the light stay the same from start to end. "
           "<Subject 1> is the only person in the whole picture: no other people anywhere, not even far away in the "
           "background. All motion plays at natural real-time speed, not slow motion.")
MV_SHOTS = [
    ("is seen in a calm medium close-up in {d}.",
     "[Shot 1] One continuous medium close-up from the front at eye level, framed from the chest up: <Subject 1> in "
     "{d}, with the surroundings softly out of focus behind. <Subject 1> looks slightly past the camera into the "
     "distance, breathes calmly, then slowly turns the eyes to the camera; a light breeze moves loose strands of hair."),
    ("closes the eyes and feels the music in a close-up in {d}.",
     "[Shot 1] One continuous close-up from the front at eye level: the face and shoulders of <Subject 1> fill most "
     "of the frame, with {d} softly blurred behind. <Subject 1> has the eyes gently closed as if feeling the music, "
     "then slowly opens them and looks softly into the distance."),
    ("sways gently to the music in a medium shot in {d}.",
     "[Shot 1] One continuous medium shot from the front, framed from the waist up: <Subject 1> in {d}, with the "
     "surroundings softly out of focus. <Subject 1> sways very gently from side to side with the rhythm of the music "
     "and rests one hand lightly on the chest for a moment."),
    ("is seen in a three-quarter close-up in {d}.",
     "[Shot 1] One continuous close-up from a three-quarter front angle at eye level: the face and shoulders of "
     "<Subject 1> fill most of the frame, with {d} softly blurred behind. <Subject 1> looks slightly to the side, "
     "blinks naturally and shows a faint, gentle smile."),
    ("brushes a strand of hair behind the ear in a medium close-up in {d}.",
     "[Shot 1] One continuous medium close-up from the front, framed from the chest up: <Subject 1> in {d}, with the "
     "surroundings softly out of focus. <Subject 1> slowly lifts one hand, gently tucks a strand of hair behind the "
     "ear, lowers the hand and looks into the camera."),
    ("looks out to the side in a medium shot in {d}.",
     "[Shot 1] One continuous medium shot from a slight side angle, framed from the waist up: <Subject 1> in {d}, "
     "with the surroundings softly out of focus. <Subject 1> looks out to the side across the view with a calm "
     "expression, then turns the face back toward the camera at a relaxed pace, without turning the body around."),
    ("looks warmly into the camera in a close-up in {d}.",
     "[Shot 1] One continuous close-up from the front at eye level: the face and shoulders of <Subject 1> fill most "
     "of the frame, with {d} softly blurred behind. <Subject 1> looks directly into the camera with a calm, warm "
     "expression, blinks naturally and slightly tilts the head once to the side and back."),
    ("is seen in a final medium close-up in {d}.",
     "[Shot 1] One continuous medium close-up from the front at eye level, framed from the chest up: <Subject 1> in "
     "{d}, with the surroundings softly out of focus. <Subject 1> gives a soft closed-lip smile to the camera, then "
     "looks slightly down and stays still and calm as the music goes on."),
]


def mv_shot_plan(max_seconds: float = 0.0):
    """(shot indices into MV_SHOTS, frames per shot) for the length cap: as many ~10 s shots as fit (at least one),
    shortened together so they fit; the last one is always the closing shot."""
    budget = cap_frames(max_seconds)
    count = max(1, min(len(MV_SHOTS), round(budget / SHOT)))
    length = shot_length(budget, count)
    idx = list(range(count - 1)) + [len(MV_SHOTS) - 1] if count > 1 else [0]
    return idx, length


def mv(who: str, scene: str, image_file: str, audio_file: str, music_start: float = 0.0,
       width: int = 1344, height: int = 768, max_seconds: float = 0.0) -> dict:
    who = clean(who, "a young woman")
    scene = clean(scene, "a sunny beach")
    idx, length = mv_shot_plan(max_seconds)
    segs = []
    for i, k in enumerate(idx):
        summ, desc = MV_SHOTS[k]
        prompt = ("subject_definitions:\n<Subject 1> is the person shown in <Picture 1>: " + who + ".\n"
                  "<Audio 1> is the song of the music video.\n\n"
                  "summary:\nreference generation + audio reuse. <Subject 1> " + summ.format(d=scene) +
                  " The appearance of <Subject 1> is taken only from <Picture 1>, and <Audio 1> is the soundtrack.\n\n"
                  "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - facial identity matching "
                  "<Picture 1> exactly, with the hair and clothing described above, identical in every moment.\n"
                  "<Audio 1> (appears in [Shot 1]): fully_copy - the song plays unchanged as the soundtrack from "
                  "start to end.\n\n"
                  "detailed_description:\n" + MV_STYLE + desc.format(d=scene) + " " + MV_KEEP + "\n\n"
                  "overall_soundscape:\nA soft, natural ambience of the place and the faint rustle of clothes.\n\n"
                  "non_diegetic_music:\n<Audio 1> plays unchanged as the soundtrack.")
        offset = round(float(music_start) + length / FPS * i, 3)
        segs.append({"length": length, "overlapIn": 0, "prompt": prompt, "images": ["img1"], "videos": [],
                     "audios": [{"id": "aud1", "mode": "locked", "offset": offset}], "audioMode": "reference"})
    return {"version": 1, "fps": FPS, "width": width, "height": height,
            "globalPrompt": HIRES_TRIGGER, "globalMode": "prepend",
            "media": {"images": [{"id": "img1", "file": image_file}], "videos": [],
                      "audios": [{"id": "aud1", "file": audio_file}], "refmods": []},
            "segments": segs, "refVideoHeight": "auto", "idRef": "img1"}


# ------------------------------------------------------------------------------------------------- motion copy
def motion_copy(who: str, scene: str, face_file: str, body_file: str | None, video_file: str,
                video_seconds: float, max_seconds: float = 0.0, width: int = 704, height: int = 1024) -> dict:
    """One long TARA shot.  Uses at most cap_seconds(max_seconds) of the guide (30 s; 0 = up to 30 s)."""
    who = clean(who, "a young woman with long dark hair, wearing a simple fitted dress")
    scene = clean(scene, "a cozy living room at night, lit by warm, soft lamps")
    use = min(float(video_seconds), cap_seconds(max_seconds))
    length = frames_5_17n_down(use * FPS)
    images = ["face"] + (["body"] if body_file else [])
    pics = ("<Picture 1> defines the face and identity, <Picture 2> the full figure and outfit"
            if body_file else "<Picture 1> defines the face, figure and outfit")
    ret = ("identity from <Picture 1>, figure and outfit from <Picture 2>" if body_file
           else "identity, figure and outfit from <Picture 1>")
    prompt = (f"Duration: {length / FPS:.2f} s ({length} frames).\n\n"
              "subject_definitions:\n<Subject 1> is the person shown in " +
              ("<Picture 1> and <Picture 2>" if body_file else "<Picture 1>") + f": {pics}. <Subject 1> is {who}. "
              "<Video 1> is the guide video: it supplies only the dance motion, poses, gestures, timing and camera "
              "framing; nothing of the guide performer's face, hair, body or outfit is kept.\n"
              "summary:\n[video generation + reference] Transfer the dance performance of <Video 1> to <Subject 1>, "
              "preserving the action order, timing, beat accents and framing of the guide, in a new setting.\n"
              f"retention_analysis:\n<Subject 1>: fully_preserved - {ret}, identical in every moment. "
              "<Video 1>: attribute_transfer - its motion and timing are transferred to <Subject 1>; its setting is "
              "replaced.\n"
              "detailed_description:\n[Shot 1] A locked-off shot matching the framing of <Video 1>. <Subject 1> "
              "follows the guide's dance exactly: every arm swing, body movement, step, hand gesture, head movement "
              "and expression in order and at matching times, each move landing on the beat. "
              f"<Subject 1> dances in {scene}; the place and the light stay the same from start to end, and "
              "<Subject 1> is the only person. The hands are empty with five natural fingers each. Do not copy the "
              "reference pictures' still pose or backdrop as the motion or setting.\n"
              "overall_soundscape:\nKeep the source soundtrack; do not invent dialogue.\n" + HIRES_TRIGGER + "\n\n"
              "non_diegetic_music:\nThe source song plays unchanged as the soundtrack.")
    media_images = [{"id": "face", "file": face_file}] + ([{"id": "body", "file": body_file}] if body_file else [])
    return {"version": 1, "fps": FPS, "width": width, "height": height,
            "globalPrompt": "motion_transfer:", "globalMode": "prepend",
            "media": {"images": media_images,
                      "videos": [{"id": "guide", "file": video_file, "duration": round(float(video_seconds), 3),
                                  "hasAudio": True}],
                      "audios": [], "refmods": []},
            "segments": [{"length": length, "overlapIn": 0, "prompt": prompt, "images": images,
                          "videos": [{"id": "guide", "trimStart": 0.0, "useSoundtrack": False, "depth": False,
                                      "useDuration": round(length / FPS, 6)}],
                          "audios": [], "audioMode": "locked", "locked": {"source": "video:guide", "offset": 0.0}}],
            "refVideoHeight": GUIDE_HEIGHT}


# ------------------------------------------------------------------------------------------------------- story
def story_shot_plan(line_count: int, max_seconds: float = 0.0):
    """(lines kept, frames per shot): all shots shortened together to fit the cap, none shorter than MIN_SHOT."""
    budget = cap_frames(max_seconds)
    keep = max(1, min(int(line_count), budget // MIN_SHOT))
    return keep, shot_length(budget, keep)


def story(who: str, scene: str, shots: str, image_file: str, width: int = 1344, height: int = 768,
          max_seconds: float = 0.0) -> dict:
    """One hard-cut shot per non-empty line of `shots` (max 8), all fitting in the length cap."""
    who = clean(who, "a young woman")
    scene = clean(scene, "")
    lines = [clean(x) for x in str(shots or "").splitlines()]
    lines = [x for x in lines if x][:8]
    if not lines:
        raise ValueError("H3MD RH story: write at least one shot line (one line per shot). "
                         "故事模板：请至少写一行镜头描述（一行一个镜头）。")
    keep, length = story_shot_plan(len(lines), max_seconds)
    notes = []
    if keep < len(lines):
        span = f"{keep + 1}" if keep + 1 == len(lines) else f"{keep + 1}-{len(lines)}"
        notes.append(f"story: {len(lines)} lines do not fit in {cap_seconds(max_seconds):g} s - kept the first {keep} "
                     f"(line {span} dropped). 故事：{len(lines)} 行放不进 {cap_seconds(max_seconds):g} "
                     f"秒，只用前 {keep} 行（第 {span} 行没用上）。")
        lines = lines[:keep]
    segs = []
    for line in lines:
        where = f" The scene: {scene}." if scene else ""
        prompt = ("subject_definitions:\n<Subject 1> is the person shown in <Picture 1>: " + who + ".\n\n"
                  "summary:\nreference generation. " + line + "\n\n"
                  "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - facial identity matching "
                  "<Picture 1> exactly, with the hair and clothing described above.\n\n"
                  "detailed_description:\nPhotorealistic live-action footage with natural skin texture, true-to-life "
                  "colors and crisp detail, in a clean frame with no subtitles, captions, overlay text, watermarks or "
                  "logos. [Shot 1] One continuous shot: " + line + where + " The face of <Subject 1> is clear and "
                  "keeps the exact facial identity of <Picture 1>. All motion plays at natural real-time speed.\n\n"
                  "overall_soundscape:\nNatural ambient sound of the place that matches the action.\n\n"
                  "non_diegetic_music:\nNone.")
        segs.append({"length": length, "overlapIn": 0, "prompt": prompt, "images": ["img1"], "videos": [],
                     "audios": [], "audioMode": "none"})
    plan = {"version": 1, "fps": FPS, "width": width, "height": height,
            "globalPrompt": HIRES_TRIGGER, "globalMode": "prepend",
            "media": {"images": [{"id": "img1", "file": image_file}], "videos": [], "audios": [], "refmods": []},
            "segments": segs, "refVideoHeight": "auto", "idRef": "img1"}
    if notes:
        plan[NOTES_KEY] = notes
    return plan


# ------------------------------------------------------------------------------------------------------- size
ASPECTS = ["16:9", "9:16", "1:1", "4:3", "3:4", "follow video"]


def size_for(aspect: str, megapixels: float, video_wh=None) -> tuple:
    """Width / height (multiples of 64) for an aspect and a pixel budget; 'follow video' uses the guide's ratio."""
    if aspect == "follow video" and video_wh and video_wh[0] and video_wh[1]:
        ar = float(video_wh[0]) / float(video_wh[1])
    else:
        a, b = (aspect if aspect != "follow video" else "9:16").split(":")
        ar = float(a) / float(b)
    px = min(MAX_MEGAPIXELS, max(0.1, float(megapixels))) * 1_000_000
    w = max(64, round(math.sqrt(px * ar) / 64) * 64)
    h = max(64, round(math.sqrt(px / ar) / 64) * 64)
    return int(w), int(h)
