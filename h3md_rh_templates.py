"""Plan templates for the RunningHub apps (H3MD RH).

Each template returns an H3MD plan v1 dict (see h3md_rh_core/h3md_director.py).  Users only fill a few fields
(who, scene, music start, shot lines); everything else is fixed here, tested on RH / locally (2026-11-06).

  mv            photo + song -> 8 hard-cut shots of 243 frames (10.125 s), medium close-ups / close-ups,
                the song locked as soundtrack, each shot at start + 10.125 * i seconds of the song
  motion_copy   guide video + face photo + full-body photo -> one long shot that copies the guide's motion
                (TARA: global prompt "motion_transfer:", guide = <Video 1>, its soundtrack locked).  The Director
                rounds the length down to 5+17n so it never runs past the clip end; autosplit cuts it into parts.
  story         photo + 1..8 shot lines -> one hard-cut shot of 243 frames per line
"""
from __future__ import annotations

import math
import re

FPS = 24
SHOT = 243                      # 10.125 s, 5 + 17 * 14
SHOT_SEC = SHOT / FPS
HIRES_TRIGGER = "r64, r34l1sm"  # high-res LoRA trigger words used by the RH workflows

TEMPLATES = ["mv", "motion_copy", "story", "custom"]

_CLEAN = re.compile(r'[\r\n\\"{}]+')


def clean(text: str, fallback: str = "") -> str:
    """User text -> one safe line (no newlines, quotes or braces that would break the prompt sections)."""
    t = _CLEAN.sub(" ", str(text or "")).strip()
    t = re.sub(r"\s{2,}", " ", t)
    return t or fallback


def frames_5_17n_down(frames: float) -> int:
    n = max(0, math.floor((float(frames) - 5) / 17))
    return 5 + 17 * n


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


def mv(who: str, scene: str, image_file: str, audio_file: str, music_start: float = 0.0,
       width: int = 1344, height: int = 768) -> dict:
    who = clean(who, "a young woman")
    scene = clean(scene, "a sunny beach")
    segs = []
    for i, (summ, desc) in enumerate(MV_SHOTS):
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
        offset = round(float(music_start) + SHOT_SEC * i, 3)
        segs.append({"length": SHOT, "overlapIn": 0, "prompt": prompt, "images": ["img1"], "videos": [],
                     "audios": [{"id": "aud1", "mode": "locked", "offset": offset}], "audioMode": "reference"})
    return {"version": 1, "fps": FPS, "width": width, "height": height,
            "globalPrompt": HIRES_TRIGGER, "globalMode": "prepend",
            "media": {"images": [{"id": "img1", "file": image_file}], "videos": [],
                      "audios": [{"id": "aud1", "file": audio_file}], "refmods": []},
            "segments": segs, "refVideoHeight": "auto", "idRef": "img1"}


# ------------------------------------------------------------------------------------------------- motion copy
def motion_copy(who: str, scene: str, face_file: str, body_file: str | None, video_file: str,
                video_seconds: float, max_seconds: float = 0.0, width: int = 704, height: int = 1024) -> dict:
    """One long TARA shot.  max_seconds > 0 limits how much of the guide is used (cost control)."""
    who = clean(who, "a young woman with long dark hair, wearing a simple fitted dress")
    scene = clean(scene, "a cozy living room at night, lit by warm, soft lamps")
    use = float(video_seconds)
    if max_seconds and max_seconds > 0:
        use = min(use, float(max_seconds))
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
            "refVideoHeight": "auto"}


# ------------------------------------------------------------------------------------------------------- story
def story(who: str, scene: str, shots: str, image_file: str, width: int = 1344, height: int = 768) -> dict:
    """One hard-cut 243-frame shot per non-empty line of `shots` (max 8)."""
    who = clean(who, "a young woman")
    scene = clean(scene, "")
    lines = [clean(x) for x in str(shots or "").splitlines()]
    lines = [x for x in lines if x][:8]
    if not lines:
        raise ValueError("H3MD RH story: write at least one shot line (one line per shot). "
                         "故事模板：请至少写一行镜头描述（一行一个镜头）。")
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
        segs.append({"length": SHOT, "overlapIn": 0, "prompt": prompt, "images": ["img1"], "videos": [],
                     "audios": [], "audioMode": "none"})
    return {"version": 1, "fps": FPS, "width": width, "height": height,
            "globalPrompt": HIRES_TRIGGER, "globalMode": "prepend",
            "media": {"images": [{"id": "img1", "file": image_file}], "videos": [], "audios": [], "refmods": []},
            "segments": segs, "refVideoHeight": "auto", "idRef": "img1"}


# ------------------------------------------------------------------------------------------------------- size
ASPECTS = ["16:9", "9:16", "1:1", "4:3", "3:4", "follow video"]


def size_for(aspect: str, megapixels: float, video_wh=None) -> tuple:
    """Width / height (multiples of 64) for an aspect and a pixel budget; 'follow video' uses the guide's ratio."""
    if aspect == "follow video" and video_wh and video_wh[0] and video_wh[1]:
        ar = float(video_wh[0]) / float(video_wh[1])
    else:
        a, b = (aspect if aspect != "follow video" else "9:16").split(":")
        ar = float(a) / float(b)
    px = max(0.1, float(megapixels)) * 1_000_000
    w = max(64, round(math.sqrt(px * ar) / 64) * 64)
    h = max(64, round(math.sqrt(px / ar) / 64) * 64)
    return int(w), int(h)
